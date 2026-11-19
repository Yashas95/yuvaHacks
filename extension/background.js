// TruthLens extension service worker: gets the media the user picked, sends it to the TruthLens server,
// follows the job, and hands the plain-English verdict back to the page.
//
// How the media is obtained, best first:
//   1. Original file   — fetch the element's http(s) URL (images, most non-streaming videos). Keeps metadata.
//   2. Screen capture  — for streamed reels (blob: / MediaSource URLs there is no file to download) we crop the
//                        element out of a screenshot of the visible tab. Only the pixels survive: no metadata,
//                        no audio, one frame. The result says so.
const DEFAULT_SERVER = "http://localhost:8000";
const MAX_BYTES = 80 * 1024 * 1024;

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "tl-check", title: "Check with TruthLens", contexts: ["all"] });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "tl-check" || !tab?.id) return;
  let media = null;
  try { media = await chrome.tabs.sendMessage(tab.id, { type: "whatsAtClick" }); } catch (e) { /* page without content script */ }
  if (!media && info.srcUrl) media = { kind: info.mediaType === "video" ? "video" : "image", src: info.srcUrl, rect: null, page: info.pageUrl };
  if (!media) return notify(tab.id, "No photo or video found there. Try the toolbar button → “Pick media on this page”.");
  analyze(tab, media);
});

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg.type === "analyze" && sender.tab) analyze(sender.tab, msg.media);
});

async function server() {
  const { server } = await chrome.storage.sync.get({ server: DEFAULT_SERVER });
  return server.replace(/\/+$/, "");
}

function notify(tabId, text, pct = null) {
  chrome.tabs.sendMessage(tabId, { type: "status", text, pct }).catch(() => {});
}

async function getFile(tab, media) {
  if (/^https?:/.test(media.src || "")) {
    try {
      const r = await fetch(media.src, { credentials: "omit" });
      const len = +r.headers.get("content-length") || 0;
      if (r.ok && len <= MAX_BYTES) {
        const blob = await r.blob();
        const type = blob.type || "";
        if (blob.size && blob.size <= MAX_BYTES && /^(image|video|audio)\//.test(type)) {
          const ext = (type.split("/")[1] || "bin").split(";")[0].replace("jpeg", "jpg").replace("quicktime", "mov");
          return { blob, name: `${media.kind}-from-page.${ext}`, how: "original" };
        }
      }
    } catch (e) { /* CORS / network: fall back to a screen capture */ }
  }
  if (!media.rect || media.rect.w < 32 || media.rect.h < 32) throw new Error("This media can't be downloaded and isn't visible on screen to capture.");
  const shot = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
  const bmp = await createImageBitmap(await (await fetch(shot)).blob());
  const k = bmp.width / (media.vw || bmp.width / (media.dpr || 1));   // screenshot px per CSS px
  const x = Math.max(0, Math.round(media.rect.x * k)), y = Math.max(0, Math.round(media.rect.y * k));
  const w = Math.min(bmp.width - x, Math.round(media.rect.w * k)), h = Math.min(bmp.height - y, Math.round(media.rect.h * k));
  const c = new OffscreenCanvas(w, h);
  c.getContext("2d").drawImage(bmp, x, y, w, h, 0, 0, w, h);
  return { blob: await c.convertToBlob({ type: "image/png" }), name: `${media.kind}-screen-capture.png`, how: "capture" };
}

async function analyze(tab, media) {
  const base = await server();
  let file;
  try { file = await getFile(tab, media); }          // capture before our own panel appears on the page
  catch (e) { return notify(tab.id, e.message); }
  notify(tab.id, file.how === "original" ? "Sending the original file to TruthLens…" : "Streamed reel: sending a screen capture of the frame…", 5);
  try {
    const fd = new FormData();
    fd.append("files", file.blob, file.name);
    fd.append("context", "");
    const r = await fetch(base + "/api/analyze", { method: "POST", body: fd, credentials: "include" });
    if (r.status === 401) throw new Error("This TruthLens server needs a password: open it in a tab and sign in first.");
    if (!r.ok) throw new Error(`Server said ${r.status}: ${(await r.text()).slice(0, 160)}`);
    const { job, case_id } = await r.json();
    for (;;) {
      await new Promise(res => setTimeout(res, 1000));
      const s = await (await fetch(`${base}/api/status/${job}`, { credentials: "include" })).json();
      if (s.status === "error") throw new Error(s.message);
      if (s.status === "done") break;
      notify(tab.id, s.message || "Analyzing…", s.progress || 5);
    }
    const result = await (await fetch(`${base}/api/summary/${case_id}`, { credentials: "include" })).json();
    if (file.how === "capture")
      result.note = "Checked a screen capture of one frame (the site streams this video, so the original file isn't available). " +
                    "Metadata and audio checks could not run — download the original for a full check.";
    chrome.tabs.sendMessage(tab.id, { type: "result", result, server: base }).catch(() => {});
  } catch (e) {
    notify(tab.id, `Couldn't reach TruthLens at ${base}: ${e.message}`);
  }
}
