// TruthLens content script: finds the photo / video the user pointed at, and shows the result panel in the page.
// Instagram, X, Facebook… cover media with transparent overlay <div>s, so a normal "right-click → image" menu
// never appears. We remember where the user right-clicked (or clicked, in pick mode) and look *through* the
// overlays with elementsFromPoint() for the media element underneath.
(() => {
  if (window.__truthlens) return;
  window.__truthlens = true;
  let last = { x: 0, y: 0 };
  addEventListener("contextmenu", e => { last = { x: e.clientX, y: e.clientY }; }, true);

  function mediaAt(x, y) {
    const els = document.elementsFromPoint(x, y);
    let m = els.find(e => e.tagName === "VIDEO") || els.find(e => e.tagName === "IMG" && e.naturalWidth > 64);
    if (!m) {   // some sites draw images as CSS backgrounds
      const bg = els.find(e => /url\(/.test(getComputedStyle(e).backgroundImage));
      if (bg) return describe(bg, getComputedStyle(bg).backgroundImage.match(/url\(["']?(.*?)["']?\)/)[1], "image");
      return null;
    }
    if (m.tagName === "VIDEO") {
      const src = m.currentSrc || m.src || (m.querySelector("source") || {}).src || "";
      return describe(m, src, "video");
    }
    return describe(m, m.currentSrc || m.src, "image");
  }

  function describe(el, src, kind) {
    const r = el.getBoundingClientRect();
    return { kind, src, rect: { x: r.left, y: r.top, w: r.width, h: r.height }, dpr: devicePixelRatio, vw: innerWidth,
             page: location.href, title: (document.title || "").slice(0, 200) };
  }

  // ---- pick mode (from the toolbar popup): hover outlines media, click selects it ----
  let picking = false, box;
  function startPick() {
    if (picking) return;
    picking = true;
    box = document.createElement("div");
    box.style.cssText = "position:fixed;z-index:2147483646;pointer-events:none;border:2px solid #e5484d;border-radius:6px;" +
      "background:rgba(229,72,77,.08);transition:all .08s;display:none";
    document.documentElement.appendChild(box);
    addEventListener("mousemove", onMove, true);
    addEventListener("click", onPick, true);
    addEventListener("keydown", onKey, true);
    panel().show("Click the photo or video you want to check · Esc to cancel", null);
  }
  function stopPick() {
    picking = false;
    box && box.remove();
    removeEventListener("mousemove", onMove, true);
    removeEventListener("click", onPick, true);
    removeEventListener("keydown", onKey, true);
  }
  function onMove(e) {
    const m = mediaAt(e.clientX, e.clientY);
    if (!m) { box.style.display = "none"; return; }
    Object.assign(box.style, { display: "block", left: m.rect.x + "px", top: m.rect.y + "px",
                               width: m.rect.w + "px", height: m.rect.h + "px" });
  }
  function onPick(e) {
    const m = mediaAt(e.clientX, e.clientY);
    if (!m) return;
    e.preventDefault(); e.stopPropagation();
    stopPick();
    chrome.runtime.sendMessage({ type: "analyze", media: m });
  }
  function onKey(e) { if (e.key === "Escape") { stopPick(); panel().hide(); } }

  // ---- result panel (shadow DOM so the site's CSS can't touch it) ----
  let P;
  function panel() {
    if (P) return P;
    const host = document.createElement("div");
    host.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:2147483647";
    const sh = host.attachShadow({ mode: "open" });
    sh.innerHTML = `<style>
      .p{width:320px;max-width:calc(100vw - 32px);background:#fff;color:#0d0d0f;border:1px solid #dcdfe3;border-radius:10px;
         box-shadow:0 10px 30px rgba(0,0,0,.18);font:14px/1.45 -apple-system,Inter,system-ui,sans-serif;padding:14px 16px}
      .h{display:flex;justify-content:space-between;align-items:center;font:600 12px ui-monospace,Menlo,monospace;letter-spacing:.06em;color:#6b7179}
      .x{border:0;background:none;font-size:18px;cursor:pointer;color:#6b7179}
      .bar{height:4px;background:#eef0f2;border-radius:2px;margin:10px 0;overflow:hidden}.bar i{display:block;height:100%;background:#0d0d0f;width:0;transition:width .4s}
      .v{display:flex;align-items:baseline;gap:8px;margin-top:6px}.v b{font-size:30px;font-weight:800}.d{width:9px;height:9px;border-radius:50%;display:inline-block}
      h3{font-size:15px;margin:6px 0 2px}p{margin:4px 0;color:#4a5058;font-size:13px}.m{color:#6b7179;font-size:12px}
      a{display:block;margin-top:10px;text-align:center;background:#0d0d0f;color:#fff;border-radius:8px;padding:9px;text-decoration:none;font-weight:600}
    </style><div class="p"><div class="h">TRUTHLENS<button class="x" title="Close">×</button></div><div class="b"></div></div>`;
    sh.querySelector(".x").onclick = () => P.hide();
    const body = sh.querySelector(".b");
    const attach = () => { if (!host.isConnected) document.documentElement.appendChild(host); };
    P = {
      show(msg, pct) {
        attach();
        body.innerHTML = `<p>${esc(msg)}</p>` + (pct == null ? "" : `<div class="bar"><i style="width:${pct}%"></i></div>`);
      },
      result(r, server) {
        attach();
        const col = { red: "#e5484d", orange: "#f76b15", amber: "#d99a00", green: "#2a9d63" }[r.color] || "#8a9099";
        body.innerHTML = `<div class="v"><b>${r.likelihood}%</b><span><i class="d" style="background:${col}"></i> ${esc(r.band)}</span></div>
          <div class="m">confidence ${esc(r.confidence)}</div><h3>${esc(r.headline)}</h3><p>${esc(r.sub)}</p>
          ${r.files.map(f => `<p><b>AI-made?</b> ${esc(f.ai)}</p>`).join("")}
          ${r.note ? `<p class="m">${esc(r.note)}</p>` : ""}
          <a href="${esc(server + r.report)}" target="_blank" rel="noopener">Open full report →</a>`;
      },
      hide() { host.remove(); }
    };
    return P;
  }
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  chrome.runtime.onMessage.addListener((msg, _s, reply) => {
    if (msg.type === "whatsAtClick") reply(mediaAt(last.x, last.y));
    else if (msg.type === "pick") startPick();
    else if (msg.type === "status") panel().show(msg.text, msg.pct);
    else if (msg.type === "result") panel().result(msg.result, msg.server);
  });
})();
