const srv = document.getElementById("srv"), st = document.getElementById("st");
chrome.storage.sync.get({ server: "http://localhost:8000" }, v => { srv.value = v.server; });

document.getElementById("save").onclick = () => {
  const v = srv.value.trim().replace(/\/+$/, "");
  if (!/^https?:\/\//.test(v)) { st.textContent = "Enter a full http(s) address."; return; }
  chrome.storage.sync.set({ server: v }, () => { st.textContent = "Saved."; });
};

document.getElementById("test").onclick = async () => {
  st.textContent = "Testing…";
  try {
    const r = await fetch(srv.value.trim().replace(/\/+$/, "") + "/api/samples", { credentials: "include" });
    st.textContent = r.ok ? "✓ TruthLens is reachable." : r.status === 401 ? "Reachable, but needs a password: open it in a tab and sign in." : `Server answered ${r.status}.`;
  } catch (e) { st.textContent = "✗ Not reachable. Is the server running?"; }
};

document.getElementById("pick").onclick = async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try { await chrome.tabs.sendMessage(tab.id, { type: "pick" }); }
  catch (e) {   // tab opened before the extension was installed: inject now
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    await chrome.tabs.sendMessage(tab.id, { type: "pick" });
  }
  window.close();
};
