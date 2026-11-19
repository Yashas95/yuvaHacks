# TruthLens Chrome extension

Check a photo, post or reel **where you see it**: on Instagram, X, Facebook, news sites, anywhere. You don't need
to download it first. The verdict appears in a small panel on the same page, with a link to the full report.

## Install (developer mode, about 1 minute)

**Easiest:** on any running TruthLens site, click **Chrome extension → Download extension (.zip)** (or open
`<server>/extension.zip`). That zip is already set to use the server you downloaded it from, so you can skip step 3.

1. Unzip it (or use this `extension/` folder from the repository). Open `chrome://extensions` and turn on **Developer mode** (top right).
2. Click **Load unpacked** and choose the unzipped `truthlens-extension` folder (or `extension/`).
3. Click the TruthLens toolbar icon. Under **TruthLens server**, enter your server address, then click **Save**
   and **Test connection**:
   - local: `http://localhost:8000` (run `.venv/bin/python app.py`)
   - hosted demo: your Codespaces link, e.g. `https://<codespace>-7860.app.github.dev`. Open that link once in a
     normal tab first, so GitHub's port notice is accepted.

## Use

- **Right-click** the photo or video → **Check with TruthLens**, or
- click the toolbar icon → **Pick media on this page**, then click the post (Esc cancels).

Instagram and similar sites cover media with invisible layers, so a normal "right-click on image" menu never
appears. The extension remembers where you clicked and looks through those layers for the photo or video
underneath (`document.elementsFromPoint`).

## What gets analyzed (honest limits)

| Media on the page | What is sent | What can be checked |
|---|---|---|
| Photo / video with a normal file URL | the **original file**, downloaded by the extension | everything: metadata, pixels, frames, audio |
| Streamed reel (`blob:` / MediaSource, e.g. Instagram reels) | a **screen capture of the visible frame** (there is no file to download) | pixel and AI-image checks on that frame only, with no metadata or audio. The panel says so |

For a full check of a streamed reel, download the original and upload it on the TruthLens page.

## Privacy

Media goes only to the TruthLens server **you** configure. The extension has no analytics and contacts no other
service. It needs `<all_urls>` access for two things: to download the media you picked from the site's CDN, and to
show the result panel on any site.

## Files

`manifest.json` (Manifest V3) · `background.js` (fetch or screen capture, upload, progress, result) ·
`content.js` (finds the media under the click, pick mode, result panel in a shadow DOM) · `popup.html/js` (server
setting, pick button).
