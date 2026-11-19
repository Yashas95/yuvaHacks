"""Create / update the TruthLens Hugging Face Space (Docker SDK, free CPU hardware).

One-time:  .venv/bin/hf auth login            (paste a *write* token from huggingface.co/settings/tokens yourself)
Deploy:    .venv/bin/python tools/deploy_hf_space.py <hf-username> [space-name]
Optional:  TRUTHLENS_PASSWORD=... .venv/bin/python tools/deploy_hf_space.py <hf-username>
           stores the demo password as a private Space secret (visitors must enter it).

Uploads the same files as the GitHub repository (personal media in cases/ and demo_samples/ is never included)
and prepends the Space configuration header to README.md. The Space rebuilds automatically after each deploy.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
IGNORE = [".venv/*", "cases/*", "demo_samples/*", ".git/*", "**/__pycache__/*", "*.pyc", ".*.log", ".claude/*",
          "README.md", ".DS_Store"]
HEADER = """---
title: TruthLens
emoji: 🔍
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Explainable multimodal forensics & deepfake detection
---

"""


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    user = sys.argv[1]
    repo = f"{user}/{sys.argv[2] if len(sys.argv) > 2 else 'truthlens'}"
    api = HfApi()
    who = api.whoami()["name"]                      # fails clearly if not logged in
    print(f"logged in as {who}; deploying to spaces/{repo}")
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    pw = os.environ.get("TRUTHLENS_PASSWORD")
    if pw:
        api.add_space_secret(repo, "TRUTHLENS_PASSWORD", pw)
        print("password stored as a Space secret")
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(ROOT), ignore_patterns=IGNORE,
                      commit_message="Deploy TruthLens")
    readme = HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo, repo_type="space",
                    commit_message="Space README")
    print(f"done → https://huggingface.co/spaces/{repo}  (first build takes ~10-20 min: torch + models)")


if __name__ == "__main__":
    main()
