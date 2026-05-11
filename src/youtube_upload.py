"""Upload a finished reel to YouTube Shorts via the official Data API v3.

First-time setup:
  1. Create a Google Cloud project, enable "YouTube Data API v3".
  2. OAuth consent screen -> add yourself as a test user.
  3. Download client_secret.json -> place at credentials/client_secret.json
  4. First run will open a browser to authorize and write yt_token.json.
"""
from __future__ import annotations

from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from .utils import env

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _get_service():
    secrets = Path(env("YT_CLIENT_SECRETS", "credentials/client_secret.json"))
    token = Path(env("YT_TOKEN_FILE", "credentials/yt_token.json"))

    creds: Credentials | None = None
    if token.exists():
        creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not secrets.exists():
                raise FileNotFoundError(
                    f"Missing OAuth client at {secrets}. See module docstring."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
            creds = flow.run_local_server(port=0)
        token.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload_short(
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    cfg: dict,
    privacy: str | None = None,
) -> str:
    """Returns the new video's YouTube ID."""
    yt = _get_service()
    # Ensure "#Shorts" appears in title or description so YT treats it as Shorts.
    if "#shorts" not in description.lower() and "#shorts" not in title.lower():
        description = description + "\n\n#Shorts"

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:20],
            "categoryId": cfg["upload"]["youtube"].get("category_id", "22"),
        },
        "status": {
            "privacyStatus": privacy or env("YT_DEFAULT_PRIVACY", "private"),
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"[yt] upload {int(status.progress() * 100)}%")
    return resp["id"]
