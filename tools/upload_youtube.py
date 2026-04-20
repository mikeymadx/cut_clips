import csv
import sys
from pathlib import Path

import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TOKEN_FILE = "token.json"
OUTPUT_DIR = Path(".tmp/clips")
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

BASE_TAGS = ["Holdfast", "holdfast band", "indie music", "music video", "holdfast music"]

DESCRIPTION = """{title}

Holdfast 🎸

Follow Holdfast:
🌐 https://holdfast.band
📸 Instagram: @holdfastbandco

#Holdfast #HoldfastBand #IndieMusic #MusicVideo
"""

DESCRIPTION_SHORTS = """{title}

Holdfast 🎸

Follow Holdfast:
🌐 https://holdfast.band
📸 Instagram: @holdfastbandco

#Holdfast #HoldfastBand #IndieMusic #MusicVideo #Shorts
"""


def load_youtube():
    import os
    if not os.path.exists(TOKEN_FILE):
        print(f"Error: {TOKEN_FILE} not found. Run: python tools/setup_youtube_auth.py")
        sys.exit(1)
    creds = google.oauth2.credentials.Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    return build("youtube", "v3", credentials=creds)


def make_title(cut_title, custom_title):
    if custom_title and custom_title.strip():
        return custom_title.strip()
    return cut_title.replace("_", " ").replace("-", " ").title() + " | Holdfast"


def make_tags(cut_title, custom_tags):
    if custom_tags and custom_tags.strip():
        return [t.strip() for t in custom_tags.split(",")]
    words = [w.capitalize() for w in cut_title.replace("-", "_").split("_") if w]
    return BASE_TAGS + words


def upload(youtube, file_path, title, description, tags):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "10",  # Music
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(file_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    print(f"[UPLOAD] {file_path.name}")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"         {int(status.progress() * 100)}%", end="\r")

    video_id = response["id"]
    print(f"[DONE]   https://www.youtube.com/watch?v={video_id}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/upload_youtube.py clips.csv")
        sys.exit(1)

    youtube = load_youtube()

    with open(sys.argv[1], newline="") as f:
        for row in csv.DictReader(f):
            if row.get("approved", "").strip().lower() != "true":
                continue

            initial = row["initial_video_title"].strip()
            cut = row["cut_title"].strip()
            title = make_title(cut, row.get("title", ""))
            tags = make_tags(cut, row.get("tags", ""))
            custom_desc = row.get("description", "").strip()

            h_path = OUTPUT_DIR / f"{initial}-{cut}-horizontal.mp4"
            v_path = OUTPUT_DIR / f"{initial}-{cut}-vertical.mp4"

            if h_path.exists():
                desc = custom_desc or DESCRIPTION.format(title=title)
                upload(youtube, h_path, title, desc, tags)
            else:
                print(f"[SKIP] {h_path.name} not found — run cut_clips.py first")

            if v_path.exists():
                desc = custom_desc or DESCRIPTION_SHORTS.format(title=title)
                upload(youtube, v_path, f"{title} #Shorts", desc, tags)
            else:
                print(f"[SKIP] {v_path.name} not found — run cut_clips.py first")


if __name__ == "__main__":
    main()
