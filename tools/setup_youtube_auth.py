import os
import sys
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"


def main():
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"Error: {CREDENTIALS_FILE} not found.")
        print("Steps:")
        print("  1. Go to console.cloud.google.com")
        print("  2. Create a project, enable YouTube Data API v3")
        print("  3. Create OAuth 2.0 credentials (Desktop app)")
        print("  4. Download and save as credentials.json in project root")
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"Auth complete. Token saved to {TOKEN_FILE}")


if __name__ == "__main__":
    main()
