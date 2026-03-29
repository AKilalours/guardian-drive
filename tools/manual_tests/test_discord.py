import os, requests

url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
assert url, "Missing DISCORD_WEBHOOK_URL"

payload = {"content": "Test: GuardianDrive webhook is working ✅"}
r = requests.post(url, json=payload, timeout=4)
print("status:", r.status_code)
print("body:", r.text[:200])
