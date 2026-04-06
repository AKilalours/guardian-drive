"""
Run this once to patch server/app.py with CORS middleware.
python server/app_patch.py
"""
import re
from pathlib import Path

app_path = Path("server/app.py")
txt = app_path.read_text()

# Add CORS middleware import if not present
cors_import = "from fastapi.middleware.cors import CORSMiddleware"
if cors_import not in txt:
    txt = txt.replace(
        "from fastapi import FastAPI",
        "from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware"
    )
    print("✓ Added CORSMiddleware import")

# Add CORS setup after app = FastAPI(...)
cors_setup = """
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
if "CORSMiddleware" not in txt or "allow_origins" not in txt:
    txt = re.sub(
        r'(app = FastAPI\([^)]*\))',
        r'\1' + cors_setup,
        txt,
        count=1
    )
    print("✓ Added CORS middleware setup")

app_path.write_text(txt)
print("✓ server/app.py patched — restart server now")
