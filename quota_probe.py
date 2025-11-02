from firebase_admin import credentials, firestore, initialize_app
from google.api_core.exceptions import ResourceExhausted

SERVICE_KEY = "agile-processes---uber-firebase-adminsdk-fbsvc-210cbe6eb9.json"

try:
    initialize_app(credentials.Certificate(SERVICE_KEY))
    db = firestore.client()
    db.collection("quota_probe").document("now").set({"ts": firestore.SERVER_TIMESTAMP})
    print("✅ A single write succeeded — you’re under cap (just throttled).")
except ResourceExhausted as e:
    print("❌ ResourceExhausted — very likely at daily cap for writes.")
    print(e)
