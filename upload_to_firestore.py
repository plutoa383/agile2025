import csv
from datetime import datetime
from dateutil import parser as dtparse
from firebase_admin import credentials, firestore, initialize_app

# ---- CONFIG ----
CSV_PATH    = "ncr_ride_bookings.csv"
SERVICE_KEY = "agile-processes---uber-firebase-adminsdk-fbsvc-210cbe6eb9.json"
COLLECTION  = "rides"
DOC_ID_FIELD = "ride_id"   # or None if no unique id
BATCH_SIZE = 450                      # Firestore limit is 500

# ---- INIT ----
cred = credentials.Certificate(SERVICE_KEY)
initialize_app(cred)
db = firestore.client()

def to_number(v):
    if v is None or v == "":
        return None
    try:
        if "." in v: return float(v)
        return int(v)
    except ValueError:
        return None

def to_timestamp(v):
    if not v: return None
    try:
        return dtparse.parse(v)       # returns a datetime; Admin SDK stores as Timestamp
    except Exception:
        return None

# Map certain columns to types if you want
NUMERIC_FIELDS = {"fare", "distance_km", "duration_min"}
TIMESTAMP_FIELDS = {"ts", "pickup_time", "dropoff_time"}

def cast_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        v = v.strip() if isinstance(v, str) else v
        if k in NUMERIC_FIELDS:
            out[k] = to_number(v)
        elif k in TIMESTAMP_FIELDS:
            out[k] = to_timestamp(v)
        else:
            out[k] = v if v != "" else None
    return out

def main():
    batch = db.batch()
    count, in_batch = 0, 0

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = cast_row(row)
            if DOC_ID_FIELD and data.get(DOC_ID_FIELD):
                doc_ref = db.collection(COLLECTION).document(str(data[DOC_ID_FIELD]))
            else:
                doc_ref = db.collection(COLLECTION).document()  # auto id

            batch.set(doc_ref, data)
            in_batch += 1
            count += 1

            if in_batch >= BATCH_SIZE:
                batch.commit()
                batch = db.batch()
                in_batch = 0

    if in_batch:
        batch.commit()

    print(f"Uploaded {count} rows to collection '{COLLECTION}'.")

if __name__ == "__main__":
    main()
