import argparse, csv, time, random, os, json
from datetime import datetime
from dateutil import parser as dtparse
from google.api_core.exceptions import ResourceExhausted, RetryError, ServiceUnavailable, DeadlineExceeded
from firebase_admin import credentials, firestore, initialize_app

# --------- CONFIG you can tweak ----------
CSV_PATH    = "ncr_ride_bookings.csv"  # your CSV
SERVICE_KEY = "agile-processes---uber-firebase-adminsdk-fbsvc-210cbe6eb9.json"
COLLECTION  = "rides"
DOC_ID_FIELD = "ride_id"   # set to None to auto-ID
BATCH_SIZE  = 100          # Firestore max = 500; smaller = gentler on quotas
MAX_RETRIES = 8
SLEEP_BETWEEN_BATCHES = 0.25  # seconds
CHECKPOINT_FILE = ".firestore_upload_checkpoint.json"
# Optional typing hints — adjust as needed:
NUMERIC_FIELDS = {"fare","distance_km","duration_min"}
TIMESTAMP_FIELDS = {"ts","pickup_time","dropoff_time"}
# -----------------------------------------

def to_number(v):
    if v is None or v == "": return None
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return None

def to_timestamp(v):
    if not v: return None
    try:
        return dtparse.parse(v)
    except Exception:
        return None

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

def commit_with_retry(batch, label=""):
    delay = 0.5
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            batch.commit()
            if label:
                print(f"✔ {label}")
            return
        except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded) as e:
            jitter = random.uniform(0, 0.5)
            sleep_for = min(30, delay) + jitter
            print(f"[{label}] quota/backoff attempt {attempt}: sleeping {sleep_for:.1f}s ({type(e).__name__})")
            time.sleep(sleep_for)
            delay *= 2
        except RetryError:
            print(f"[{label}] retry timeout; backing off…")
            time.sleep(min(30, delay))
            delay *= 2
    raise RuntimeError("Gave up committing after retries.")

def save_checkpoint(last_row_idx):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_row_idx": last_row_idx}, f)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("last_row_idx")
    except Exception:
        return None

def main(start, limit, resume):
    # Init Firebase
    cred = credentials.Certificate(SERVICE_KEY)
    initialize_app(cred)
    db = firestore.client()

    # Determine start from checkpoint if resume
    if resume:
        cp = load_checkpoint()
        if cp is not None:
            print(f"Resuming from checkpoint at row index {cp+1}")
            start = cp + 1

    total_written = 0
    in_batch = 0
    batch = db.batch()
    last_row_idx = -1  # 0-based across data rows (excludes header)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Skip to start
        for _ in range(start):
            try:
                next(reader)
            except StopIteration:
                print("Reached end of file before start offset.")
                return

        for i, row in enumerate(reader):
            # Enforce limit window
            if limit is not None and i >= limit:
                break

            data = cast_row(row)
            if DOC_ID_FIELD and data.get(DOC_ID_FIELD):
                doc_ref = db.collection(COLLECTION).document(str(data[DOC_ID_FIELD]))
            else:
                doc_ref = db.collection(COLLECTION).document()
            batch.set(doc_ref, data)

            in_batch += 1
            total_written += 1
            last_row_idx = start + i

            if in_batch >= BATCH_SIZE:
                commit_with_retry(batch, label=f"Committed {total_written} rows so far")
                time.sleep(SLEEP_BETWEEN_BATCHES)
                batch = db.batch()
                in_batch = 0
                save_checkpoint(last_row_idx)

    if in_batch:
        commit_with_retry(batch, label=f"Committed {total_written} rows total")
        save_checkpoint(last_row_idx)

    print(f"✅ Done. Wrote {total_written} rows in this run. Checkpoint at row index {last_row_idx}.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunked CSV → Firestore uploader")
    ap.add_argument("--start", type=int, default=0, help="0-based data-row offset to start from (excludes header)")
    ap.add_argument("--limit", type=int, default=5000, help="max rows to upload this run (None for all)")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint")
    args = ap.parse_args()
    main(args.start, None if str(args.limit).lower()=="none" else args.limit, args.resume)
