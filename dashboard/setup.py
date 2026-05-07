# save as run_setup.py in your dashboard/ folder, then run: python run_setup.py

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Lavanya M\Downloads\tensile-method-459009-k2-83ba5f9806a9.json"

from google.cloud import storage

PROJECT = "tensile-method-459009-k2"
BUCKET  = f"ml-scheduler-jobs-{PROJECT}"
REGION  = "us-central1"

client = storage.Client(project=PROJECT)

# Create bucket
try:
    b = client.create_bucket(BUCKET, location=REGION)
    b.iam_configuration.uniform_bucket_level_access_enabled = True
    b.patch()
    print(f"✓ Bucket created: gs://{BUCKET}")
except Exception as e:
    print(f"  Bucket exists or error: {e}")

bucket = client.bucket(BUCKET)

# Upload files
files = {
    "trainer/train.py":   "trainer/train.py",
    "trainer/startup.sh": "trainer/startup.sh",
}
for local, gcs_path in files.items():
    try:
        bucket.blob(gcs_path).upload_from_filename(local)
        print(f"✓ Uploaded {local} → gs://{BUCKET}/{gcs_path}")
    except FileNotFoundError:
        print(f"✗ Not found locally: {local} — make sure you're running from dashboard/")

print(f"\n✓ Done. Add to .env:")
print(f"  CHECKPOINT_GCS_BUCKET={BUCKET}")