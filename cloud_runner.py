import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

try:
    from google.cloud import storage
    from google.cloud import logging as cloud_logging
    from google.cloud import firestore
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Cloud Logging if available
if GCS_AVAILABLE:
    try:
        logging_client = cloud_logging.Client()
        log_name = os.environ.get("CLOUD_LOG_NAME", "python") # Default to 'python' (default handler behavior) or custom
        logging_handler = logging_client.get_default_handler(name=log_name)
        logger.addHandler(logging_handler)
    except Exception as e:
        logger.warning(f"Could not set up Cloud Logging: {e}")

def parse_gcs_path(path):
    """Parses gs://bucket/blob_path into (bucket, blob_path)."""
    if not path.startswith("gs://"):
        return None, None
    parts = path[5:].split("/", 1)
    if len(parts) < 2:
        return parts[0], ""
    return parts[0], parts[1]

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage not installed.")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded {source_blob_name} to {destination_file_name}")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage not installed.")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(f"Uploaded {source_file_name} to {destination_blob_name}")

def update_firestore(doc_ref, data):
    """Updates Firestore document with progress data."""
    if not doc_ref:
        return
    try:
        doc_ref.set(data, merge=True)
    except Exception as e:
        logger.warning(f"Failed to update Firestore: {e}")

def run_caustic_design(executable_path, args, temp_dir, firestore_doc=None):
    """
    Runs the caustic_design binary and streams stdout to parse JSON progress.
    """
    cmd = [executable_path]
    
    # Process arguments to handle local vs GCS paths
    processed_args = []
    
    # Track inputs to download
    inputs_map = {
        "-in_trg": "target.png",
        "-in_src": "source.png" 
    }
    
    # Track output to upload
    output_gcs_path = None
    local_output_path = os.path.join(temp_dir, "output.obj")
    
    i = 0
    while i < len(args):
        key = args[i]
        val = args[i+1] if i + 1 < len(args) and not args[i+1].startswith("-") else None
        
        if key in inputs_map and val and val.startswith("gs://"):
            bucket, blob = parse_gcs_path(val)
            local_filename = inputs_map[key]
            local_path = os.path.join(temp_dir, local_filename)
            logger.info(f"Downloading input {val} -> {local_path}")
            download_blob(bucket, blob, local_path)
            cmd.extend([key, local_path])
            i += 2
        elif key == "-output" and val and val.startswith("gs://"):
            output_gcs_path = val
            # Point C++ binary to local temp output
            cmd.extend([key, local_output_path])
            i += 2
        else:
            cmd.append(key)
            if val:
                cmd.append(val)
                i += 2
            else:
                i += 1
                
    # Force JSON progress
    cmd.append("--json-progress")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Read stdout line by line
    with process.stdout:
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("{"):
                try:
                    progress_data = json.loads(line)
                    # Log nicely
                    logger.info(f"PROGRESS: [{progress_data.get('progress', 0)*100:.1f}%] {progress_data.get('message', '')}")
                    
                    # Update Firestore if configured
                    # We throttle this by only updating on significant changes or specific stages to save writes
                    # But for now, let's update on every message as they aren't super frequent (10 iterations)
                    if firestore_doc:
                         update_firestore(firestore_doc, {
                             "status": progress_data.get("status"),
                             "stage": progress_data.get("stage"),
                             "progress": progress_data.get("progress"),
                             "last_update": firestore.SERVER_TIMESTAMP
                         })
                         
                except json.JSONDecodeError:
                    logger.info(f"STDOUT: {line}")
            else:
                logger.info(f"STDOUT: {line}")
                
    return_code = process.wait()
    
    if return_code != 0:
        stderr_output = process.stderr.read()
        logger.error(f"Command failed with return code {return_code}")
        logger.error(f"STDERR: {stderr_output}")
        raise RuntimeError("Caustic design process failed")
        
    return local_output_path, output_gcs_path

def main():
    parser = argparse.ArgumentParser(description="Cloud Runner for Caustic Design")
    parser.add_argument("--bin", default="./apps/build/caustic_design", help="Path to caustic_design binary")
    parser.add_argument("--firestore-doc", help="Firestore document path to update (e.g. jobs/123)")
    # All other args are passed through
    
    args, unknown = parser.parse_known_args()
    
    # Setup Firestore if requested
    firestore_doc_ref = None
    if args.firestore_doc and GCS_AVAILABLE:
        try:
            db = firestore.Client()
            firestore_doc_ref = db.document(args.firestore_doc)
            logger.info(f"Reporting progress to Firestore document: {args.firestore_doc}")
        except Exception as e:
            logger.warning(f"Could not connect to Firestore: {e}")

    # Create temp directory for current job
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            if firestore_doc_ref:
                update_firestore(firestore_doc_ref, {"status": "STARTING", "progress": 0.0})

            local_out, gcs_out = run_caustic_design(args.bin, unknown, temp_dir, firestore_doc_ref)

            
            if gcs_out and os.path.exists(local_out):
                bucket, blob = parse_gcs_path(gcs_out)
                logger.info(f"Uploading result to {gcs_out}")
                upload_blob(bucket, blob, local_out)
                
                if firestore_doc_ref:
                     update_firestore(firestore_doc_ref, {
                         "status": "COMPLETED", 
                         "progress": 1.0, 
                         "output_uri": gcs_out,
                         "completed_at": firestore.SERVER_TIMESTAMP
                     })

            elif gcs_out:
                logger.error("Output file expected but not found!")
                if firestore_doc_ref:
                     update_firestore(firestore_doc_ref, {"status": "FAILED", "error": "Output generation failed"})
            else:
                logger.info(f"Run complete. Output left at {local_out} (temp dir will be deleted)")
                if firestore_doc_ref:
                     update_firestore(firestore_doc_ref, {"status": "COMPLETED", "progress": 1.0})
                
        except Exception as e:
            logger.error(f"Job failed: {e}")
            if firestore_doc_ref:
                 update_firestore(firestore_doc_ref, {"status": "FAILED", "error": str(e)})
            sys.exit(1)

if __name__ == "__main__":
    main()
