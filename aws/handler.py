"""API Gateway entrypoint: validate a book, store it, and submit an AWS Batch GPU job."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "common"))

import boto3
import jobtext

s3 = boto3.client("s3")
batch = boto3.client("batch")


def _response(code: int, payload: dict) -> dict:
    return {
        "statusCode": code,
        "headers": {"content-type": "application/json", "access-control-allow-origin": "*"},
        "body": json.dumps(payload),
    }


def _body(event: dict) -> dict:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode("utf-8")
    return json.loads(raw)


def submit(event: dict) -> dict:
    try:
        job = jobtext.validate(_body(event))
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        return _response(400, {"error": str(exc)})

    bucket = os.environ["BUCKET"]
    job_id = str(uuid.uuid4())
    text = jobtext.to_job_text(job)
    s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}/job.txt", Body=text.encode("utf-8"))
    submitted = batch.submit_job(
        jobName=f"mcrisk-{job_id[:8]}",
        jobQueue=os.environ["JOB_QUEUE"],
        jobDefinition=os.environ["JOB_DEFINITION"],
        containerOverrides={
            "environment": [
                {"name": "JOB_S3", "value": f"s3://{bucket}/jobs/{job_id}/job.txt"},
                {"name": "OUT_S3", "value": f"s3://{bucket}/jobs/{job_id}/result.json"},
                {"name": "MC_KERNEL", "value": "/opt/kernels/mc_risk.cu"},
                {"name": "NVRTC_PATH", "value": "/usr/local/cuda/lib64/libnvrtc.so.12"},
            ]
        },
    )
    s3.put_object(
        Bucket=bucket,
        Key=f"jobs/{job_id}/meta.json",
        Body=json.dumps({"batch_job_id": submitted["jobId"]}).encode("utf-8"),
    )
    return _response(202, {"job_id": job_id, "batch_job_id": submitted["jobId"], "status": "SUBMITTED"})


def status(event: dict) -> dict:
    job_id = (event.get("pathParameters") or {}).get("job_id", "")
    if not job_id or any(c not in "0123456789abcdef-" for c in job_id):
        return _response(400, {"error": "job_id is missing"})
    bucket = os.environ["BUCKET"]
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}/result.json")
        result = json.loads(obj["Body"].read().decode("utf-8"))
        result["job_id"] = job_id
        result["status"] = "SUCCEEDED"
        return _response(200, result)
    except s3.exceptions.NoSuchKey:
        pass
    except Exception as exc:  # result not written yet, or a real S3 error
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code not in ("NoSuchKey", "404", "NotFound"):
            return _response(502, {"error": str(exc)})

    try:
        meta = json.loads(s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}/meta.json")["Body"].read().decode("utf-8"))
    except Exception:
        return _response(404, {"error": "unknown job"})
    described = batch.describe_jobs(jobs=[meta["batch_job_id"]])
    jobs = described.get("jobs") or []
    state = jobs[0]["status"] if jobs else "UNKNOWN"
    return _response(200, {"job_id": job_id, "batch_job_id": meta["batch_job_id"], "status": state})


def lambda_handler(event, context):
    route = (event.get("requestContext") or {}).get("http", {}).get("method", "GET").upper()
    path = event.get("rawPath") or ""
    if route == "POST" and path.rstrip("/").endswith("/jobs"):
        return submit(event)
    if route == "GET" and "/jobs/" in path:
        return status(event)
    return _response(404, {"error": "not found"})
