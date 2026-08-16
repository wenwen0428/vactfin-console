"""VACT-Fin public console — self-contained Streamlit app for HF Spaces.

Reads the Supabase state mirror (registry, scores) and the R2 bundle store.
Guests browse tasks, download verified bundles, submit predictions before a
live task's resolve time, and leave feedback. All credentials live in Space
secrets server-side; the browser never sees a key, and submissions are
timestamped server-side so the punctuality audit cannot be gamed.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime, timezone

import requests
import streamlit as st

BUCKET = os.environ.get("VACTFIN_BUCKET", "vactfin-artifacts")
STATE_PREFIX = "state"
BUNDLE_PREFIX = "bundles"


def _cfg(name: str) -> str:
    """Secrets come from env vars (HF Spaces) or st.secrets (Streamlit Cloud)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets[name])
    except (KeyError, FileNotFoundError):
        raise RuntimeError(f"missing secret: {name}") from None


def _supabase() -> tuple[str, dict[str, str]]:
    url = _cfg("SUPABASE_URL").rstrip("/")
    key = _cfg("SUPABASE_SECRET_KEY")
    return url, {"Authorization": f"Bearer {key}", "apikey": key}


def _state_list(subdir: str) -> list[str]:
    url, headers = _supabase()
    response = requests.post(
        f"{url}/storage/v1/object/list/{BUCKET}",
        json={"prefix": f"{STATE_PREFIX}/{subdir}".rstrip("/"), "limit": 1000},
        headers=headers, timeout=30)
    response.raise_for_status()
    return sorted(row["name"] for row in response.json() if row.get("name"))


def _state_get(path: str) -> dict:
    url, headers = _supabase()
    response = requests.get(
        f"{url}/storage/v1/object/{BUCKET}/{STATE_PREFIX}/{path}",
        headers=headers, timeout=30)
    response.raise_for_status()
    return json.loads(response.content)


def _state_put(path: str, payload: dict) -> None:
    url, headers = _supabase()
    response = requests.post(
        f"{url}/storage/v1/object/{BUCKET}/{STATE_PREFIX}/{path}",
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={**headers, "x-upsert": "true",
                 "content-type": "application/json"},
        timeout=30)
    response.raise_for_status()


@st.cache_resource
def _r2():
    import boto3

    return boto3.client(
        service_name="s3",
        endpoint_url=_cfg("CLOUDFLARE_R2_ENDPOINT_URL"),
        aws_access_key_id=_cfg("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_cfg("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


@st.cache_data(ttl=300)
def _bundle_ids() -> list[str]:
    paginator = _r2().get_paginator("list_objects_v2")
    ids = set()
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{BUNDLE_PREFIX}/",
                                   Delimiter="/"):
        for prefix in page.get("CommonPrefixes") or []:
            ids.add(prefix["Prefix"].split("/")[1])
    return sorted(ids)


@st.cache_data(ttl=300)
def _bundle_manifest(task_id: str) -> dict:
    obj = _r2().get_object(
        Bucket=BUCKET, Key=f"{BUNDLE_PREFIX}/{task_id}/manifest.json")
    return json.loads(obj["Body"].read())


def _bundle_zip(task_id: str) -> bytes:
    client = _r2()
    buffer = io.BytesIO()
    paginator = client.get_paginator("list_objects_v2")
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for page in paginator.paginate(Bucket=BUCKET,
                                       Prefix=f"{BUNDLE_PREFIX}/{task_id}/"):
            for obj in page.get("Contents") or []:
                relative = obj["Key"].removeprefix(f"{BUNDLE_PREFIX}/{task_id}/")
                body = client.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
                archive.writestr(f"{task_id}/{relative}", body)
    return buffer.getvalue()


def _registry_rows() -> list[dict]:
    return [_state_get(f"registry/{name}") for name in _state_list("registry")]


def main() -> None:
    st.set_page_config(page_title="VACT-Fin", page_icon="V", layout="wide")
    st.title("VACT-Fin — point-in-time-correct financial tasks")
    st.caption(
        "Generated, verified task bundles with measurable leakage inflation. "
        "Live tasks resolve on a clock; submissions after resolution are "
        "refused by the scoring audit.")

    tabs = st.tabs(["Live Tasks", "Historical Bundles", "Submit", "Leaderboard",
                    "Feedback"])

    with tabs[0]:
        st.subheader("Live task registry")
        try:
            rows = _registry_rows()
        except Exception as exc:
            st.error(f"registry unavailable: {exc}")
            rows = []
        if rows:
            now = datetime.now(timezone.utc).isoformat()
            for row in rows:
                row["submission_open"] = (
                    row.get("status") == "pending_resolution"
                    and str(row.get("resolve_after", "")) > now)
            st.dataframe(
                [{k: row.get(k) for k in ("task_id", "status", "assets",
                                          "resolve_after", "submission_open",
                                          "metric_id")}
                 for row in rows],
                hide_index=True, width="stretch")
            st.caption("submission_open = you can still submit; scoring "
                       "refuses anything stamped after resolution.")
        else:
            st.info("No live tasks mirrored yet.")

    with tabs[1]:
        st.subheader("Verified bundles (download and run locally)")
        try:
            ids = _bundle_ids()
        except Exception as exc:
            st.error(f"bundle store unavailable: {exc}")
            ids = []
        if not ids:
            st.info("No bundles published yet.")
        for task_id in ids:
            with st.expander(task_id):
                try:
                    manifest = _bundle_manifest(task_id)
                    st.write({
                        "task_class": manifest.get("task_class"),
                        "family": manifest.get("family"),
                        "assets": manifest.get("assets"),
                        "status": manifest.get("status"),
                        "artifacts": len(manifest.get("artifacts") or {}),
                        "evidence": sorted(manifest.get("evidence") or {}),
                    })
                    st.download_button(
                        f"Download {task_id}.zip",
                        data=_bundle_zip(task_id),
                        file_name=f"{task_id}.zip",
                        mime="application/zip",
                        key=f"dl_{task_id}",
                    )
                    st.caption("Every file is listed with its sha256 in "
                               "manifest.json — verify after download.")
                except Exception as exc:
                    st.error(f"could not read bundle: {exc}")

    with tabs[2]:
        st.subheader("Submit predictions for a pending live task")
        try:
            pending = [row["task_id"] for row in _registry_rows()
                       if row.get("status") == "pending_resolution"]
        except Exception:
            pending = []
        if not pending:
            st.info("No pending live tasks right now.")
        else:
            task_id = st.selectbox("Task", pending)
            system_id = st.text_input("Your system id", "guest_system")
            payload_text = st.text_area(
                "Predictions JSON (row_id -> number)",
                '{"AAPL_...": 0.0}',
                help="Row ids are listed in the bundle's live_observations.json.")
            if st.button("Submit"):
                try:
                    submission = json.loads(payload_text)
                    assert isinstance(submission, dict) and submission
                except Exception:
                    st.error("Predictions must be a non-empty JSON object.")
                else:
                    stamped = {
                        "task_id": task_id,
                        "system_id": system_id.strip() or "guest_system",
                        "submission": submission,
                        "submitted_at": datetime.now(timezone.utc).isoformat(),
                    }
                    _state_put(
                        f"submissions/{task_id}__{stamped['system_id']}.json",
                        stamped)
                    st.success(
                        f"Submitted at {stamped['submitted_at']}. Scoring "
                        "happens at resolution; check the leaderboard after "
                        "the task's resolve time.")

    with tabs[3]:
        st.subheader("Leaderboard (live-resolved scores)")
        try:
            score_rows = [_state_get(f"scores/{name}")
                          for name in _state_list("scores")]
        except Exception as exc:
            st.error(f"scores unavailable: {exc}")
            score_rows = []
        if score_rows:
            by_system: dict[tuple[str, str], list[float]] = {}
            for row in score_rows:
                key = (str(row.get("system_id")), str(row.get("metric_id")))
                by_system.setdefault(key, []).append(float(row.get("score", 0.0)))
            aggregates = []
            for (system_id, metric_id), scores in sorted(by_system.items()):
                n = len(scores)
                mean = sum(scores) / n
                std = ((sum((s - mean) ** 2 for s in scores) / (n - 1)) ** 0.5
                       if n > 1 else None)
                aggregates.append({
                    "system_id": system_id, "metric_id": metric_id,
                    "rounds": n, "mean_score": round(mean, 6),
                    "score_std": round(std, 6) if std is not None else None,
                    "sample": "ok" if n >= 5 else "insufficient (<5 rounds)",
                })
            st.dataframe(aggregates, hide_index=True, width="stretch")
            st.caption("One round cannot separate skill from luck; systems "
                       "under 5 scored rounds are flagged, not ranked.")
            with st.expander("Raw per-round scores"):
                st.dataframe(
                    [{k: row.get(k) for k in ("task_id", "system_id", "metric_id",
                                              "score", "metric_orientation")}
                     for row in score_rows],
                    hide_index=True, width="stretch")
        else:
            st.info("No scores yet.")

    with tabs[4]:
        st.subheader("Feedback")
        kind = st.selectbox("About", ["task_quality", "bug", "request"])
        text = st.text_area("What should we know?")
        contact = st.text_input("Contact (optional)")
        if st.button("Send feedback"):
            if not text.strip():
                st.error("Feedback text is empty.")
            else:
                stamp = datetime.now(timezone.utc).isoformat()
                _state_put(
                    f"feedback/{kind}_{stamp.replace(':', '-')}.json",
                    {"kind": kind, "text": text.strip(),
                     "contact": contact.strip(), "recorded_at": stamp})
                st.success("Recorded — thank you.")


main()
