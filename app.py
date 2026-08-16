"""VACT-Fin public console — self-contained Streamlit app.

Reads the Supabase state mirror (registry, scores) and the R2 bundle store.
All credentials live server-side (Streamlit secrets / env); the browser never
sees a key, and submissions are timestamped server-side so the deadline audit
cannot be gamed.
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


def _cfg(name: str) -> str:
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
def _bundle_ids(prefix: str) -> list[str]:
    paginator = _r2().get_paginator("list_objects_v2")
    ids = set()
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{prefix}/",
                                   Delimiter="/"):
        for common in page.get("CommonPrefixes") or []:
            ids.add(common["Prefix"].split("/")[1])
    return sorted(ids)


@st.cache_data(ttl=300)
def _bundle_manifest(prefix: str, task_id: str) -> dict:
    obj = _r2().get_object(Bucket=BUCKET, Key=f"{prefix}/{task_id}/manifest.json")
    return json.loads(obj["Body"].read())


def _bundle_zip(prefix: str, task_id: str) -> bytes:
    client = _r2()
    buffer = io.BytesIO()
    paginator = client.get_paginator("list_objects_v2")
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for page in paginator.paginate(Bucket=BUCKET,
                                       Prefix=f"{prefix}/{task_id}/"):
            for obj in page.get("Contents") or []:
                relative = obj["Key"].removeprefix(f"{prefix}/{task_id}/")
                body = client.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
                archive.writestr(f"{task_id}/{relative}", body)
    return buffer.getvalue()


def _registry_rows() -> list[dict]:
    return [_state_get(f"registry/{name}") for name in _state_list("registry")]


def _score_rows(subdir: str) -> list[dict]:
    return [_state_get(f"{subdir}/{name}") for name in _state_list(subdir)]


def _aggregate(rows: list[dict]) -> list[dict]:
    by_system: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row.get("system_id")), str(row.get("metric_id")))
        by_system.setdefault(key, []).append(float(row.get("score", 0.0)))
    out = []
    for (system_id, metric_id), scores in sorted(by_system.items()):
        n = len(scores)
        mean = sum(scores) / n
        std = ((sum((s - mean) ** 2 for s in scores) / (n - 1)) ** 0.5
               if n > 1 else None)
        out.append({
            "system": system_id, "metric": metric_id, "rounds": n,
            "avg score": round(mean, 6),
            "spread": round(std, 6) if std is not None else None,
            "sample": "✅ ok" if n >= 5 else "⚠️ <5 rounds",
        })
    return out


def _time_left(resolve_after: str) -> str:
    try:
        deadline = datetime.fromisoformat(resolve_after)
    except ValueError:
        return "—"
    delta = deadline - datetime.now(timezone.utc)
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "closed"
    if seconds < 3600:
        return f"{seconds // 60}m left"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m left"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600}h left"


def _submit_form(pending: list[str], kind: str) -> None:
    task_id = st.selectbox("Pick a task", pending, key=f"task_{kind}")
    system_id = st.text_input("Your team / model name", "my_model",
                              key=f"system_{kind}")
    payload_text = st.text_area(
        "Your predictions (JSON: row_id → number)",
        "{}", key=f"payload_{kind}",
        help="Row ids are inside the task you downloaded.")
    if st.button("🚀 Submit", key=f"btn_{kind}"):
        try:
            submission = json.loads(payload_text)
            assert isinstance(submission, dict) and submission
        except Exception:
            st.error("Predictions must be a non-empty JSON object.")
            return
        stamped = {
            "task_id": task_id,
            "system_id": system_id.strip() or "anonymous",
            "submission": submission,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        _state_put(f"submissions/{task_id}__{stamped['system_id']}.json", stamped)
        st.success(f"Locked in at {stamped['submitted_at'][:19]} UTC. "
                   "Scores appear on the leaderboard after resolution.")
        st.balloons()


def page_live() -> None:
    st.header("🔴 Live challenges")
    st.write("Predict what the market does next. Each task has a real "
             "deadline — submit before it resolves.")
    try:
        rows = _registry_rows()
    except Exception as exc:
        st.error(f"Registry unavailable: {exc}")
        return
    pending = [r for r in rows if r.get("status") == "pending_resolution"]
    resolved = [r for r in rows if r.get("status") == "resolved"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Open now", len(pending))
    c2.metric("Resolved", len(resolved))
    next_deadline = min((r.get("resolve_after", "") for r in pending),
                        default="")
    c3.metric("Next deadline", _time_left(next_deadline) if next_deadline else "—")
    if not rows:
        st.info("Nothing live right now — check back soon.")
        return
    st.dataframe(
        [{"task": r.get("task_id"), "assets": r.get("assets"),
          "status": "🟢 open" if r in pending else "🏁 resolved",
          "deadline": _time_left(str(r.get("resolve_after", ""))),
          "scored by": r.get("metric_id")}
         for r in rows],
        hide_index=True, width="stretch")


def page_library() -> None:
    st.header("📚 Task library")
    st.write("Download a task, work on it locally, and (for challenge "
             "bundles) send us your predictions.")

    st.subheader("🎯 Challenge bundles")
    st.caption("Answers withheld — submit your predictions and we score them.")
    try:
        challenge_ids = _bundle_ids("public_bundles")
    except Exception as exc:
        st.error(f"Bundle store unavailable: {exc}")
        challenge_ids = []
    if not challenge_ids:
        st.info("No challenge bundles yet.")
    for task_id in challenge_ids:
        _bundle_card("public_bundles", task_id)

    st.subheader("🧪 Practice bundles")
    st.caption("Answers included — score yourself locally, no submission.")
    try:
        practice_ids = _bundle_ids("bundles")
    except Exception:
        practice_ids = []
    if not practice_ids:
        st.info("No practice bundles yet.")
    for task_id in practice_ids:
        _bundle_card("bundles", task_id)


def _bundle_card(prefix: str, task_id: str) -> None:
    with st.expander(f"📦 {task_id}"):
        try:
            manifest = _bundle_manifest(prefix, task_id)
            c1, c2, c3 = st.columns(3)
            c1.metric("Type", str(manifest.get("family", "?")).replace("_", " "))
            c2.metric("Assets", ", ".join(manifest.get("assets") or [])[:24] or "—")
            c3.metric("Files", len(manifest.get("artifacts") or {}))
            extras = sorted(manifest.get("evidence") or {})
            if extras:
                st.write("Extra evidence: " + ", ".join(f"`{e}`" for e in extras))
            st.download_button(
                "⬇️ Download",
                data=_bundle_zip(prefix, task_id),
                file_name=f"{task_id}.zip",
                mime="application/zip",
                key=f"dl_{prefix}_{task_id}",
            )
            st.caption("manifest.json lists a sha256 for every file — "
                       "verify after download.")
        except Exception as exc:
            st.error(f"could not read bundle: {exc}")


def page_submit() -> None:
    st.header("📤 Submit predictions")
    kind = st.radio("What are you submitting for?",
                    ["🔴 Live challenge", "🎯 Historical challenge"],
                    horizontal=True)
    if kind.startswith("🔴"):
        try:
            pending = [r["task_id"] for r in _registry_rows()
                       if r.get("status") == "pending_resolution"]
        except Exception:
            pending = []
        if not pending:
            st.info("No open live tasks right now.")
            return
        st.caption("Live tasks are scored the moment they resolve. "
                   "Late submissions are refused automatically.")
        _submit_form(pending, "live")
    else:
        try:
            challenge_ids = _bundle_ids("public_bundles")
        except Exception:
            challenge_ids = []
        if not challenge_ids:
            st.info("No historical challenges published yet.")
            return
        st.caption("We keep the answers; your predictions are scored "
                   "server-side, usually within a day.")
        _submit_form(challenge_ids, "historical")


def page_leaderboard() -> None:
    st.header("🏆 Leaderboard")
    st.caption("Live and historical results never mix — they are different "
               "games. Under 5 rounds is flagged: one lucky round proves "
               "nothing.")
    for title, subdir in [("🔴 Live", "scores"),
                          ("🎯 Historical", "historical_scores")]:
        st.subheader(title)
        try:
            rows = _score_rows(subdir)
        except Exception as exc:
            st.error(f"scores unavailable: {exc}")
            continue
        if not rows:
            st.info("No scores here yet — be the first.")
            continue
        st.dataframe(_aggregate(rows), hide_index=True, width="stretch")
        with st.expander("Every scored round"):
            st.dataframe(
                [{k: row.get(k) for k in ("task_id", "system_id", "metric_id",
                                          "score")}
                 for row in rows],
                hide_index=True, width="stretch")


def page_feedback() -> None:
    st.header("💬 Tell us what to fix")
    st.write("Bad task? Confusing data? Want a new market or task type? "
             "This goes straight into the next generation cycle.")
    kind = st.selectbox("Topic", ["a task felt wrong", "bug", "feature request",
                                  "something else"])
    text = st.text_area("Details")
    contact = st.text_input("Contact (optional)")
    if st.button("Send"):
        if not text.strip():
            st.error("Please write something first.")
        else:
            stamp = datetime.now(timezone.utc).isoformat()
            _state_put(
                f"feedback/{kind.replace(' ', '_')}_{stamp.replace(':', '-')}.json",
                {"kind": kind, "text": text.strip(),
                 "contact": contact.strip(), "recorded_at": stamp})
            st.success("Got it — thank you! 🙏")


PAGES = {
    "🔴 Live challenges": page_live,
    "📚 Task library": page_library,
    "📤 Submit": page_submit,
    "🏆 Leaderboard": page_leaderboard,
    "💬 Feedback": page_feedback,
}


def main() -> None:
    st.set_page_config(page_title="VACT-Fin", page_icon="📈", layout="wide")
    with st.sidebar:
        st.title("📈 VACT-Fin")
        st.caption("Market prediction challenges, scored fairly and on time.")
        choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption("How it works: grab a task → predict → submit before the "
                   "deadline → see your score.")
    PAGES[choice]()


main()
