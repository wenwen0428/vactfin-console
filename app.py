"""VACT-Fin public console — self-contained Streamlit app.

Reads the Supabase state mirror (registry, scores, requests) and the R2
bundle store. All credentials live server-side; the browser never sees a key,
and submissions are timestamped server-side so the deadline audit cannot be
gamed.
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
FAMILIES = ["return_forecasting", "volatility_forecasting",
            "cross_sectional_ranking", "portfolio_trading"]


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
    seconds = int((deadline - datetime.now(timezone.utc)).total_seconds())
    if seconds <= 0:
        return "closed"
    if seconds < 3600:
        return f"{seconds // 60}m left"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m left"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600}h left"


def _submit_form(task_id: str, key: str) -> None:
    system_id = st.text_input("Your team / model name", "my_model",
                              key=f"system_{key}")
    payload_text = st.text_area(
        "Your predictions (JSON: row_id → number)", "{}",
        key=f"payload_{key}",
        help="Row ids are inside the task you downloaded.")
    if st.button("🚀 Submit", key=f"btn_{key}"):
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
        st.success(f"Locked in at {stamped['submitted_at'][:19]} UTC.")
        st.balloons()


def _task_scores(rows: list[dict], task_id: str) -> None:
    mine = [row for row in rows if str(row.get("task_id")) == task_id]
    if mine:
        st.dataframe(
            [{"system": r.get("system_id"), "metric": r.get("metric_id"),
              "score": r.get("score")} for r in mine],
            hide_index=True, width="stretch")
    else:
        st.caption("No scores for this task yet.")


def _feedback_form(scope: str, key: str, hint: str) -> None:
    with st.form(f"fb_{key}"):
        text = st.text_area(hint, key=f"fbtext_{key}")
        if st.form_submit_button("Send feedback") and text.strip():
            stamp = datetime.now(timezone.utc).isoformat()
            _state_put(
                f"feedback/{scope}_{stamp.replace(':', '-')}.json",
                {"kind": scope, "text": text.strip(), "recorded_at": stamp})
            st.success("Got it — thank you! 🙏")


def page_live() -> None:
    st.header("🔴 Live challenges")
    st.write("Predict what the market does next. Each task is its own card — "
             "open one to submit before its deadline.")
    try:
        rows = _registry_rows()
    except Exception as exc:
        st.error(f"Registry unavailable: {exc}")
        return
    pending = [r for r in rows if r.get("status") == "pending_resolution"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Open now", len(pending))
    c2.metric("Resolved", len(rows) - len(pending))
    next_deadline = min((r.get("resolve_after", "") for r in pending), default="")
    c3.metric("Next deadline", _time_left(next_deadline) if next_deadline else "—")

    try:
        score_rows = _score_rows("scores")
    except Exception:
        score_rows = []
    for row in rows:
        task_id = str(row.get("task_id"))
        is_open = row.get("status") == "pending_resolution"
        badge = "🟢" if is_open else "🏁"
        with st.expander(f"{badge} {task_id} · {row.get('assets')} · "
                         f"{_time_left(str(row.get('resolve_after', '')))}"):
            st.write({"assets": row.get("assets"),
                      "scored by": row.get("metric_id"),
                      "deadline (UTC)": row.get("resolve_after")})
            if is_open:
                st.markdown("**Submit your prediction**")
                _submit_form(task_id, f"live_{task_id}")
            st.markdown("**Scores on this task**")
            _task_scores(score_rows, task_id)
    if not rows:
        st.info("Nothing live right now — request one on the "
                "📝 Request page.")
    st.divider()
    st.subheader("💬 Feedback on live challenges")
    _feedback_form("live_feedback", "live_page",
                   "What should the next live tasks look like? "
                   "(assets, horizons, data you want to see)")


def page_historical() -> None:
    st.header("🎯 Historical challenges")
    st.write("Each card is one task: download it, work locally, submit your "
             "predictions here. We hold the answers and score server-side.")
    try:
        challenge_ids = _bundle_ids("public_bundles")
    except Exception as exc:
        st.error(f"Bundle store unavailable: {exc}")
        challenge_ids = []
    try:
        score_rows = _score_rows("historical_scores")
    except Exception:
        score_rows = []
    if not challenge_ids:
        st.info("No challenges yet — request one on the 📝 Request page.")
    for task_id in challenge_ids:
        with st.expander(f"📦 {task_id}"):
            _bundle_details("public_bundles", task_id)
            st.markdown("**Submit your prediction**")
            _submit_form(task_id, f"hist_{task_id}")
            st.markdown("**Scores on this task**")
            _task_scores(score_rows, task_id)
            st.markdown("**Feedback on this task**")
            _feedback_form(f"task_{task_id}", f"task_{task_id}",
                           "Anything wrong or confusing about this task?")

    st.divider()
    st.subheader("🧪 Practice bundles")
    st.caption("Answers included — score yourself locally, nothing to submit.")
    try:
        practice_ids = _bundle_ids("bundles")
    except Exception:
        practice_ids = []
    for task_id in practice_ids:
        with st.expander(f"🧪 {task_id}"):
            _bundle_details("bundles", task_id)
    if not practice_ids:
        st.caption("No practice bundles yet.")


def _bundle_details(prefix: str, task_id: str) -> None:
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
            "⬇️ Download", data=_bundle_zip(prefix, task_id),
            file_name=f"{task_id}.zip", mime="application/zip",
            key=f"dl_{prefix}_{task_id}")
        st.caption("manifest.json lists a sha256 for every file — verify "
                   "after download.")
    except Exception as exc:
        st.error(f"could not read bundle: {exc}")


def page_request() -> None:
    st.header("📝 Request a task")
    st.write("Tell us what you want to practice on; we generate it, verify "
             "it, and it appears in the challenge pages — usually within a "
             "day.")
    with st.form("request"):
        kind = st.radio("Task type", ["historical", "live"], horizontal=True)
        family = st.selectbox("Task family", FAMILIES,
                              format_func=lambda f: f.replace("_", " "))
        assets_text = st.text_input("Assets (comma-separated tickers)",
                                    "AAPL, MSFT")
        c1, c2 = st.columns(2)
        horizon_days = c1.number_input("Horizon (trading days, historical)",
                                       1, 30, 5)
        horizon_seconds = c2.number_input("Horizon (seconds, live)",
                                          60, 604_800, 3600)
        notes = st.text_area("Anything else? (optional)")
        submitted = st.form_submit_button("📨 Send request")
    if submitted:
        assets = [a.strip().upper() for a in assets_text.split(",") if a.strip()]
        if not assets:
            st.error("Please list at least one asset.")
        else:
            stamp = datetime.now(timezone.utc).isoformat()
            _state_put(f"requests/req_{stamp.replace(':', '-')}.json", {
                "kind": kind, "family": family, "assets": assets,
                "horizon_trading_days": int(horizon_days),
                "horizon_seconds": int(horizon_seconds),
                "notes": notes.strip(), "status": "pending",
                "requested_at": stamp,
            })
            st.success("Request received! Check back here for its status.")

    st.subheader("Recent requests")
    try:
        recent = [_state_get(f"requests/{name}")
                  for name in _state_list("requests")[-10:]]
    except Exception:
        recent = []
    if recent:
        st.dataframe(
            [{"when": str(r.get("requested_at", ""))[:16],
              "type": r.get("kind"), "family": r.get("family"),
              "assets": ", ".join(r.get("assets") or []),
              "status": {"pending": "⏳ pending",
                         "fulfilled": "✅ ready",
                         "rejected": "❌ rejected"}.get(
                             str(r.get("status")), str(r.get("status"))),
              "task": r.get("task_id", r.get("reason", ""))}
             for r in reversed(recent)],
            hide_index=True, width="stretch")
    else:
        st.caption("No requests yet — yours could be the first.")


def page_leaderboard() -> None:
    st.header("🏆 Leaderboard")
    st.caption("Live and historical results never mix — different games. "
               "Under 5 rounds is flagged: one lucky round proves nothing. "
               "Per-task scores live inside each task's card.")
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
                                          "score")} for row in rows],
                hide_index=True, width="stretch")


def page_feedback() -> None:
    st.header("💬 About this site")
    st.write("Bugs, ideas, anything about the console itself. Task-specific "
             "feedback lives inside each task's card.")
    _feedback_form("site", "site", "What should we fix or add?")


PAGES = {
    "🔴 Live challenges": page_live,
    "🎯 Historical challenges": page_historical,
    "📝 Request a task": page_request,
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
        st.caption("How it works: request or grab a task → predict → submit "
                   "before the deadline → see your score.")
    PAGES[choice]()


main()
