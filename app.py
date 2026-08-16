"""VACT-Fin public console — self-contained Streamlit app.

Reads the Supabase state mirror (registry, scores, requests) and the R2
bundle store. All credentials live server-side; the browser never sees a key,
and submissions are timestamped server-side so the deadline audit cannot be
gamed. Each task opens as its own sub-page.
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


def _open_task(kind: str, task_id: str) -> None:
    st.session_state["view"] = {"kind": kind, "task_id": task_id}
    st.rerun()


def _task_row(icon: str, task_id: str, subtitle: str, kind: str) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 1])
        left.markdown(f"**{icon} {task_id}**  \n{subtitle}")
        if right.button("Open →", key=f"open_{kind}_{task_id}",
                        use_container_width=True):
            _open_task(kind, task_id)


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
        st.caption("No scores for this task yet — be the first.")


def _feedback_form(scope: str, key: str, hint: str) -> None:
    with st.form(f"fb_{key}"):
        text = st.text_area(hint, key=f"fbtext_{key}")
        if st.form_submit_button("Send feedback") and text.strip():
            stamp = datetime.now(timezone.utc).isoformat()
            _state_put(
                f"feedback/{scope}_{stamp.replace(':', '-')}.json",
                {"kind": scope, "text": text.strip(), "recorded_at": stamp})
            st.success("Got it — this feeds the next generation cycle. 🙏")


# ---------- task detail sub-pages ----------

def detail_live(task_id: str) -> None:
    rows = {r.get("task_id"): r for r in _registry_rows()}
    row = rows.get(task_id)
    if row is None:
        st.error("Task not found.")
        return
    is_open = row.get("status") == "pending_resolution"
    st.title(("🟢 " if is_open else "🏁 ") + task_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets", str(row.get("assets")))
    c2.metric("Deadline", _time_left(str(row.get("resolve_after", ""))))
    c3.metric("Scored by", str(row.get("metric_id")))
    st.caption(f"Deadline (UTC): {row.get('resolve_after')}")
    if is_open:
        st.subheader("📤 Submit your prediction")
        _submit_form(task_id, f"live_{task_id}")
    else:
        st.info("This task has resolved — submissions are closed.")
    st.subheader("🏅 Scores on this task")
    try:
        _task_scores(_score_rows("scores"), task_id)
    except Exception as exc:
        st.error(f"scores unavailable: {exc}")


def detail_bundle(task_id: str, *, challenge: bool) -> None:
    prefix = "public_bundles" if challenge else "bundles"
    st.title(("🎯 " if challenge else "🧪 ") + task_id)
    try:
        manifest = _bundle_manifest(prefix, task_id)
    except Exception as exc:
        st.error(f"could not read bundle: {exc}")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Type", str(manifest.get("family", "?")).replace("_", " "))
    c2.metric("Assets", ", ".join(manifest.get("assets") or [])[:24] or "—")
    c3.metric("Files", len(manifest.get("artifacts") or {}))
    extras = sorted(manifest.get("evidence") or {})
    if extras:
        st.write("Extra evidence: " + ", ".join(f"`{e}`" for e in extras))
    st.download_button(
        "⬇️ Download task", data=_bundle_zip(prefix, task_id),
        file_name=f"{task_id}.zip", mime="application/zip",
        key=f"dl_{prefix}_{task_id}")
    st.caption("manifest.json lists a sha256 for every file — verify after "
               "download." + ("" if challenge else
                              " Answers are included: score yourself locally."))
    if challenge:
        st.subheader("📤 Submit your prediction")
        _submit_form(task_id, f"hist_{task_id}")
        st.subheader("🏅 Scores on this task")
        try:
            _task_scores(_score_rows("historical_scores"), task_id)
        except Exception as exc:
            st.error(f"scores unavailable: {exc}")
        st.subheader("💬 Feedback on this task")
        _feedback_form(f"task_{task_id}", f"task_{task_id}",
                       "Anything wrong or confusing about this task?")


# ---------- list pages ----------

def page_live() -> None:
    st.header("🔴 Live challenges")
    st.write("Predict what the market does next — open a task to submit "
             "before its deadline.")
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
    for row in rows:
        task_id = str(row.get("task_id"))
        is_open = row.get("status") == "pending_resolution"
        _task_row("🟢" if is_open else "🏁", task_id,
                  f"{row.get('assets')} · "
                  f"{_time_left(str(row.get('resolve_after', '')))}",
                  "live")
    if not rows:
        st.info("Nothing live right now — ask for one on the 📝 Request page.")
    st.divider()
    st.subheader("💬 Shape the next live tasks")
    _feedback_form("live_feedback", "live_page",
                   "What should the next live tasks look like? "
                   "(assets, horizons, data you want)")


def page_historical() -> None:
    st.header("🎯 Historical challenges")
    st.write("Open a task to download it, submit predictions, and see its "
             "scores. We hold the answers.")
    try:
        challenge_ids = _bundle_ids("public_bundles")
    except Exception as exc:
        st.error(f"Bundle store unavailable: {exc}")
        challenge_ids = []
    for task_id in challenge_ids:
        _task_row("🎯", task_id, "challenge · answers withheld", "challenge")
    if not challenge_ids:
        st.info("No challenges yet — ask for one on the 📝 Request page.")
    st.divider()
    st.subheader("🧪 Practice bundles")
    st.caption("Answers included — score yourself locally.")
    try:
        practice_ids = _bundle_ids("bundles")
    except Exception:
        practice_ids = []
    for task_id in practice_ids:
        _task_row("🧪", task_id, "practice · answers included", "practice")
    if not practice_ids:
        st.caption("No practice bundles yet.")


def page_request() -> None:
    st.header("📝 Request a task")
    st.write("Describe what you want in your own words — that description "
             "drives the generator. The structured fields are optional "
             "nudges.")
    with st.form("request"):
        request_text = st.text_area(
            "What kind of task do you want? *",
            placeholder=("e.g. A tough weekly volatility forecasting task on "
                         "big tech, with news headlines as extra evidence."),
            help="Required — this is the main input to the generator.")
        difficulty = st.select_slider(
            "How hard should it be?",
            ["easy", "medium", "hard"], value="medium")
        kind = st.radio("Task type", ["historical", "live"], horizontal=True)
        with st.expander("Optional: pin down specifics"):
            family = st.selectbox("Task family", ["let the generator decide",
                                                  *FAMILIES],
                                  format_func=lambda f: f.replace("_", " "))
            assets_text = st.text_input("Assets (comma-separated tickers)", "")
            c1, c2 = st.columns(2)
            horizon_days = c1.number_input(
                "Horizon (trading days, historical)", 1, 30, 5)
            horizon_seconds = c2.number_input(
                "Horizon (seconds, live)", 60, 604_800, 3600)
        submitted = st.form_submit_button("📨 Send request")
    if submitted:
        if not request_text.strip():
            st.error("Please describe the task — the description is required.")
        else:
            stamp = datetime.now(timezone.utc).isoformat()
            payload = {
                "kind": kind,
                "request_text": request_text.strip(),
                "difficulty": difficulty,
                "assets": [a.strip().upper() for a in assets_text.split(",")
                           if a.strip()],
                "horizon_trading_days": int(horizon_days),
                "horizon_seconds": int(horizon_seconds),
                "status": "pending",
                "requested_at": stamp,
            }
            if family != "let the generator decide":
                payload["family"] = family
            _state_put(f"requests/req_{stamp.replace(':', '-')}.json", payload)
            st.success("Request received! Generation usually lands within a "
                       "day — track it below.")

    st.subheader("Your requests")
    try:
        names = _state_list("requests")[-15:]
        recent = [(name, _state_get(f"requests/{name}")) for name in names]
    except Exception:
        recent = []
    status_icon = {"pending": "⏳ pending", "awaiting_review": "👀 review me",
                   "changes_requested": "🔁 regenerating",
                   "fulfilled": "✅ ready", "rejected": "❌ rejected",
                   "failed": "❌ failed"}
    for name, request in reversed(recent):
        label = status_icon.get(str(request.get("status")),
                                str(request.get("status")))
        with st.container(border=True):
            st.markdown(
                f"**{label}** · {request.get('kind')} · "
                f"_{str(request.get('request_text', ''))[:80]}_")
            detail_bits = []
            if request.get("task_id"):
                detail_bits.append(f"task: `{request['task_id']}`")
            if request.get("measured_difficulty") is not None:
                detail_bits.append(
                    f"measured difficulty: {request['measured_difficulty']:.2f} "
                    f"(asked: {request.get('difficulty', '—')})")
            if request.get("reason"):
                detail_bits.append(f"reason: {request['reason']}")
            if detail_bits:
                st.caption(" · ".join(detail_bits))
            if request.get("status") == "awaiting_review":
                st.write("The task is generated — check it on the "
                         "🎯 Historical page, then decide:")
                approve_col, change_col = st.columns(2)
                if approve_col.button("✅ Approve", key=f"appr_{name}"):
                    request["status"] = "fulfilled"
                    _state_put(f"requests/{name}", request)
                    st.rerun()
                change_text = change_col.text_input(
                    "What should change?", key=f"chg_{name}")
                if change_col.button("🔁 Request changes", key=f"chgbtn_{name}"):
                    if not change_text.strip():
                        st.error("Say what should change first.")
                    else:
                        request.setdefault("review_feedback", []).append(
                            change_text.strip())
                        request["status"] = "changes_requested"
                        _state_put(f"requests/{name}", request)
                        st.rerun()
    if not recent:
        st.caption("No requests yet — yours could be the first.")


def page_leaderboard() -> None:
    st.header("🏆 Leaderboard")
    st.caption("Live and historical never mix — different games. Under 5 "
               "rounds is flagged: one lucky round proves nothing.")
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
    st.write("Bugs, ideas, anything about the console itself. Task feedback "
             "lives inside each task's page.")
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
        choice = st.radio("Navigate", list(PAGES), key="nav",
                          label_visibility="collapsed")
        st.divider()
        st.caption("How it works: request or grab a task → predict → submit "
                   "before the deadline → see your score.")

    if st.session_state.get("last_nav") != choice:
        st.session_state["last_nav"] = choice
        st.session_state.pop("view", None)

    view = st.session_state.get("view")
    if view:
        if st.button("⬅ Back"):
            st.session_state.pop("view", None)
            st.rerun()
        kind, task_id = view["kind"], view["task_id"]
        if kind == "live":
            detail_live(task_id)
        elif kind == "challenge":
            detail_bundle(task_id, challenge=True)
        else:
            detail_bundle(task_id, challenge=False)
    else:
        PAGES[choice]()


main()
