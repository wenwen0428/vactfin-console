"""VACT-Fin public console — self-contained Streamlit app.

Reads the Supabase state mirror (registry, scores, requests) and the R2
bundle store. All credentials live server-side; the browser never sees a key,
and submissions are timestamped server-side so the deadline audit cannot be
gamed. Each task opens as its own sub-page; requests live at the top of the
page they belong to.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from datetime import date, datetime, timezone

import requests
import streamlit as st

BUCKET = os.environ.get("VACTFIN_BUCKET", "vactfin-artifacts")
STATE_PREFIX = "state"
FAMILIES = ["return_forecasting", "volatility_forecasting",
            "cross_sectional_ranking", "portfolio_trading"]
HIST_MODALITIES = ["chart", "news", "news_article", "filing_text",
                   "filing_table", "macro"]
OPERATORS = {
    "future_direction_hint": "realistic leak — a subtle hint of the future",
    "label_as_feature": "control leak — the answer hidden in a column",
    "purge_violation": "control leak — train/test windows overlap",
}
LIVE_SHAPES = {
    "single": ("One-shot forecast",
               "Predict once; scored at one deadline."),
    "chain": ("Multi-round forecast",
              "A fresh round every interval — submit before each round's "
              "deadline; rounds average into your score."),
    "portfolio": ("Portfolio paper-trading",
                  "Allocate a paper portfolio over recent real ticks."),
    "polymarket": ("Prediction-market paper-trading",
                   "Paper-trade real Polymarket YES/NO markets."),
}

CSS = """
<style>
.vf-pills {display:flex;flex-wrap:wrap;gap:.4rem;margin:.15rem 0 .9rem 0;}
.vf-pill {display:inline-flex;align-items:center;gap:.32rem;
  padding:.18rem .7rem;border-radius:999px;font-size:.78rem;font-weight:600;
  letter-spacing:.01em;white-space:nowrap;}
.vf-indigo {background:rgba(99,102,241,.13);color:#6366F1;
  border:1px solid rgba(99,102,241,.35);}
.vf-teal {background:rgba(20,184,166,.13);color:#0D9488;
  border:1px solid rgba(20,184,166,.35);}
.vf-amber {background:rgba(245,158,11,.15);color:#B45309;
  border:1px solid rgba(245,158,11,.4);}
.vf-slate {background:rgba(100,116,139,.12);color:#64748B;
  border:1px solid rgba(100,116,139,.3);}
.vf-rose {background:rgba(244,63,94,.12);color:#E11D48;
  border:1px solid rgba(244,63,94,.35);}
.vf-title {font-size:1.02rem;font-weight:700;margin-bottom:.1rem;}
.vf-sub {font-size:.8rem;opacity:.65;}
</style>
"""


def _pills(items: list[tuple[str, str]]) -> str:
    """[(text, tone)] -> one pills row. Tones: indigo/teal/amber/slate/rose."""
    spans = "".join(
        f'<span class="vf-pill vf-{tone}">{text}</span>' for text, tone in items)
    return f'<div class="vf-pills">{spans}</div>'


def _pretty(task_id: str, manifest: dict | None = None) -> str:
    if manifest and manifest.get("title"):
        return str(manifest["title"])
    bare = re.sub(r"(_\d{4}){1,2}$", "", task_id)
    bare = re.sub(r"_\d{8}_\d{6}$", "", bare)
    tickers = {str(a).lower() for a in (manifest or {}).get("assets") or []}
    words = [word.upper() if word.lower() in tickers else word.title()
             for word in bare.replace("__", "_·_").split("_")]
    return " ".join(words)


def _seal_api_key(plain: str) -> str:
    """Seal the guest's key with the operator's X25519 public key.

    Only ciphertext ever reaches storage; the matching private key exists
    solely on the generation machine. Returns "" when no public key is
    configured, which the caller treats as "key submission disabled".
    """
    try:
        public_b64 = _cfg("VACTFIN_REQUEST_PUBLIC_KEY")
    except RuntimeError:
        return ""
    import base64

    from nacl.public import PublicKey, SealedBox

    sealed = SealedBox(PublicKey(base64.b64decode(public_b64))).encrypt(
        plain.encode("utf-8"))
    return base64.b64encode(sealed).decode("utf-8")


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
def _dataset_row_ids(prefix: str, task_id: str) -> list[str]:
    try:
        obj = _r2().get_object(
            Bucket=BUCKET,
            Key=f"{prefix}/{task_id}/environment/data/dataset.json")
        dataset = json.loads(obj["Body"].read())
        return [str(row.get("row_id")) for row in dataset.get("test") or []]
    except Exception:
        return []


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


@st.cache_data(ttl=60)
def _task_lifecycle() -> dict[str, str]:
    """task_id -> lifecycle, derived from requests. Absent = operator-published."""
    try:
        names = _state_list("requests")
    except Exception:
        return {}
    lifecycle = {}
    for name in names:
        try:
            request = _state_get(f"requests/{name}")
        except Exception:
            continue
        task_id = str(request.get("task_id") or "")
        if not task_id:
            continue
        lifecycle[task_id] = {
            "pending": "generating",
            "changes_requested": "improving",
            "awaiting_review": "pending review",
            "fulfilled": "approved",
            "rejected": "failed",
            "failed": "failed",
        }.get(str(request.get("status")), "approved")
    return lifecycle


LIFECYCLE_TONE = {"generating": "slate", "improving": "amber",
                  "pending review": "amber", "approved": "teal",
                  "failed": "rose"}


def _lifecycle_pill(task_id: str) -> tuple[str, str]:
    state = _task_lifecycle().get(task_id, "approved")
    return (state, LIFECYCLE_TONE.get(state, "slate"))


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


def _task_row(task_id: str, title: str, pills: list[tuple[str, str]],
              kind: str) -> None:
    with st.container(border=True):
        left, right = st.columns([6, 1])
        with left:
            st.markdown(f'<div class="vf-title">{title}</div>'
                        f'<div class="vf-sub">{task_id}</div>'
                        + _pills(pills), unsafe_allow_html=True)
        if right.button("Open →", key=f"open_{kind}_{task_id}",
                        use_container_width=True):
            _open_task(kind, task_id)


def _parse_submission(uploaded, payload_text: str) -> tuple[dict | None, str]:
    """Uploaded .json/.csv wins over the pasted text. Returns (data, error)."""
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8", errors="replace")
        if uploaded.name.lower().endswith(".csv"):
            rows = [line.split(",") for line in raw.strip().splitlines()
                    if line.strip()]
            if rows and rows[0][0].strip().lower() in ("row_id", "id"):
                rows = rows[1:]
            try:
                return ({cell[0].strip(): float(cell[1])
                         for cell in rows if len(cell) >= 2}, "")
            except ValueError:
                return None, "CSV rows must be `row_id,number`."
        try:
            data = json.loads(raw)
            assert isinstance(data, dict) and data
            return data, ""
        except Exception:
            return None, "The uploaded JSON must be a non-empty object."
    try:
        data = json.loads(payload_text)
        assert isinstance(data, dict) and data
        return data, ""
    except Exception:
        return None, "Predictions must be a non-empty JSON object."


def _submit_form(task_id: str, key: str,
                 expected_row_ids: list[str] | None = None) -> None:
    system_id = st.text_input("Your team / model name", "my_model",
                              key=f"system_{key}")
    uploaded = st.file_uploader(
        "Upload predictions (.json or .csv with row_id,prediction)",
        type=["json", "csv"], key=f"file_{key}")
    payload_text = st.text_area(
        "…or paste JSON (row_id → number)", "{}",
        key=f"payload_{key}",
        help="Row ids are inside the task you downloaded.")
    if st.button("🚀 Submit", key=f"btn_{key}", type="primary"):
        submission, error = _parse_submission(uploaded, payload_text)
        if submission is None:
            st.error(error)
            return
        if expected_row_ids:
            expected = set(expected_row_ids)
            got = set(submission)
            missing, extra = sorted(expected - got), sorted(got - expected)
            if missing or extra:
                st.error(
                    "Row ids do not match this task. "
                    + (f"Missing {len(missing)} (e.g. {missing[:3]}). "
                       if missing else "")
                    + (f"Unknown {len(extra)} (e.g. {extra[:3]})."
                       if extra else ""))
                return
        stamped = {
            "task_id": task_id,
            "system_id": system_id.strip() or "anonymous",
            "submission": submission,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        _state_put(f"submissions/{task_id}__{stamped['system_id']}.json", stamped)
        checked = " (row ids verified)" if expected_row_ids else ""
        st.success(f"Locked in at {stamped['submitted_at'][:19]} UTC{checked}.")
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


# ---------- requests (embedded per page) ----------

def _chart_render_estimate(assets_text: str, pinned: bool,
                           win_start, win_end) -> int:
    """Browser-side preview of the builder's assets x months render cap.

    The builder enforces the real cap fail-closed at generation time; this
    estimate exists so an oversized request is refused before it is sent,
    not hours later in a failure status. Unpinned windows use the default
    generation period (~30 months) as the estimate.
    """
    n_assets = max(1, len([a for a in assets_text.split(",") if a.strip()]))
    months = (max(1, (win_end - win_start).days // 30) if pinned else 30)
    return n_assets * months


def _request_section(kind: str) -> None:
    label = ("✨ Request a new live challenge" if kind == "live"
             else "✨ Request a new historical challenge")
    with st.expander(label):
        st.write("Describe what you want in your own words — that "
                 "description drives the generator.")
        if kind == "live":
            live_shape = st.radio(
                "Challenge shape", list(LIVE_SHAPES),
                key=f"reqshape_{kind}", horizontal=True,
                format_func=lambda v: LIVE_SHAPES[v][0],
                captions=None)
            st.caption(LIVE_SHAPES[live_shape][1])
        else:
            mode = st.radio(
                "Mode", ["competition", "practice"], key=f"reqmode_{kind}",
                horizontal=True,
                format_func=lambda m: ("🏆 Competition — answers withheld; "
                                       "submit and rank"
                                       if m == "competition" else
                                       "🧪 Practice — answers included; "
                                       "score yourself, no leaderboard"))
        with st.form(f"request_{kind}"):
            request_text = st.text_area(
                "What kind of task do you want? *",
                placeholder=("e.g. An hourly forecast on chip stocks with "
                             "recent news as context."
                             if kind == "live" else
                             "e.g. A tough weekly volatility task on big "
                             "tech, 2024 data."),
                key=f"reqtext_{kind}")
            difficulty = st.select_slider(
                "How hard should it be?",
                ["auto", "easy", "medium", "hard"], value="auto",
                key=f"reqdiff_{kind}",
                help="auto = we read it from your words.")
            api_key = st.text_input(
                "Your DeepSeek API key *", type="password",
                key=f"reqkey_{kind}",
                help="Pays for understanding your request (and writing a "
                     "custom metric if you ask for one) — typically well "
                     "under $0.01. Encrypted in your browser's request with "
                     "our public key, decrypted only on the generation "
                     "machine, and deleted the moment your request is "
                     "handled. Prefer a spending-capped key.")
            with st.expander("Optional: pin down specifics"):
                assets_text = st.text_input(
                    "Assets (comma-separated tickers)", "",
                    key=f"reqassets_{kind}")
                if kind == "historical":
                    family = st.selectbox(
                        "Task family",
                        ["let the generator decide", *FAMILIES],
                        format_func=lambda f: f.replace("_", " "),
                        key=f"reqfam_{kind}")
                    horizon_days = st.number_input(
                        "Horizon (trading days)", 1, 30, 5,
                        key=f"reqhd_{kind}")
                    modalities = st.multiselect(
                        "Extra evidence", HIST_MODALITIES,
                        key=f"reqmod_{kind}",
                        format_func=lambda m: m.replace("_", " "),
                        help="Charts, headlines, SEC filings/tables, macro — "
                             "on top of daily prices.")
                    pair_style = st.radio(
                        "Task style", ["clean", "clean + leaky pair"],
                        key=f"reqpair_{kind}", horizontal=True,
                        help="A pair publishes BOTH arms under the mode "
                             "you chose above. In competition, submit to "
                             "both tasks — your own score gap IS your "
                             "inflation, measured on you. The leaky twin "
                             "declares itself in its manifest and ranks on "
                             "its own board; it never pollutes the clean "
                             "leaderboard.")
                    pair_operator = st.selectbox(
                        "Leak type (pair only)", list(OPERATORS),
                        key=f"reqop_{kind}",
                        format_func=lambda o: f"{o.replace('_', ' ')} — "
                                              f"{OPERATORS[o]}")
                    pin_window = st.checkbox(
                        "Pin the data window", key=f"reqwin_{kind}",
                        help="Unchecked = we pick the period (one of the "
                             "knobs our curriculum learns) and the dates "
                             "below are ignored. Checked = your dates are "
                             "honored as a hard constraint.")
                    w1, w2 = st.columns(2)
                    win_start = w1.date_input(
                        "From", date(2024, 1, 2),
                        min_value=date(2021, 1, 4),
                        max_value=date.today(),
                        key=f"reqws_{kind}")
                    win_end = w2.date_input(
                        "To", date.today(),
                        min_value=date(2021, 1, 4),
                        max_value=date.today(),
                        key=f"reqwe_{kind}")
                    st.caption("Any start from 2021-01-04 on; the window "
                               "must span at least ~6 months so there is "
                               "enough data to train, split, and label. "
                               "Used only when pinned.")
                else:
                    rounds = 1
                    if live_shape in ("chain", "portfolio"):
                        c1, c2 = st.columns(2)
                        rounds = c1.number_input(
                            "Rounds", 2, 10, 3, key=f"reqrounds_{kind}",
                            help="How many prediction rounds the challenge "
                                 "runs for.")
                        horizon_seconds = c2.number_input(
                            "Round interval (seconds)", 60, 604_800, 3600,
                            key=f"reqhs_{kind}",
                            help="Real time between rounds; also each "
                                 "round's horizon.")
                    elif live_shape == "polymarket":
                        horizon_seconds = 3600
                        st.caption("A prediction-market episode is organised "
                                   "per market: one allocation per selected "
                                   "market, marked to the observed odds "
                                   "path. Multi-step rebalancing is on the "
                                   "roadmap.")
                    else:
                        horizon_seconds = st.number_input(
                            "Horizon (seconds)", 60, 604_800, 3600,
                            key=f"reqhs_{kind}",
                            help="How far ahead you are predicting; also "
                                 "the submission deadline.")
                    if live_shape == "polymarket":
                        live_evidence = []
                        st.caption("Evidence for prediction-market episodes "
                                   "is automatic: real YES/NO odds series, "
                                   "odds charts, and order-book depth — "
                                   "news/filings do not apply to these "
                                   "markets.")
                    else:
                        live_evidence = st.multiselect(
                            "Evidence", ["charts", "news", "filings"],
                            default=["charts"], key=f"reqev_{kind}",
                            help="What the solver sees besides the quote: "
                                 "tick charts, fresh headlines, recent SEC "
                                 "filings.")
            submitted = st.form_submit_button("📨 Send request",
                                              type="primary")
        if submitted:
            sealed_key = _seal_api_key(api_key.strip()) if api_key.strip() else ""
            if not request_text.strip():
                st.error("Please describe the task — the description is "
                         "required.")
            elif api_key.strip() and not sealed_key:
                st.error("Key submission is disabled: the operator has not "
                         "published an encryption key.")
            elif not api_key.strip():
                st.error("Please add your DeepSeek API key — generation "
                         "runs on your own quota.")
            elif (kind == "historical" and pin_window
                  and (win_end - win_start).days < 180):
                st.error("A pinned window needs at least 180 calendar days "
                         "(≈6 months): room for lookback, a purged "
                         "train/test split, and forward labels.")
            elif (kind == "historical" and "chart" in (modalities or [])
                  and _chart_render_estimate(
                      assets_text, pin_window, win_start, win_end) > 500):
                st.error("Chart evidence would need too many rendered "
                         "images for this window × asset count (cap 500 — "
                         "one chart per asset per month). Narrow the "
                         "window, list fewer assets, or drop charts.")
            else:
                stamp = datetime.now(timezone.utc).isoformat()
                payload = {
                    "kind": kind,
                    "request_text": request_text.strip(),
                    "difficulty": difficulty,
                    "api_key_sealed": sealed_key,
                    "assets": [a.strip().upper()
                               for a in assets_text.split(",") if a.strip()],
                    "status": "pending",
                    "requested_at": stamp,
                }
                if kind == "historical":
                    if family != "let the generator decide":
                        payload["family"] = family
                    payload["horizon_trading_days"] = int(horizon_days)
                    if modalities:
                        payload["modalities"] = modalities
                    if pair_style == "clean + leaky pair":
                        payload["pair_operator"] = pair_operator
                    if pin_window:
                        payload["date_start"] = win_start.isoformat()
                        payload["date_end"] = win_end.isoformat()
                    payload["mode"] = mode
                else:
                    payload["live_shape"] = live_shape
                    payload["rounds"] = int(rounds)
                    payload["horizon_seconds"] = int(horizon_seconds)
                    payload["evidence"] = live_evidence
                _state_put(f"requests/req_{stamp.replace(':', '-')}.json",
                           payload)
                st.success("Request received! Track it below.")
        _request_status_list(kind)


def _request_status_list(kind: str) -> None:
    try:
        names = _state_list("requests")[-20:]
        recent = [(name, _state_get(f"requests/{name}")) for name in names]
    except Exception:
        return
    recent = [(n, r) for n, r in recent if r.get("kind") == kind]
    if not recent:
        return
    st.markdown("**Your requests**")
    tone = {"pending": ("⏳ pending", "slate"),
            "awaiting_review": ("👀 review me", "amber"),
            "changes_requested": ("🔁 regenerating", "amber"),
            "fulfilled": ("✅ ready", "teal"),
            "rejected": ("❌ rejected", "rose"),
            "failed": ("❌ failed", "rose")}
    for name, request in reversed(recent):
        label, colour = tone.get(str(request.get("status")),
                                 (str(request.get("status")), "slate"))
        with st.container(border=True):
            pills = [(label, colour)]
            if request.get("difficulty"):
                pills.append((f"asked: {request['difficulty']}", "indigo"))
            if request.get("measured_difficulty") is not None:
                pills.append(
                    (f"measured: {request['measured_difficulty']:.2f}", "teal"))
            if request.get("payer"):
                pills.append(("💳 your key" if request["payer"] == "guest_key"
                              else "🎁 shared quota", "slate"))
            if request.get("llm_cost_usd") is not None:
                pills.append((f"${request['llm_cost_usd']:.4f}", "slate"))
            st.markdown(
                f'<div class="vf-title">'
                f'{_pretty(str(request.get("task_id") or "")) or "…"}</div>'
                f'<div class="vf-sub">'
                f'{str(request.get("request_text", ""))[:90]}</div>'
                + _pills(pills), unsafe_allow_html=True)
            if request.get("reason"):
                st.caption(f"reason: {request['reason']}")
            if request.get("status") == "awaiting_review":
                st.write("The task is generated — open it above, then decide:")
                approve_col, change_col = st.columns(2)
                if approve_col.button("✅ Approve", key=f"appr_{name}"):
                    request["status"] = "fulfilled"
                    _state_put(f"requests/{name}", request)
                    st.rerun()
                change_text = change_col.text_input(
                    "What should change?", key=f"chg_{name}")
                if change_col.button("🔁 Request changes",
                                     key=f"chgbtn_{name}"):
                    if not change_text.strip():
                        st.error("Say what should change first.")
                    else:
                        request.setdefault("review_feedback", []).append(
                            change_text.strip())
                        request["status"] = "changes_requested"
                        _state_put(f"requests/{name}", request)
                        st.rerun()


# ---------- task detail sub-pages ----------

def detail_live(task_id: str) -> None:
    rows = {r.get("task_id"): r for r in _registry_rows()}
    row = rows.get(task_id)
    if row is None:
        st.error("Task not found.")
        return
    is_open = row.get("status") == "pending_resolution"
    st.title(_pretty(task_id))
    st.markdown(_pills([
        ("🟢 open" if is_open else "🏁 resolved", "teal" if is_open else "slate"),
        (f"⏱ {_time_left(str(row.get('resolve_after', '')))}",
         "amber" if is_open else "slate"),
        (f"📊 {row.get('metric_id')}", "indigo"),
        *[(a.strip(), "slate")
          for a in str(row.get("assets", "")).split(",") if a.strip()],
    ]), unsafe_allow_html=True)
    st.caption(f"{task_id} · deadline {row.get('resolve_after')} UTC")
    if is_open:
        st.subheader("📤 Submit your prediction")
        _submit_form(task_id, f"live_{task_id}",
                     expected_row_ids=row.get("row_ids") or None)
    else:
        st.info("This task has resolved — submissions are closed.")
    st.subheader("🏅 Scores on this task")
    try:
        _task_scores(_score_rows("scores"), task_id)
    except Exception as exc:
        st.error(f"scores unavailable: {exc}")


def detail_bundle(task_id: str, *, challenge: bool) -> None:
    prefix = "public_bundles" if challenge else "bundles"
    try:
        manifest = _bundle_manifest(prefix, task_id)
    except Exception as exc:
        st.error(f"could not read bundle: {exc}")
        return
    st.title(_pretty(task_id, manifest))
    baselines = manifest.get("baselines") or {}
    pills = [
        _lifecycle_pill(task_id) if challenge else ("🧪 practice", "slate"),
        ("🎯 challenge" if challenge else "🧪 practice",
         "indigo" if challenge else "slate"),
        (str(manifest.get("family", "?")).replace("_", " "), "teal"),
        *[(a, "slate") for a in (manifest.get("assets") or [])[:8]],
        (f"🗂 {len(manifest.get('artifacts') or {})} files", "slate"),
    ]
    if baselines.get("difficulty_estimate") is not None:
        pills.append(
            (f"difficulty {baselines['difficulty_estimate']:.2f}", "amber"))
    for extra in sorted(manifest.get("evidence") or {}):
        pills.append((f"➕ {extra.replace('_', ' ')}", "indigo"))
    st.markdown(_pills(pills), unsafe_allow_html=True)
    st.caption(task_id)
    st.download_button(
        "⬇️ Download task", data=_bundle_zip(prefix, task_id),
        file_name=f"{task_id}.zip", mime="application/zip",
        key=f"dl_{prefix}_{task_id}", type="primary")
    st.caption("manifest.json lists a sha256 for every file — verify after "
               "download." + ("" if challenge else
                              " Answers are included: score yourself locally."))
    if challenge:
        st.subheader("📤 Submit your prediction")
        _submit_form(task_id, f"hist_{task_id}",
                     expected_row_ids=_dataset_row_ids(prefix, task_id))
        st.subheader("🏅 Scores on this task")
        try:
            _task_scores(_score_rows("historical_scores"), task_id)
        except Exception as exc:
            st.error(f"scores unavailable: {exc}")
        lifecycle = _task_lifecycle().get(task_id, "approved")
        if lifecycle in ("pending review", "improving"):
            st.subheader("💬 Feedback on this task")
            st.caption("This task is still under review — feedback here "
                       "drives its regeneration.")
            _feedback_form(f"task_{task_id}", f"task_{task_id}",
                           "Anything wrong or confusing about this task?")
        else:
            with st.expander("🚩 Report an issue with this task"):
                st.caption("Found a data problem, leak, or scoring bug after "
                           "solving it? This is the highest-value feedback "
                           "there is — it goes straight into task memory.")
                _feedback_form(f"task_{task_id}", f"task_{task_id}",
                               "What did you find?")


# ---------- list pages ----------

def page_live() -> None:
    st.header("🔴 Live challenges")
    st.write("Predict what the market does next — open a task to submit "
             "before its deadline.")
    _request_section("live")
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
        _task_row(task_id, _pretty(task_id), [
            ("🟢 open" if is_open else "🏁 resolved",
             "teal" if is_open else "slate"),
            (f"⏱ {_time_left(str(row.get('resolve_after', '')))}",
             "amber" if is_open else "slate"),
            *[(a.strip(), "slate")
              for a in str(row.get("assets", "")).split(",") if a.strip()],
        ], "live")
    if not rows:
        st.info("Nothing live right now — request one above.")
    st.divider()
    st.subheader("💬 Shape the next live tasks")
    st.caption("Goes into task memory and steers upcoming live generation.")
    _feedback_form("live_feedback", "live_page",
                   "What should the next live tasks look like? "
                   "(assets, horizons, data you want)")


def page_historical() -> None:
    st.header("🎯 Historical challenges")
    st.write("Open a task to download it, submit predictions, and see its "
             "scores. We hold the answers.")
    _request_section("historical")
    rows = []
    try:
        for task_id in _bundle_ids("public_bundles"):
            rows.append((task_id, "challenge"))
    except Exception as exc:
        st.error(f"Bundle store unavailable: {exc}")
    try:
        for task_id in _bundle_ids("bundles"):
            rows.append((task_id, "practice"))
    except Exception:
        pass
    for task_id, kind in sorted(rows):
        try:
            prefix = "public_bundles" if kind == "challenge" else "bundles"
            manifest = _bundle_manifest(prefix, task_id)
        except Exception:
            manifest = {}
        mode_pill = (("🏆 competition · answers withheld", "indigo")
                     if kind == "challenge"
                     else ("🧪 practice · answers included", "slate"))
        _task_row(task_id, _pretty(task_id, manifest), [
            _lifecycle_pill(task_id),
            mode_pill,
            (str(manifest.get("family", "task")).replace("_", " "), "teal"),
            *[(a, "slate") for a in (manifest.get("assets") or [])[:6]],
        ], kind)
    if not rows:
        st.info("No tasks yet — request one above.")


def page_leaderboard() -> None:
    st.header("🏆 Leaderboard")
    st.caption("Live and historical never mix — different games. Under 5 "
               "rounds is flagged: one lucky round proves nothing.")
    for title, caption, subdir in [
        ("🔴 Live challenges",
         "Scored the moment each deadline resolves against the market.",
         "scores"),
        ("🎯 Historical challenges",
         "Scored server-side against withheld answers.",
         "historical_scores"),
    ]:
        st.subheader(title)
        st.caption(caption)
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
    "🏆 Leaderboard": page_leaderboard,
    "💬 Feedback": page_feedback,
}


def main() -> None:
    st.set_page_config(page_title="VACT-Fin", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
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
