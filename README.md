# VACT-Fin Console

Public console for VACT-Fin: browse point-in-time-correct financial task
bundles, submit predictions for pending live tasks before their resolve time,
and watch the resolved leaderboard.

Backed by a Supabase state mirror and an R2 bundle store. Scoring runs on the
maintainer's resolver daemon; submissions are timestamped server-side and
anything stamped after a task's resolution is refused.

Deployed on Streamlit Community Cloud; secrets are configured in the app
settings, never in this repository.
