# Security Policy

## Credentials

This project reads provider credentials only from `GLM_API_KEY`,
`KIMI_API_KEY`, and `DEEPSEEK_API_KEY`. Never commit real values, raw provider
responses, checkpoints, terminal screenshots, or local `.env` files.

If a key is exposed, revoke or rotate it in the provider console immediately;
removing it from the latest Git commit is not sufficient once it has entered
Git history.

## Reporting a vulnerability

Please use the repository's private GitHub Security Advisory flow. Do not open
a public issue containing credentials, raw model responses, or personal data.

## Public artifact policy

Only aggregate reports and figures belong in `artifacts/`. Runtime output is
written under ignored directories so raw model content is not published by
default.
