# Security policy

## Supported version

Only the current `main` branch is maintained. This prototype has not undergone
an independent security audit or certification.

## Reporting a vulnerability

Do not disclose surveillance-system vulnerabilities in a public issue. Contact
the repository owner privately through their verified GitHub contact channel
and include reproduction steps, impact, and affected revision. Do not include
real camera credentials, faces, number plates, or operational footage.

## Deployment requirements

- Set `IBVAP_DASHBOARD_PASSWORD` and `IBVAP_SESSION_SECRET` through a secret
  manager or protected process environment. Never put credentials in Git.
- Use HTTPS, either directly with `IBVAP_TLS_CERTFILE` and
  `IBVAP_TLS_KEYFILE`, or behind a trusted TLS reverse proxy. Set
  `IBVAP_REQUIRE_HTTPS=true` and `IBVAP_COOKIE_SECURE=true`.
- For proxy TLS, set `IBVAP_HTTPS_BEHIND_PROXY=true` and set
  `IBVAP_FORWARDED_ALLOW_IPS` only to trusted proxy addresses. Never use `*`.
- Set `IBVAP_ALLOWED_HOSTS` to approved public hostnames and
  `IBVAP_ALLOWED_ORIGINS` to exact externally visible HTTPS origins.
- Restrict network access with a firewall/VPN. The shared operator account is
  suitable for a prototype, not a large organization; production deployments
  should place the app behind organizational SSO and role-based access.
- Rotate credentials, TLS material, and session secrets regularly.
- Protect and rotate `logs/audit.jsonl`, `logs/alerts.csv`, and recorded output.
- Disable ANPR and SCRFD/GFPGAN face enhancement unless their use is authorized
  and necessary. Review model licenses before installation.

## Data handling

Alerts, footage, faces, and number plates may be sensitive personal or
operational data. Define access, encryption, retention, deletion, incident
response, and lawful-use policies before collecting real data. The application
does not substitute for those organizational controls.

GFPGAN output is synthesized and can change identity-relevant details. Preserve
the raw source, restrict enhanced-thumbnail access, apply short retention, and
never present an enhanced image as an authentic evidentiary reconstruction.

## Known scope limitations

In-memory rate limits and sessions are per process. Run one application worker,
or move authentication, rate limiting, and session management to a trusted
shared gateway before scaling horizontally.
