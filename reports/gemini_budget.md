# Gemini API & Vertex AI Budget Estimate (`reports/gemini_budget.md`)

- **Model IDs:**
  - `gemini-4-argon` (reserved for human-review queue explanations, uncertain crop descriptions, and M12 600-crop benchmark per §5.4 & §9.3)
  - Flash-tier model (used for optional M7.5 bulk pseudo-box verification only if detector fine-tuning is triggered)
- **Budget Cap (Decision D3 Resolved):** `$25.00` maximum per bulk job (hard stop-and-ask gate per Ground Rule #8 & §9.4).

## Pricing & Token Cost Extrapolation (§9.4)

| Window | Input Rate (per 1M tokens) | Output Rate (per 1M tokens) | Cached Input Discount |
|---|---:|---:|---:|
| Introductory (through `2026-11-30`) | `$2.00` | `$10.00` | `95% off` |
| Standard (starting `2026-12-01`) | `$4.00` | `$20.00` | `95% off` |

## Job Cost Estimates

| Job | Image / Call Count | Est. Input Tokens / Call | Est. Output Tokens / Call | Est. Cost (Introductory) | Est. Cost (Post-2026-11-30) | Execution Status |
|---|---:|---:|---:|---:|---:|---|
| 50-Call Billing Pilot (§9.4) | `50` | `~450` | `~60` | `$0.08` | `$0.15` | `NOT_RUN` (No live `GOOGLE_API_KEY` or Vertex billing credentials configured on local Mac) |
| M12 Zero-Shot Benchmark (`field_holdout` + `ood_test`) | `700` | `~450` | `~60` | `$1.05` | `$2.10` | `NOT_RUN` (Pending `data/field_holdout/` collection and Vertex AI credentials; well under `$25.00` cap) |
| M7.5 Pseudo-Box Audit (Optional; skipped by default per §5.4) | `16,000` | `~450` | `~80` | `$27.20` (Argon) / `< $3.50` (Flash) | `$54.40` (Argon) / `< $7.00` (Flash) | `SKIPPED` (Skipped by default per §5.4; if triggered, must use Flash tier or stop and ask Harsh before exceeding `$25.00`) |
