# Paper baseline v0

- Data version: `v0.1` (feature version), manifest sha256 `ef5f71a5323d`, 40 races in the manifest (2025 and 2026).
- Test year: 2026. Training uses races before each test race only (expanding window).
- Test races: 16 in the 2026 manifest (`2026-01` to `2026-16`: Australia, China, Japan, Miami, Canada, Monaco, Barcelona, Austria, Britain, Belgium, Hungary, Netherlands, Italy, Spain, Azerbaijan, Bahrain). The brief said 15; the manifest holds 16. Needs checking before the paper cites a count.
- Config hash `efdd6e6d50e4`. Commit: see the `paper-baseline-v0` tag.
- Regenerate: `pitsense bench build --year 2025 2026`, then `pitsense bench run --out reports/baseline`, then `pitsense results --out reports/baseline --readme`.
