# Local arena review screenshots

Created 2026-11-03T18:37:40.030670+00:00 by Codex GPT-6 session `/root/arena_profiles` on the owner's Mac, original directory `/Users/cdjk/github/neptts-bench/benchmark/judge_calibration_v1/screenshots`. Owner: Ampixa / Arbiter nenglish-73. Status: active PR review evidence; last review 2026-11-04; next review 2026-11-11 or PR closure. NOT on HF. Project software licence applies; screenshots contain only local UI and synthetic fixtures, no real audio or private submissions.

Inventory: `before.png` shows the unchanged arena verified-evaluations page; `after.png` shows the completed six-profile tabs, measured Whisper-small calibration and single-take filter. They were generated against `python3 -m submission_app.server` bound to loopback port 5197, with a fresh empty local SQLite store under `.local/judge-profiles-preview`. The final capture used the connected Playwright MCP at a 1440 × 1100 CSS-pixel viewport because the prior playwright-cli executable was unavailable. The browser was scrolled to the profile tabs. No real submission audio or private scores were loaded. Reproduce with the source revision in the PR and the same local run. No backup or remote copy was made; Git/PR may retain these files. Delete transient browser logs/state within 24 hours after review; screenshots retain value as the requested before/after PR evidence until PR closure.

|File|Bytes|SHA256|
|---|---:|---|
|before.png|137203|`6fa0374c8f55326de571e80de77c66fde779b62cc1c85128134b46de399060b4`|
|after.png|153550|`21255178f839f530ff294ec68423157f5fa9835fc54dd40c3acefb82f719965c`|

Final visual check 2026-11-03 20:39 UTC: the after screenshot was inspected at full viewport size. The browser exposed all six profile tabs, `single_take` as the default generation filter, the exact Whisper-small profile ID, PV/Fresh n and CER/CIs, loanword policy and the spontaneous-Nepali status. Selecting D32 streaming displayed its local-only status and 11.80% / 18.27% legacy corpus CER; changing the filter to best-of-N preserved the chosen profile. The only browser console error was a missing optional favicon (`/favicon.ico`, 404). The local server and browser were stopped after capture. An initial after PNG was superseded after a wording-only availability change; its size/hash and reason are in `PROGRESS.md` and it is no longer byte-restorable.

Transient cleanup 2026-11-03T18:38:00.692493+00:00: browser was closed and the following task-created snapshot/console diagnostics were removed after the before image was visually verified. They contained the same empty local preview; original screenshot remains, browser diagnostics can be reproduced and are no longer restorable byte-for-byte. No existing source or other job data was deleted. Reclaimed 8,149 apparent bytes (12,288 allocated bytes).

|Deleted path|Bytes|SHA256 before deletion|
|---|---:|---|
|.playwright-cli/console-2026-11-03T18-33-56-125Z.log|138|`18d5f9b447f3276b2d0abd5f24832f78b0486cc71fe8e7c12c85b07e88b41153`|
|.playwright-cli/page-2026-11-03T18-33-56-277Z.yml|8011|`c9f2c7f74aeb45809f0d691a10b68101e495d9aa06031d2b1ff6c9e759380d18`|

Disposable cleanup 2026-11-03T20:42:24.552864+00:00: no open files in the exact target directories (`lsof +D`), no listener on port 5197, browser and local server stopped. These files served only the empty localhost screenshot and browser checks; the before/after PNGs remain. The exact deleted file inventory is below.

| Deleted path | Bytes | SHA256 |
|---|---:|---|
| `.playwright-mcp/console-2026-11-03T20-38-01-004Z.log` | 138 | `da8ad9fc92d0bdb84a78b5fe2b8c8ee18780a8c0b993d8b517f3e49a64f2772e` |
| `.playwright-mcp/page-2026-11-03T20-38-02-371Z.yml` | 8240 | `ccc9b411ddd6230e7cbd1fcdc369cf102ac8ef8bc0eb9478d6e3186ec0417d3b` |
| `.playwright-mcp/page-2026-11-03T20-38-13-790Z.yml` | 9462 | `2a3c23378d10734193327c748fb4582a3c920202c049a755960266356ba75783` |
| `.playwright-mcp/page-2026-11-03T20-38-48-005Z.yml` | 9654 | `a4a7e91b6e96e9dadc7b08de3f4b3d5c7b2c8575aa52ff810787502a67029347` |
| `.playwright-mcp/page-2026-11-03T20-39-18-750Z.yml` | 9654 | `a03db725f0465d51b2d9d387242ae2ca65143ebb9d584604ca47d56603088c52` |
| `.playwright-mcp/page-2026-11-03T20-39-38-334Z.yml` | 9414 | `6be5efbb66ee27d9b4f733b46a6f15f12d61a6b5d7ad1a60be36fd5bab90c778` |
| `.playwright-mcp/page-2026-11-03T20-39-42-908Z.yml` | 9730 | `f9ac3fb4cb0e0705f67da762ea1694d6dbc2a0a5e81df2a2e2c5c2ceab3bdfc8` |
| `.playwright-cli/README.md` | 976 | `b51dfc6d4d921d683c8f55e178f9cf43c25741f1f7f2a778b32d49dde0266422` |
| `.playwright-cli/console-2026-11-03T18-44-40-427Z.log` | 138 | `da50f84a4c72fdbb0fc94fed01de9f93562594341a711402c6a83896a50cb9d9` |
| `.playwright-cli/page-2026-11-03T18-44-40-525Z.yml` | 9064 | `dcbfea3109c2e69a2e2202f8d8d93f230276327dde1ddae3d57501af15d600b9` |
| `.local/judge-profiles-preview/README.md` | 860 | `08c0073cd22959e76b43ad7b7263541526f720f2e61df9102495283ee2952d4b` |
| `.local/judge-profiles-preview/submissions.db` | 49152 | `fe05d80d6fd10f9908993770267d67eafd60d5dfe20ea6d8b125b4169644e142` |

Deleted 12 files, 116,482 apparent bytes / 151,552 allocated file bytes; measured filesystem free-space change +147,456 bytes (host-concurrent activity may affect this delta). The deleted browser/database state is reproducible from the local server and has no remaining unique content; it is not byte-restorable.
