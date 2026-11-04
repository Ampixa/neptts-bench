# Arbiter post-merge import

The PR is intentionally unmerged. **No D32 result has been imported to the live arena.** These commands are for the Arbiter only after review, merge and a successful `arena.yml` deployment. The five approved Whisper reports remain immutable in the original `submissions` rows; D32 imports use the new `profile_evaluations` table and require their own approval. The three reference systems are report context only and are never imported.

Created 2026-11-04 UTC by Codex `/root`, job `arena_d32_rescore_v1`; owner Ampixa / Arbiter nenglish-73, active review instructions, next review 2026-11-11. This source document owns itself; no secrets or audio, NOT on HF.

From the reviewed repository checkout, verify PR #7 was merged. The state check below stops the sequence while it is open.

```sh
PR_NUMBER=7
test "$(gh pr view "$PR_NUMBER" --json state --jq .state)" = MERGED
gh pr view "$PR_NUMBER" --json state,mergeCommit,url
MERGE_SHA=$(gh pr view "$PR_NUMBER" --json mergeCommit --jq '.mergeCommit.oid')
DEPLOY_RUN=$(gh run list --workflow arena.yml --commit "$MERGE_SHA" --event push --json databaseId --jq '.[0].databaseId')
test -n "$DEPLOY_RUN"
test "$DEPLOY_RUN" != null
gh run watch "$DEPLOY_RUN" --exit-status
```

The importer verifies that the deployed `SOURCE_COMMIT` equals the exact merged commit. It reads credentials only in VM memory via `sudo`, checks all five legacy report hashes before and after each operation, verifies the submission/audio binding through the reviewer API, and uses only numeric report JSON already present in the deployed source. It fails on any changed legacy report or conflicting D32 result. Repeated successful imports leave identical existing results alone. A partial failure does not roll back earlier independent imports; inspect the error, resolve it, and rerun the same command.

Set `ARENA_SSH_HOST` (user@host of the arena VM) and `ARENA_SSH_KEY` (operator key path) from the private operator notes first.

Import into review, with **no publication**:

```sh
ssh -i "$ARENA_SSH_KEY" "$ARENA_SSH_HOST" \
  "python3 /opt/ampixa-arena/current/benchmark/arena_d32_rescore_v1/import_results.py --deployed-commit '$MERGE_SHA' --mode import"
```

After inspecting the five D32 reports and the profile card, publish the separate profile results:

```sh
ssh -i "$ARENA_SSH_KEY" "$ARENA_SSH_HOST" \
  "python3 /opt/ampixa-arena/current/benchmark/arena_d32_rescore_v1/import_results.py --deployed-commit '$MERGE_SHA' --mode approve"
```

Each command emits only sanitized IDs, states, deployed revision and before/after legacy report hashes. Save its output with this job's review evidence (and document that saved receipt). No private audio or weights are required for import. The exact endpoints are `POST /api/admin/submissions/{id}/profiles/{profile_id}/results` and then `/decision` with `{"action":"approve","note":"..."}`; both require the reviewer cookie and `X-Neptts-Request: submission`. The helper never submits reference-system results.

Verify the D32 leaderboard with `selection=all` because legacy declaration eligibility stays unchanged. The default leaderboard remains Whisper and the default selection filter remains single take:

```sh
PROFILE_ID=d32v2-s60000-full-translit-279b835f5fe9
curl --fail --silent --show-error "https://ampixa.benchmarks.cloud/api/leaderboard?selection=all&profile=$PROFILE_ID"
curl --fail --silent --show-error "https://ampixa.benchmarks.cloud/api/leaderboard?selection=all&profile=whisper-small-ne-048909d9dafd"
```

Reproduction of the scoring itself is documented in `neptts-eval/README.md`; it requires Ampixa's private model and the authorized audio originals. The job-local submitted WAV/ZIP copies were deleted after scoring with a manifest; the original arena uploads remain intact. Never copy credentials into shell arguments or paste them into review, and never transfer submitter audio to an API.
