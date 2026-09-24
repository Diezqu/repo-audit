# Draft repository question set

`questions.jsonl` contains ten draft questions about the public [FastMCP repository](https://github.com/jlowin/fastmcp), pinned to commit `f4ae8bb0af04cb315eef262d38433af4b71d9c38`. The source package at this commit is under `fastmcp_slim/fastmcp/` in the monorepo. Each line is one JSON object with `id`, `question`, `repo_url`, `repo_commit`, `expected_answer`, `reference_citations`, and `review_status`. Each citation has a repository-relative `file`, inclusive one-based `line_start` and `line_end`, and an exact `snippet` from those lines at the pinned commit.

These reference answers were drafted by AI from source review. They are **not human gold labels**. Every record is marked `draft_requires_human_review`. A human reviewer should check that each question is unambiguous, the cited code supports every answer clause, the answer states appropriate limits, and the set measures repository understanding rather than string recall. Questions `FMCP-02`, `FMCP-03`, and `FMCP-10` deliberately ask where inference must stop. The final evaluation quality metric and acceptance threshold are pending human judgment; this file claims no measured quality, model result, or production readiness.

The corpus was assembled from a clean local checkout at the pinned commit. Source files were read as data via `git show`; target repository code was not imported or executed. No network, external instructions from the target, or paid APIs were used. A future runner may produce candidate responses and actual results, but it should keep this draft corpus distinct from reviewed labels and report its scoring method.

To verify citation integrity against a local checkout, check its HEAD and compare each `snippet` with the inclusive line range in `git show <repo_commit>:<file>`. Human review is still required even when those mechanical checks pass.

## Offline validation

Use a fresh dedicated checkout of FastMCP at `f4ae8bb0af04cb315eef262d38433af4b71d9c38`, and pass its Git top-level directory as `--repo`. Keep virtual environments, caches, and scratch files outside that checkout: evaluation rejects ignored files as well as ordinary untracked or modified files, since repository tools could otherwise read code absent from the pinned commit.

```sh
PYTHONPATH=src python scripts/evaluate.py validate-corpus --repo /path/to/fastmcp
```

This checks the actual Git HEAD, a clean working tree including untracked and ignored files, all ten questions, and every snippet against its inclusive line range at the pinned commit. It makes no model or network calls. The current draft has 20 source references. A matching snippet means the reference is mechanically intact; it does not establish that the prose answer is correct.

## Run bounded measurements

The runner requires a new output path outside both the engine and source repositories. It refuses to overwrite an earlier record. Select one question first to bound calls; `--repeats 3` is available for a latency range after the setup is verified.

```sh
PYTHONPATH=src python scripts/evaluate.py run \
  --repo /path/to/fastmcp --mode demo --id FMCP-01 \
  --output /tmp/repo-audit-demo.jsonl
PYTHONPATH=src python scripts/evaluate.py summarize \
  --records /tmp/repo-audit-demo.jsonl
```

Demo mode forces both model tiers off, even if credentials are present in the environment. Its results are synthetic and **must not be used as quality metrics**. Do not publish a demo report as a real repository answer.

For real runs, export all six `CHEAP_MODEL`, `CHEAP_API_KEY`, `CHEAP_BASE_URL`, `FLAGSHIP_MODEL`, `FLAGSHIP_API_KEY`, and `FLAGSHIP_BASE_URL` variables in the shell before invoking the script. Run from the engine repository, not from the target repository. The evaluator requires both tiers and a separate acknowledgment flag:

```sh
PYTHONPATH=src python scripts/evaluate.py run \
  --repo /path/to/fastmcp --mode real --allow-api \
  --id FMCP-01 --repeats 3 --max-concurrency 8 \
  --output /tmp/repo-audit-real-fmcp01.jsonl
```

The runner enables the deterministic citation verifier. Each JSONL record includes source HEAD, dirty state, and ignored-file presence before and after, engine identity, corpus SHA-256, Python and dependency versions, tier model names, budget, task count, configured concurrency limit, elapsed time, report, raw and checked claims, and callback token usage by tier. Worker provider failures carry a sanitized runtime `worker_error` type on their claims, plus `failed_worker_count` and `failed_worker_ids` in the record; ordinary evidence insufficiency has no worker error. When some Workers fail, the stable-source result is `partial_failure`; when all fail, it is `error`. Both keep available claims and report but are excluded from real latency measurements. A graph exception yields a sanitized `error_type` and no report. A source change, including an ignored file appearing during a run, yields `source_changed`, drops the report, and stops the batch. The task count is the Planner's number of subtasks; configured concurrency is a separate limit, not observed simultaneous API calls.

`raw_supported_count` is the Worker's own status. `raw_citation_count` counts every claim citation. `citation_valid_claim_count` counts claims whose deterministic `citation_status` is `valid` and have citations. `citations_on_all_valid_claims` counts citations on those claims. `semantic_verified_count` counts non-null independent verdicts; in this deterministic first round it should be zero. These are not answer accuracy measurements. Token fields are `null` if any model response omitted usage or a call failed in that tier; failed calls are counted without recording provider error text.

Cost is `null` unless an explicit rate JSON file is passed using `--rates` and all usage is known. Rates use the actual configured tier model prices and must include a currency, source, date, and finite nonnegative numeric input/output prices per million tokens; booleans and nonfinite values are rejected before a run. Arithmetic overflow also yields `null`. For example, **replace these illustrative values with verified rates before use**:

```json
{
  "currency": "USD",
  "source": "provider pricing page URL",
  "date": "YYYY-MM-DD",
  "tiers": {
    "cheap": {"input_per_million": 1.0, "output_per_million": 2.0},
    "flagship": {"input_per_million": 3.0, "output_per_million": 6.0}
  }
}
```

The draft references remain unreviewed. Human scoring needs a separate, explicit label file and review protocol; no expected-answer string match or snippet overlap is accepted as semantic truth. The summary reports separate `error_count`, `partial_failure_count`, and combined `operational_failure_count`; real-run latencies include only eligible `ok` records. Quality metrics remain pending human labels.
