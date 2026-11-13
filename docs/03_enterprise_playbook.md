# 03 · Enterprise playbook: running evals for real

A checklist of practices this project demonstrates, and what they look like at company scale.

## 1. Treat the golden dataset as a product
- [ ] **Owned** by a named team, with SMEs (support, compliance, legal) writing or approving references.
- [ ] **Versioned in git** next to the code (`data/golden/*.jsonl`); changes reviewed like code.
- [ ] **Tagged** for slicing: question type, product line, language, customer segment, risk level.
- [ ] **Deliberately hard:** multi-hop, numeric, unanswerable, adversarial (prompt injection, advice-seeking), PII.
- [ ] **Sized for the decision:** ~20 cases to learn (this repo); 100–300 for a release gate; thousands for
      fine-grained comparisons. With *n* cases, one flip moves an average by 1/*n*.
- [ ] **Fed by production:** every incident and every flagged conversation becomes a new case.
- [ ] **Bootstrapped, not replaced, by synthetic data** (notebook 06): always human-reviewed.
- [ ] **Re-baselined** whenever the dataset changes (old and new scores are not comparable).

## 2. Design the metric suite around decisions
- [ ] Cover **retrieval, generation, business rules, and operations** (latency, cost) together; single metrics
      get gamed (the `loose_prompt` config improved tone while breaking compliance).
- [ ] Prefer **code checks** for anything exact (citations, formats, side effects, PII regexes).
- [ ] Use **LLM judges** for meaning and policy, with **concrete checklists** stating what fails *and* what is allowed.
- [ ] Give each judge **one responsibility** and say so in its prompt ("accuracy is graded elsewhere"); judges
      that drift into other dimensions make gates flaky.
- [ ] Watch for **flaky gates**: re-run the same config; if the decision flips, fix the judge before trusting the gate.
- [ ] Choose metrics **per scenario type** (goal accuracy ≠ refusals) and mark non-applicable rows as **NaN**,
      never as a fake 0 or 1.
- [ ] Know each metric's **edge cases** (0/0 cases in F1 and topic adherence, refusals in answer relevancy).

## 3. Earn trust in the judge
- [ ] Separate **judge** from **system under test** (different client, often a stronger model).
- [ ] **Calibrate** each judge metric against human labels: accuracy, **Cohen's κ**, false-pass vs false-fail.
      Common bar: κ ≥ 0.6 to use in gates; higher for safety.
- [ ] Keep a **held-out** labelled set you never tune prompts on.
- [ ] **Re-calibrate** when the judge model or prompt changes. Pin judge model versions where the provider allows.
- [ ] Reduce variance on hard judgements with **majority voting** (`MajorityVote`), and spot-check judge reasons.
- [ ] Periodically sample judge decisions for **human audit**.

## 4. Make results reproducible
- [ ] Save **per-sample** outputs, scores and **judge reasons**, not just averages.
- [ ] Save **provenance**: timestamp, app/judge models, library versions, full config (`summary.json`).
- [ ] **Pin** fragile dependencies (`requirements.txt` explains the `langchain-community==0.4.1` pin).
- [ ] Use temperature 0 for app and judge in evaluation runs, and still expect some variance.

## 5. Gate releases automatically
- [ ] Thresholds in a reviewed file (`config/quality_gates.yaml`), set with the business owner.
- [ ] **Overall + slice + regression** gates. Slices catch what averages hide; regression catches slow decay.
- [ ] Gates on metrics that **cannot be computed fail closed** (NaN never passes).
- [ ] The eval CLI **exits non-zero** on failure, so CI blocks the merge.
- [ ] A committed, reviewed **reference baseline**, changed only by explicit promotion.
- [ ] **Tiered CI** (`.github/workflows/evals.yml`): harness unit tests on every push → full gates on relevant
      PRs → nightly drift run.
- [ ] Attach `scores.csv` to the PR as evidence for reviewers.

## 6. Control cost and latency of the evals themselves
- [ ] **Cache** judge calls (Ragas `DiskCacheBackend`; cache directory restored in CI).
- [ ] **Bound concurrency** (`gather_limited`) and configure retries/timeouts for rate limits.
- [ ] Disable library telemetry (`RAGAS_DO_NOT_TRACK=true`). In ragas 0.4.3 it also made scoring ~5× faster.
- [ ] Run cheap subsets on every PR (`--limit`), the full suite before release.
- [ ] Spend expensive judges (strong models, multiple votes) only where they change decisions.

## 7. Test the evaluation harness
- [ ] Unit-test converters, gate logic, code metrics and dataset schemas (`tests/test_offline.py`).
- [ ] Add a cheap **live smoke test** (`tests/test_live_smoke.py`), skipped when no key is present.
- [ ] When a score surprises you, **check the plumbing first**: the built-in LangChain→Ragas converter silently
      dropping tool calls made a perfect agent score 0.

## 8. Monitor production
- [ ] Sample live traffic and score it with **reference-free** metrics (faithfulness, compliance, citations).
- [ ] Dashboards and alerts on rolling averages **per slice** (product, channel).
- [ ] Route failures to **human review**; confirmed failures go into the golden dataset.
- [ ] Combine with explicit user feedback and business KPIs (escalation rate, CSAT); evals are a proxy.

## 9. Governance & privacy (regulated industries)
- [ ] Scrub or tokenise **PII** before sending production logs to an external judge.
- [ ] Keep an **audit trail**: which version shipped, with which eval results, approved by whom.
- [ ] Document the purpose and known limitations of each judge metric (a "metric card").
