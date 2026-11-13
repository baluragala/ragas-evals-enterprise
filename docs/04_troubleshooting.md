# 04 · Troubleshooting

Every issue below was actually hit while building this project with **ragas 0.4.3 + LangChain 1.x**.

## Installation / import

### `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` on `import ragas`
ragas 0.4.3 imports a module that **langchain-community 0.4.2 removed**.
**Fix:** `pip install "langchain-community==0.4.1"` (already pinned in `requirements.txt`).

### Colab prints `ERROR: pip's dependency resolver does not currently take into account...`
Colab preinstalls many packages; pip warns when our pins differ from what some unrelated package wants.
These warnings are harmless for this project. If an import then fails, use **Runtime → Restart session** and
re-run the setup cell.

### `OPENAI_API_KEY is not set`
* Colab: 🔑 Secrets (left sidebar) → add `OPENAI_API_KEY` → enable **Notebook access** → re-run setup.
* Local: `cp .env.example .env` and add the key, or `export OPENAI_API_KEY=sk-...`.

### `git clone` fails in Colab
`REPO_URL` in the setup cell must point at a repo you can access. If you forked the project, change it. Without
GitHub: upload the project zip to Colab, `!unzip ragas-evals-enterprise.zip`, `%cd ragas-evals-enterprise`, then
run the setup cell (the clone step is skipped when the folder exists).

## Running evaluations

### Evaluations are very slow; judge calls seem to run one at a time
Ragas' anonymous usage tracking makes a **synchronous** HTTP request inside the async event loop, which blocks all
concurrent judge calls. Measured: 8 parallel Faithfulness scores take 13.8 s with tracking, 2.75 s without.
**Fix:** `RAGAS_DO_NOT_TRACK=true`. `aurora_evals` sets it automatically on import, and the notebooks set it too.

### `RuntimeError: Cannot call sync score() from an async context. Use ascore() instead.`
Jupyter/Colab always has a running event loop. Use `await metric.ascore(...)` in notebooks. In scripts, wrap your
code in `async def main()` and call `asyncio.run(main())`.

### `PicklingError: Can't pickle <class 'ragas.metrics.base.DiscreteResponseModel'>`
`DiscreteMetric` creates its response model class dynamically, and Ragas' disk cache can't pickle it.
**Fix:** give `DiscreteMetric`s an **uncached** judge: `get_judge_llm(cache=False)` (done in `business_suite`).

### `IncompleteOutputException: The output is incomplete due to a max_tokens length limit`
Long answers produce long statement-by-statement verdicts that exceed Ragas' default `max_tokens=1024`.
**Fix:** `llm_factory(..., max_tokens=4096)` (done in `judges.get_judge_llm`).

### `ValueError: Collections metrics only support modern InstructorLLM`
Metrics from `ragas.metrics.collections` need a judge built with `llm_factory(model, client=AsyncOpenAI())`, not a
LangChain model wrapped in `LangchainLLMWrapper`. Use `aurora_evals.judges.get_judge_llm()`.

### Rate limits (`429`) / timeouts
Lower `concurrency` (e.g. `evaluate_rag(..., concurrency=4)`). The OpenAI client already retries 6 times with
backoff. Failed metric calls become NaN with the error in an `<metric>__error` column (`run.errors()`), and they
fail any gate on that metric.

### Scores change between runs
Expected: both the app and the judge are LLMs. Look at *which rows* changed, re-run 2–3 times for important
decisions, and grow the dataset. Judge calls are cached, so identical answers get identical verdicts on re-runs.

## Agent evaluation

### Agent tool metrics are 0 even though the agent clearly called the right tools
You are probably using `ragas.integrations.langgraph.convert_to_ragas_messages`. In ragas 0.4.3 it reads tool calls
from `AIMessage.additional_kwargs`, where LangChain 1.x no longer puts them. **Fix:** use
`aurora_evals.messages.to_ragas_messages` (reads `AIMessage.tool_calls`).

### `tool_call_accuracy` is low for correct searches
Exact-match on free-text arguments. Add the tool/argument to `messages.FREE_TEXT_ARGS` so it is canonicalised on
both sides.

### `goal_accuracy` is 0 for a correct refusal, or flips between runs
Goal accuracy infers an *end state*; refusals don't produce one. Set `reference_goal: null` for refusal scenarios
(→ NaN, not applicable) and grade them with `refusal_quality`. For variance, use a stronger judge and
`MajorityVote`.

## Synthetic test generation

### `RuntimeError: Event loop is closed` / `APIConnectionError: Connection error.` in a script
The generator runs several internal async stages; LangChain's cached async HTTP client gets bound to a loop that is
later closed. It works in notebooks (one long-lived loop). In scripts, call the generator from inside a running
loop:
```python
async def main():
    testset = generator.generate_with_langchain_docs(docs, testset_size=8)
asyncio.run(main())
```
