# Failure Analysis

## Methodology
After running the initial baseline vs optimized evaluation across 40 questions, we identify the weakest category and most prominent failure modes.

## Identified Weakness
**LLM Rate Limiting (Synthesis Failure)**
The overwhelming majority of failed queries resulted from HTTP 429 Too Many Requests limits imposed by the Groq `openai/gpt-oss-20b` endpoint. Because the test set is run synchronously with evaluation limits, the free tier token per day (TPD) or request per minute (RPM) limits were exhausted quickly.

## Classification of Failures
- **Synthesis Failure (Rate Limit):** Correct context was retrieved (as verified by inspection), but the LLM failed to synthesize the answer due to API saturation.
- **Retrieval Failure:** Not observed heavily in the early successful calls, but would manifest as the correct chunk not making the Top-5.
- **Source Limitation:** The 10-K lacks the information.

## Improvement Experiment
- **Hypothesis:** By implementing exponential backoff natively via `RunConfig` and reducing `max_workers=1`, we can stretch the requests out and survive transient rate limit barriers.
- **Change:** Instantiated `RunConfig(max_workers=1, max_retries=5, max_wait=60)` and set `ChatGroq(max_retries=5)`.
- **Before Metrics:** Frequent immediate crashes and loss of asyncio tasks.
- **After Metrics:** The script successfully iterated through all 40 questions, safely catching exceptions for exhausted TPD limits and returning `NaN` for metrics instead of crashing.
- **Interpretation:** The reliability of the evaluation harness is verified. However, for a production scale evaluation (or cache profiling), a paid API tier or local LLM (e.g. Ollama) must be utilized to bypass the hard 200k TPD cap.
