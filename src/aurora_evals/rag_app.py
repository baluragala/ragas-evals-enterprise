"""The system under test: Aurora Bank's customer-support RAG assistant, built with LangChain 1.x.

The assistant is deliberately *configurable* (chunking, top-k, prompt version) because the
point of evaluation is to compare versions of a system and pick the better one with evidence.

    app = AuroraRAG(PRESET_CONFIGS["baseline"])
    result = await app.ainvoke("What is the foreign transaction fee on Platinum?")
    result.response, result.retrieved_contexts, result.latency_s, result.total_tokens
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import APP_MODEL
from .knowledge_base import build_vector_store

# --------------------------------------------------------------------------------------
# Prompt versions. Treat prompts like code: version them and evaluate every change.
# --------------------------------------------------------------------------------------
PROMPTS: dict[str, str] = {
    "grounded_v2": """You are the customer-support assistant for Aurora Bank.

Rules:
1. Answer ONLY with facts found in the CONTEXT below. Do not use outside knowledge.
2. If the CONTEXT does not contain the answer, say: "I don't have that information in Aurora's
   policies. Please contact Aurora Bank directly." Do not guess.
3. Never give personalised investment, tax, or legal advice. Refer those questions to a licensed
   Aurora Wealth advisor.
4. Never repeat a full card number, account number, or SSN. Show at most the last 4 digits.
5. Be concise and end your answer with the sources you used, e.g. [Sources: credit_cards.md].

CONTEXT:
{context}""",
    # v3 = v2 + fixes for the failures found in notebook 02 (failure analysis):
    #   * advice questions were refused with no referral and no useful facts  → rule 3
    #   * multi-step fee questions were answered with a missing number        → rule 6
    #   * yes/no answers ignored exceptions such as the $50 overdraft buffer  → rule 7
    "grounded_v3": """You are the customer-support assistant for Aurora Bank.

Rules:
1. Answer ONLY with facts found in the CONTEXT below. Do not use outside knowledge.
2. If the CONTEXT does not contain the answer, say: "I don't have that information in Aurora's
   policies. Please contact Aurora Bank directly." Do not guess.
3. Never give personalised investment, tax, or legal advice. Say you can't advise on that, refer the
   customer to a licensed Aurora Wealth advisor, and then share any relevant published Aurora facts
   from the CONTEXT (rates, fees, insurance limits).
4. Never repeat a full card number, account number, or SSN. Show at most the last 4 digits.
5. Be concise and end your answer with the sources you used, e.g. [Sources: credit_cards.md].
6. If the question combines several facts or needs arithmetic, find EACH fact in the CONTEXT and show
   the short calculation.
7. Before answering yes/no about a fee, check the CONTEXT for exceptions, thresholds, waivers and
   benefits that change the answer.

CONTEXT:
{context}""",
    "loose_v1": """You are a friendly, confident banking assistant for Aurora Bank.
Use the context below if it is helpful, and your general banking knowledge otherwise.
Always give the customer a complete, decisive answer and a recommendation.

Context:
{context}""",
}


@dataclass(frozen=True)
class RAGConfig:
    """Every knob that changes the assistant's behaviour. Frozen so it can be logged and hashed."""

    name: str
    chunk_size: int = 800
    chunk_overlap: int = 100
    header_aware: bool = True
    top_k: int = 4
    prompt_version: str = "grounded_v2"
    model: str = APP_MODEL
    temperature: float = 0.0


PRESET_CONFIGS: dict[str, RAGConfig] = {
    # The version currently "in production".
    "baseline": RAGConfig(name="baseline"),
    # A cost-cutting proposal: smaller chunks, fewer of them, no header awareness.
    "naive_chunking": RAGConfig(
        name="naive_chunking", chunk_size=200, chunk_overlap=0, header_aware=False, top_k=2
    ),
    # A "make it friendlier" prompt proposal that quietly drops the grounding rules.
    "loose_prompt": RAGConfig(name="loose_prompt", prompt_version="loose_v1"),
    # The fix built from failure analysis (notebook 02): better prompt + more context.
    "candidate_v3": RAGConfig(name="candidate_v3", prompt_version="grounded_v3", top_k=6),
}


@dataclass
class RAGResult:
    """Everything Ragas needs to grade one answer, plus operational telemetry."""

    user_input: str
    response: str
    retrieved_contexts: list[str]
    retrieved_sources: list[str]
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    config_name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return asdict(self)


def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs)


class AuroraRAG:
    """Retrieve → stuff into prompt → generate. Classic 2-step RAG with LangChain 1.x LCEL."""

    def __init__(self, config: RAGConfig | str = "baseline"):
        self.config = PRESET_CONFIGS[config] if isinstance(config, str) else config
        store = build_vector_store(
            self.config.chunk_size, self.config.chunk_overlap, self.config.header_aware
        )
        self.retriever = store.as_retriever(search_kwargs={"k": self.config.top_k})
        prompt = ChatPromptTemplate.from_messages(
            [("system", PROMPTS[self.config.prompt_version]), ("human", "{question}")]
        )
        llm = ChatOpenAI(model=self.config.model, temperature=self.config.temperature, max_retries=5)
        # LCEL pipeline: a Runnable that maps {"context", "question"} → AIMessage
        self.chain = prompt | llm

    async def ainvoke(self, question: str) -> RAGResult:
        start = time.perf_counter()
        docs = await self.retriever.ainvoke(question)
        message = await self.chain.ainvoke({"context": format_context(docs), "question": question})
        latency = time.perf_counter() - start

        usage = message.usage_metadata or {}
        return RAGResult(
            user_input=question,
            response=message.content,
            retrieved_contexts=[d.page_content for d in docs],
            retrieved_sources=[d.metadata["source"] for d in docs],
            latency_s=round(latency, 3),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            config_name=self.config.name,
        )
