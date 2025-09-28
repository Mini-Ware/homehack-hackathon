from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # The import is optional to keep startup graceful during testing.
	from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime when package missing.
	OpenAI = None  # type: ignore


logger = logging.getLogger("policy_nav_backend")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.0

EXA_API_BASE_URL = os.getenv("EXA_API_BASE_URL", "https://api.exa.ai")
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
EXA_API_KEY_ENV = "EXA_API_KEY"


app = FastAPI(
	title="PolicyNav Intelligence API",
	version="0.1.0",
	description=(
		"Backend services for the Singapore Policy Navigator UI. Provides semantic search "
		"and conversational policy assistance with web-grounded responses."
	),
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class SearchResult(BaseModel):
	"""Lightweight representation of a single web result."""

	title: str = Field(..., description="Title of the source result.")
	url: str = Field(..., description="Canonical URL for the result.")
	snippet: Optional[str] = Field(None, description="Short summary snippet from the source.")


class SearchRequest(BaseModel):
	query: str = Field(..., min_length=3, description="End-user search query.")
	max_results: int = Field(5, ge=1, le=10, description="Number of results to fetch from Exa.")


class SearchResponse(BaseModel):
	summary: str
	used_web_results: bool
	results: List[SearchResult]


class ChatMessage(BaseModel):
	role: Literal["user", "assistant", "system"]
	content: str


class ChatRequest(BaseModel):
	message: str = Field(..., min_length=1)
	history: List[ChatMessage] = Field(default_factory=list)
	max_results: int = Field(3, ge=1, le=10)


class ChatResponse(BaseModel):
	answer: str
	used_web_results: bool
	web_results: List[SearchResult]


def _get_env(name: str) -> Optional[str]:
	value = os.getenv(name)
	if not value:
		logger.warning("Environment variable %s is not set.", name)
	return value


def get_openai_client() -> OpenAI:
	if OpenAI is None:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="openai package is not installed. Install it to enable AI responses.",
		)

	api_key = os.getenv(OPENAI_API_KEY_ENV)
	if not api_key:
		api_key = ""
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="OpenAI API key is not configured.",
		)

	return OpenAI(api_key=api_key)


async def call_openai_with_messages(messages: list[dict[str, str]]) -> str:
	"""Execute an OpenAI chat completion call in a background thread."""

	client = get_openai_client()

	def _invoke() -> str:
		response = client.chat.completions.create(
			model=OPENAI_MODEL,
			temperature=OPENAI_TEMPERATURE,
			messages=messages,
		)
		return response.choices[0].message.content.strip()

	try:
		return await asyncio.to_thread(_invoke)
	except Exception as exc:  # pragma: no cover - network related
		logger.exception("OpenAI call failed: %s", exc)
		raise HTTPException(
			status_code=status.HTTP_502_BAD_GATEWAY,
			detail="OpenAI service call failed.",
		) from exc


async def fetch_exa_results(query: str, max_results: int) -> list[SearchResult]:
	"""Fetch web search results from Exa. Returns an empty list if unavailable."""

	# if not api_key:
	api_key = "795e1fc5-5044-4795-b13d-c1e061991924"


	payload = {
		"query": query,
		"type": "neural",
		"numResults": max_results,
		"useAutoprompt": True,
	}

	headers = {
		"Content-Type": "application/json",
		"x-api-key": api_key,
	}

	try:
		async with httpx.AsyncClient(timeout=15.0) as client:
			response = await client.post(
				f"{EXA_API_BASE_URL.rstrip('/')}/search",
				headers=headers,
				json=payload,
			)
			print(response.status_code)
			print(response.content)
			# response.raise_for_status()
	except httpx.HTTPError as exc:  # pragma: no cover - external dependency
		logger.warning("Exa search failed for '%s': %s", query, exc)
		return []

	data = response.json()
	results = []

	for item in data.get("results", []):
		title = item.get("title") or item.get("id") or "Untitled"
		url = item.get("url") or item.get("link") or ""
		snippet = item.get("text") or item.get("snippet") or item.get("summary")

		if not url:
			continue

		results.append(SearchResult(title=title.strip(), url=url.strip(), snippet=snippet))

	return results


def build_search_context(results: list[SearchResult]) -> str:
	lines = []
	for idx, result in enumerate(results, start=1):
		snippet = result.snippet.strip() if result.snippet else ""
		lines.append(
			f"[{idx}] Title: {result.title}\nURL: {result.url}\nSummary: {snippet}"
		)
	return "\n\n".join(lines)


async def generate_search_summary(query: str, results: list[SearchResult]) -> str:
	if not results:
		messages = [
			{
				"role": "system",
				"content": (
					"You are an AI assistant helping users understand Singapore policy topics. "
					"Provide a concise, factual answer based on your general knowledge when no "
					"web results are available."
				),
			},
			{
				"role": "user",
				"content": f"Summarize the topic: {query}",
			},
		]
		return await call_openai_with_messages(messages)

	context = build_search_context(results)
	messages = [
		{
			"role": "system",
			"content": (
				"You synthesize Singapore policy information for a professional civic audience. "
				"Use only the provided sources and cite them inline using [n] style references."
			),
		},
		{
			"role": "user",
			"content": (
				f"Query: {query}\n\nSources:\n{context}\n\n"
				"Produce a concise summary (max 4 bullets) and mention the most relevant [n] citations."
			),
		},
	]
	return await call_openai_with_messages(messages)


async def generate_chat_answer(
	history: List[ChatMessage],
	message: str,
	web_results: list[SearchResult],
) -> str:
	system_prompt = (
		"You are PolicyNav, a Singapore governance assistant. Provide well-sourced, neutral, "
		"and actionable answers. If you reference web sources, cite them with [n] markers. If "
		"you lack sufficient information, state that transparently."
	)

	messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

	for msg in history:
		messages.append(msg.model_dump())

	if web_results:
		messages.append(
			{
				"role": "system",
				"content": (
					"Use the following web findings to ground your answer. Each entry is referenced "
					"with its [n] identifier.\n" + build_search_context(web_results)
				),
			}
		)

	messages.append({"role": "user", "content": message})

	return await call_openai_with_messages(messages)


@app.get("/health")
async def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest) -> SearchResponse:
	web_results = await fetch_exa_results(request.query, request.max_results)
	summary = await generate_search_summary(request.query, web_results)

	return SearchResponse(
		summary=summary,
		used_web_results=bool(web_results),
		results=web_results,
	)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
	web_results = await fetch_exa_results(request.message, request.max_results)
	answer = await generate_chat_answer(request.history, request.message, web_results)

	return ChatResponse(
		answer=answer,
		used_web_results=bool(web_results),
		web_results=web_results,
	)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
	import uvicorn
	import sys
	import os

	# Add the parent directory to sys.path so 'backend' can be imported
	sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

	uvicorn.run(
		"backend.main:app",
		host="0.0.0.0",
		port=int(os.getenv("PORT", "8000")),
		reload=True,
	)