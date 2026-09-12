"""Structured local-data, WebCmd, and optional AI integrations."""

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urlerror
from urllib import request

import pandas as pd


class WebCmdError(RuntimeError):
    """Raised when WebCmd cannot complete a request."""


class AIClientError(RuntimeError):
    """Raised when the configured AI provider cannot complete a request."""


def csv_context(csv_path: str, sample_rows: int = 5, chunk_size: int = 10000) -> Dict[str, Any]:
    """Return compact, JSON-serializable context without loading the whole CSV."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if sample_rows < 0 or chunk_size < 1:
        raise ValueError("sample_rows must be non-negative and chunk_size must be positive")

    try:
        chunks = pd.read_csv(csv_path, chunksize=chunk_size)
        columns: Optional[List[str]] = None
        sample: List[Dict[str, Any]] = []
        row_count = 0
        numeric_stats: Dict[str, Dict[str, float]] = {}

        for chunk in chunks:
            if columns is None:
                columns = [str(column) for column in chunk.columns]
            row_count += len(chunk)
            if len(sample) < sample_rows:
                sample.extend(chunk.head(sample_rows - len(sample)).to_dict(orient="records"))

            for column in chunk.select_dtypes(include="number").columns:
                values = chunk[column].dropna()
                if values.empty:
                    continue
                name = str(column)
                current = numeric_stats.setdefault(
                    name,
                    {"min": float(values.min()), "max": float(values.max()), "sum": 0.0, "count": 0.0},
                )
                current["min"] = min(current["min"], float(values.min()))
                current["max"] = max(current["max"], float(values.max()))
                current["sum"] += float(values.sum())
                current["count"] += float(values.count())

        if columns is None:
            raise ValueError("CSV is empty or has no header row")

        summary = {
            name: {
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["sum"] / stats["count"],
            }
            for name, stats in numeric_stats.items()
            if stats["count"]
        }
        return {
            "file": os.path.abspath(csv_path),
            "row_count": row_count,
            "columns": columns,
            "numeric_summary": summary,
            "sample": sample,
        }
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse CSV '{csv_path}': {exc}") from exc


class WebCmdClient:
    """Run the verified WebCmd OmniSearch research command."""

    def __init__(self, executable: Optional[str] = None, timeout: Optional[float] = None):
        configured = executable or os.environ.get("WEBCMD_EXECUTABLE")
        self.executable = configured or shutil.which("webcmd") or "webcmd"
        self.timeout = timeout or float(os.environ.get("WEBCMD_TIMEOUT", "60"))

    def search(self, query: str) -> Any:
        if not query.strip():
            raise ValueError("query must not be empty")
        command = [self.executable, "omnisearch", "research", query, "-f", "json"]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise WebCmdError(
                f"WebCmd executable not found: {self.executable}. Install WebCmd or set WEBCMD_EXECUTABLE."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise WebCmdError(f"WebCmd timed out after {self.timeout:g} seconds.") from exc

        if result.returncode != 0:
            message = result.stderr.strip() or "WebCmd returned a non-zero exit code."
            raise WebCmdError(message)
        output = (result.stdout or "").strip()
        if not output:
            raise WebCmdError("WebCmd returned no output.")
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"format": "plain", "content": output}

    def fetch_url(self, url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Extract readable content from a public URL through WebCmd."""
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Unsupported URL: {url}")
        command = [
            self.executable,
            "web",
            "fetch",
            "--url",
            url,
            "--max-chars",
            str(max_chars),
            "-f",
            "json",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise WebCmdError(f"WebCmd executable not found: {self.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise WebCmdError(f"WebCmd timed out after {self.timeout:g} seconds.") from exc
        if result.returncode != 0:
            message = result.stderr.strip() or "WebCmd URL fetch failed."
            raise WebCmdError(message)
        try:
            data = json.loads(result.stdout or "")
        except json.JSONDecodeError as exc:
            raise WebCmdError("WebCmd URL fetch returned invalid JSON.") from exc
        return {
            "url": url,
            "title": data.get("title"),
            "status": data.get("status"),
            "content": str(data.get("content", ""))[:max_chars],
        }


class AIClient:
    """Call an OpenAI-compatible chat endpoint using environment variables.

    Required variables: ULTRAQUERY_AI_ENDPOINT, ULTRAQUERY_AI_API_KEY,
    and ULTRAQUERY_AI_MODEL. No provider or secret is hard-coded.
    """

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        self.endpoint = endpoint or os.environ.get("ULTRAQUERY_AI_ENDPOINT")
        self.api_key = api_key or os.environ.get("ULTRAQUERY_AI_API_KEY")
        self.model = model or os.environ.get("ULTRAQUERY_AI_MODEL")
        if self.model and self.model.startswith("models/"):
            self.model = self.model.removeprefix("models/")
        if self.endpoint and self.endpoint.rstrip("/").endswith("/v1beta/openai"):
            self.endpoint = self.endpoint.rstrip("/") + "/chat/completions"

    def generate_answer(self, question: str, context: Dict[str, Any]) -> str:
        missing = [name for name, value in {
            "ULTRAQUERY_AI_ENDPOINT": self.endpoint,
            "ULTRAQUERY_AI_API_KEY": self.api_key,
            "ULTRAQUERY_AI_MODEL": self.model,
        }.items() if not value]
        if missing:
            raise AIClientError("Configure these environment variables first: " + ", ".join(missing))

        body = json.dumps({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": (
                    "Answer the user's question using only the supplied context. "
                    "Mention when evidence is missing.\n\n"
                    + json.dumps({"question": question, "context": context}, default=str)
                ),
            }],
        }).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "UltraQuery/0.0.3",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except OSError:
                detail = "No response details available."
            raise AIClientError(f"AI request failed with HTTP {exc.code}: {detail}") from exc
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AIClientError(f"AI request failed: {exc}") from exc

        try:
            return str(result["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise AIClientError("AI response did not contain choices[0].message.content") from exc


def tool_payload(csv_path: str, question: str, webcmd: Optional[WebCmdClient] = None) -> Dict[str, Any]:
    """Combine local CSV context and optional WebCmd research."""
    payload: Dict[str, Any] = {"question": question, "local_data": csv_context(csv_path)}
    if webcmd is not None:
        payload["web_data"] = webcmd.search(question)
    return payload


def _url_columns(frame: pd.DataFrame) -> List[str]:
    columns = []
    for column in frame.columns:
        values = frame[column].dropna().astype(str).str.strip()
        if not values.empty and values.str.startswith(("http://", "https://")).mean() >= 0.5:
            columns.append(str(column))
    return columns


def scrape_csv_urls(csv_path: str, webcmd: WebCmdClient, max_urls: int = 5) -> List[Dict[str, Any]]:
    """Find URL columns in a CSV and fetch a bounded number of pages."""
    if max_urls < 1:
        return []
    frame = pd.read_csv(csv_path, nrows=max(100, max_urls * 10))
    results: List[Dict[str, Any]] = []
    for column in _url_columns(frame):
        for value in frame[column].dropna().astype(str).str.strip():
            if not value.startswith(("http://", "https://")):
                continue
            try:
                results.append(webcmd.fetch_url(value))
            except (WebCmdError, ValueError) as error:
                results.append({"url": value, "error": str(error)})
            if len(results) >= max_urls:
                return results
    return results


def multi_csv_context(csv_paths: Sequence[str]) -> Dict[str, Any]:
    """Create comparable context for multiple CSV datasets."""
    if not csv_paths:
        raise ValueError("At least one CSV file is required")
    datasets = [csv_context(path) for path in csv_paths]
    ignored_fragments = ('lat', 'lon', 'time', 'date', 'scan', 'track', 'version', 'index', 'id')
    common_columns = sorted(
        column for column in set.intersection(*(set(item["numeric_summary"]) for item in datasets))
        if not any(fragment in column.lower() for fragment in ignored_fragments)
    )
    comparison = {
        column: {
            "means": [item["numeric_summary"][column]["mean"] for item in datasets],
            "min": min(item["numeric_summary"][column]["min"] for item in datasets),
            "max": max(item["numeric_summary"][column]["max"] for item in datasets),
        }
        for column in common_columns
    }
    return {
        "datasets": datasets,
        "dataset_count": len(datasets),
        "comparison": comparison,
    }


def advanced_payload(
    csv_paths: Sequence[str],
    question: str,
    webcmd: Optional[WebCmdClient] = None,
    scrape_urls: bool = False,
    max_urls: int = 5,
) -> Dict[str, Any]:
    """Combine multi-file analytics, URL pages, and optional web research."""
    payload: Dict[str, Any] = {
        "question": question,
        "local_data": multi_csv_context(csv_paths),
    }
    if webcmd is not None and scrape_urls:
        payload["scraped_pages"] = [
            {"file": path, "pages": scrape_csv_urls(path, webcmd, max_urls)}
            for path in csv_paths
        ]
    if webcmd is not None:
        payload["web_research"] = webcmd.search(question)
    return payload
