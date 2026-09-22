"""Prepared transport and safety checks for the workshop.

The notebook contains the learner-facing tools, recipes and agent loop.
This module makes no network calls when imported. Only GeminiClient.ask()
uses the network, after the learner explicitly connects and runs a live cell.
Source contract checked 2026-09-15. Live-account rehearsal is still required.
"""
from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import dataclass
from getpass import GetPassWarning, getpass
from importlib.metadata import PackageNotFoundError, version

MODEL_DEFAULT = "gemini-3.5-flash-lite"
SDK_VERSION = "2.23.0"
MAX_REQUESTS = 5
MAX_TOOLS = 6


def package_check():
    """Local checks only. Does not install anything or contact Google."""
    import sys
    result = {"python": sys.version.split()[0], "kernel": sys.executable}
    for name in ("google-genai", "ipykernel"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "MISSING"
    result["ready"] = sys.version_info >= (3, 10) and (
        result["google-genai"] == SDK_VERSION and result["ipykernel"] != "MISSING"
    )
    return result


def safe_error(error):
    """Do not print raw HTTP errors: they may contain request data."""
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if not isinstance(code, int):
        code = getattr(getattr(error, "response", None), "status_code", None)
    messages = {
        400: "Request rejected. Check the current model, SDK and tool schema.",
        401: "Authentication failed. Check your own key privately.",
        403: "Access denied. Check account eligibility and project permissions.",
        404: "Model or endpoint unavailable. Ask the facilitator for the tested model.",
        429: "Quota/rate limit reached. Check AI Studio; pause or use simulation.",
        500: "Google service error. Pause; do not repeatedly rerun.",
        503: "Google service unavailable. Pause or use simulation.",
    }
    return messages.get(code, "Connection or SDK error. Check setup/network with a helper.")


class GeminiClient:
    """A small adapter. No generated code execution or automatic tool calling."""

    mode = "LIVE GEMINI"

    def __init__(self, sdk_client):
        self.sdk_client = sdk_client

    def ask(self, *, model, history, instructions, tools):
        return self.sdk_client.interactions.create(
            model=model,
            input=history,
            system_instruction=instructions,
            tools=tools,
            store=False,
            generation_config={"max_output_tokens": 1800},
        )

    def close(self):
        self.sdk_client.close()


def connect():
    """Prompt privately. Fail closed if this frontend cannot hide input."""
    from google import genai
    from google.genai import types

    if version("google-genai") != SDK_VERSION:
        raise RuntimeError("Use requirements.txt and select the workshop .venv kernel.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", GetPassWarning)
        key = getpass("Paste your own Gemini API key (hidden), then press Enter: ").strip()
    if not key:
        raise ValueError("No key entered. Nothing was sent.")
    try:
        sdk = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=0),
            ),
        )
    finally:
        del key
    return GeminiClient(sdk)


def describe_tools(registry):
    """Build JSON schemas from the notebook's small, explicit allowlist."""
    return [
        {
            "type": "function",
            "name": name,
            "description": rule["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "enum": list(choices)}
                    for key, choices in rule["arguments"].items()
                },
                "required": list(rule["arguments"]),
                "additionalProperties": False,
            },
        }
        for name, rule in registry.items()
    ]


def execute_safe(name, arguments, registry):
    """Enforce local permissions independently of the model's schema."""
    if not isinstance(name, str) or name not in registry:
        return {"status": "rejected", "reason": "Tool is not allowed."}
    rule = registry[name]
    expected = rule["arguments"]
    if not isinstance(arguments, dict) or set(arguments) != set(expected):
        return {"status": "rejected", "reason": "Unexpected or missing arguments."}
    for key, allowed in expected.items():
        if not isinstance(arguments[key], str) or arguments[key] not in allowed:
            return {"status": "rejected", "reason": "Unsupported argument value."}
    try:
        result = rule["function"](**arguments)
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(encoded) > 8000:
            return {"status": "rejected", "reason": "Tool result is too large."}
        return result
    except Exception:
        return {"status": "tool_error", "reason": "Local tool failed. Ask a helper."}


class AgentSession:
    """Conversation state, observable trace and fixed per-run budgets."""

    def __init__(self, request, max_requests=MAX_REQUESTS, max_tools=MAX_TOOLS):
        if not isinstance(request, str) or not request.strip() or len(request) > 3000:
            raise ValueError("Use a non-empty fictional request of at most 3000 characters.")
        if type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS:
            raise ValueError("Request budget must be 1 to 5.")
        if type(max_tools) is not int or not 0 <= max_tools <= MAX_TOOLS:
            raise ValueError("Tool budget must be 0 to 6.")
        self.max_requests, self.max_tools = max_requests, max_tools
        self.history = [{"type": "user_input", "content": [{"type": "text", "text": request}]}]
        self.requests, self.tool_count = 0, 0
        self.trace, self.call_ids = [], set()
        self.status, self.draft, self.mode = "running", "", "NOT RUN"
        self.started = time.monotonic()

    def stop(self, status):
        self.status = status

    def ask(self, client, model, instructions, tools):
        if client is None:
            self.stop("not_connected")
            return None
        if self.requests >= self.max_requests:
            self.stop("request_limit")
            return None
        if time.monotonic() - self.started > 150:
            self.stop("time_limit")
            return None
        if len(json.dumps(self.history, ensure_ascii=False)) > 80_000:
            self.stop("history_limit")
            return None
        self.mode = client.mode
        self.requests += 1
        try:
            return client.ask(model=model, history=self.history,
                              instructions=instructions, tools=tools)
        except Exception as error:
            self.stop("api_error")
            self.trace.append({"event": "error", "message": safe_error(error)})
            return None

    def record(self, reply):
        # Preserve ALL returned steps, including opaque signature/thought fields.
        # Do not expose those fields in the learner-facing trace.
        steps = getattr(reply, "steps", None) or []
        self.history.extend(step.model_dump() for step in steps)
        return [step for step in steps if step.type == "function_call"]

    def can_execute(self, calls):
        if self.requests >= self.max_requests:
            self.stop("request_limit")
            return False  # No orphan tool work without a following model turn.
        if self.tool_count + len(calls) > self.max_tools:
            self.stop("tool_limit")
            return False
        ids = [getattr(call, "id", None) for call in calls]
        if (any(not isinstance(i, str) or not i for i in ids)
                or len(ids) != len(set(ids)) or self.call_ids.intersection(ids)):
            self.stop("invalid_call_ids")
            return False
        self.call_ids.update(ids)
        return True

    def add_tool_result(self, call, result):
        self.tool_count += 1
        self.trace.append({
            "event": "tool_result", "tool": call.name,
            "arguments": copy.deepcopy(call.arguments), "result": copy.deepcopy(result),
        })
        self.history.append({
            "type": "function_result", "name": call.name, "call_id": call.id,
            "result": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
        })

    def finish(self, text):
        self.draft = text.strip() if isinstance(text, str) else ""
        self.stop("draft_needs_review" if self.draft else "no_final_text")

    def result(self):
        if self.status == "running":
            self.stop("request_limit")
        return {
            "mode": self.mode, "status": self.status, "model_requests": self.requests,
            "tool_calls": self.tool_count, "draft": self.draft,
            "trace": copy.deepcopy(self.trace),
            "warning": "A draft or evidence ID does not prove truth. Check each claim.",
        }


def run_demo(client, model):
    """One deliberately small request for readiness and the first API exercise."""
    session = AgentSession("Give one short tip for practising an interview.", max_requests=1)
    reply = session.ask(client, model, "Use plain English. Maximum 35 words.", [])
    if reply is not None:
        calls = session.record(reply)
        if calls:
            session.stop("unexpected_tool_request")
        else:
            session.finish(getattr(reply, "output_text", ""))
    return session.result()


def show_result(result):
    print("MODE:", result["mode"], "| STATUS:", result["status"])
    label = "Simulated model turns:" if result["mode"].startswith("SIMULATION") else "Model requests:"
    print(label, result["model_requests"], "| Tool attempts:", result["tool_calls"])
    for entry in result["trace"]:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    if result["draft"]:
        print("\nDRAFT TO REVIEW\n" + result["draft"])
    print("\n" + result["warning"])


@dataclass
class FakeStep:
    type: str
    name: str = ""
    arguments: dict | None = None
    id: str = ""

    def model_dump(self):
        return {"type": self.type, "name": self.name, "arguments": self.arguments, "id": self.id}


@dataclass
class FakeReply:
    steps: list
    output_text: str = ""


class SimulationClient:
    """Authored test fixture. No model, no network, no claim of live intelligence."""

    mode = "SIMULATION: authored fixture, no model call"

    def __init__(self, role="project_coordinator", skill="communication", tone="friendly"):
        self.role, self.skill, self.tone, self.turn = role, skill, tone, 0

    def ask(self, **kwargs):
        self.turn += 1
        if self.turn == 1:
            return FakeReply([
                FakeStep("function_call", "get_role", {"role_id": self.role}, "sim-1"),
                FakeStep("function_call", "get_evidence", {"skill": self.skill}, "sim-2"),
                FakeStep("function_call", "get_question",
                         {"skill": self.skill, "tone": self.tone}, "sim-3"),
            ])
        return FakeReply([], (
            "SIMULATION ONLY. Read the tool results above. Select a returned experience "
            "and write your own STAR outline. If evidence is missing, state the gap. "
            "Use the returned practice question. This text is an authored instruction, "
            "not a generated answer and not responsive to your skill recipe."
        ))
