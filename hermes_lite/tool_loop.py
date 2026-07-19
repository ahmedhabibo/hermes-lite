"""hermes_lite.tool_loop — Two-tier tool-calling loop with termination guards.

Extracted from the orchestrator so the agent loop stays focused on session
wiring and prompt handling.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from hermes_lite.config import get_config
from hermes_lite.llm import ChatRequest, ChatResponse, chat, tool_def
from hermes_lite.registry import (
    PluginRegistry,
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
)
from hermes_lite.sanitize import sanitize_tool_args

logger = logging.getLogger(__name__)


@dataclass
class ToolLoopResult:
    """Outcome of a ToolLoop run — fed back to the orchestrator."""

    response: str
    """Final text to show the user."""

    iterations: int
    """How many LLM round-trips the loop took."""

    tool_calls_made: int
    """Total tool invocations across all iterations."""

    tool_names: list[str]
    """Names of every tool called, in order."""

    terminated_by: str
    """Why the loop stopped: 'complete', 'max_iterations', 'repeated_error', 'malformed_tool_call'."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra structured data for persistence (routing, model, etc.)."""


class ToolLoop:
    """Two-tier tool-calling loop with termination guards.

    Termination conditions:
    - **complete**: LLM returned a response with no ``tool_calls``
    - **max_iterations**: exceeded ``max_iterations``
    - **repeated_error**: the same tool produced the same error twice
    - **malformed_tool_call**: invalid JSON args after one retry nudge
    """

    def __init__(
        self,
        registry: PluginRegistry,
        chat_fn: Callable[[ChatRequest], Any] = chat,
        max_iterations: int | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
    ) -> None:
        self.registry = registry
        self.chat_fn = chat_fn
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else get_config().max_iterations
        )
        self.on_tool_call = on_tool_call

    async def run(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ToolLoopResult:
        history = list(messages)
        tools = self._build_openai_tools()
        iterations = 0
        total_tool_calls = 0
        all_tool_names: list[str] = []
        last_error_key: tuple[str, str] | None = None
        nudged_json = False
        turn_start_ms = int(time.monotonic() * 1000)
        cumulative_prompt_tokens = 0
        cumulative_completion_tokens = 0
        turn_errors: list[str] = []
        last_tier = "local"

        while iterations < self.max_iterations:
            iterations += 1
            req = ChatRequest(
                messages=history,
                model=model,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp: ChatResponse = await self.chat_fn(req)
            last_tier = resp.tier

            cumulative_prompt_tokens += resp.usage.get("prompt_tokens", 0)
            cumulative_completion_tokens += resp.usage.get("completion_tokens", 0)

            if not resp.tool_calls:
                return self._result(
                    response=resp.content or "",
                    iterations=iterations,
                    total_tool_calls=total_tool_calls,
                    all_tool_names=all_tool_names,
                    terminated_by="complete",
                    model=model,
                    tier=resp.tier,
                    turn_start_ms=turn_start_ms,
                    prompt_tokens=cumulative_prompt_tokens,
                    completion_tokens=cumulative_completion_tokens,
                    errors=turn_errors,
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.content or None,
            }
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in resp.tool_calls
            ]
            history.append(assistant_msg)

            any_malformed = False

            for tc in resp.tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
                tc_args_str = tc.get("arguments", "{}")
                total_tool_calls += 1
                all_tool_names.append(tc_name)
                first_arg = self._extract_first_arg(tc_args_str)

                result_content: str
                try:
                    args = json.loads(tc_args_str)
                except json.JSONDecodeError:
                    any_malformed = True
                    if not nudged_json:
                        nudged_json = True
                        result_content = (
                            "Error: your tool call arguments were not valid JSON. "
                            "Please respond with strict JSON arguments only — "
                            "no comments, no trailing commas."
                        )
                    else:
                        result_content = (
                            "Error: repeated malformed JSON in tool call arguments. "
                            "Stopping."
                        )
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result_content,
                        })
                        turn_errors.append("malformed JSON in tool call")
                        return self._result(
                            response=(
                                "I encountered repeated errors formatting tool calls. "
                                "Please try rephrasing your request."
                            ),
                            iterations=iterations,
                            total_tool_calls=total_tool_calls,
                            all_tool_names=all_tool_names,
                            terminated_by="malformed_tool_call",
                            model=model,
                            tier=resp.tier,
                            turn_start_ms=turn_start_ms,
                            prompt_tokens=cumulative_prompt_tokens,
                            completion_tokens=cumulative_completion_tokens,
                            errors=turn_errors,
                        )
                else:
                    if self.on_tool_call:
                        self.on_tool_call(tc_name, first_arg)

                    try:
                        sanitized = sanitize_tool_args(tool_name=tc_name, args=args)
                        if not sanitized.is_clean:
                            issues = "; ".join(sanitized.issues)
                            logger.warning("[SANITIZE] '%s' blocked: %s", tc_name, issues)
                            result_content = (
                                f"Tool call blocked: security policy violation — {issues}"
                            )
                            turn_errors.append(f"{tc_name}: {issues}")
                        else:
                            raw = self.registry.call_tool(
                                tc_name,
                                sanitized.args,
                                auth_token=self.registry.auth_token,
                            )
                            if isinstance(raw, dict) and raw.get("ok"):
                                result_content = raw.get("output", "")
                            elif isinstance(raw, dict) and not raw.get("ok"):
                                err = raw.get("error", "unknown error")
                                result_content = f"Tool error: {err}"
                            else:
                                result_content = str(raw)
                    except ToolNotFoundError:
                        result_content = f"Tool error: '{tc_name}' is not a registered tool."
                        turn_errors.append(f"{tc_name}: not registered")
                    except ToolValidationError as exc:
                        result_content = f"Tool error: validation failed — {exc}"
                        turn_errors.append(f"{tc_name}: validation failed")
                    except ToolError as exc:
                        result_content = f"Tool error: {exc}"
                        turn_errors.append(f"{tc_name}: {exc}")
                    except Exception as exc:
                        result_content = (
                            f"Tool error: unexpected {type(exc).__name__}: {exc}"
                        )
                        turn_errors.append(f"{tc_name}: {type(exc).__name__}")

                    if result_content.startswith("Tool error:"):
                        error_key = (tc_name, result_content[:80])
                        if error_key == last_error_key:
                            history.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": result_content,
                            })
                            turn_errors.append(f"{tc_name}: repeated error")
                            return self._result(
                                response=(
                                    f"I hit a repeated error with the `{tc_name}` tool "
                                    f"and stopped. Last error: {result_content}"
                                ),
                                iterations=iterations,
                                total_tool_calls=total_tool_calls,
                                all_tool_names=all_tool_names,
                                terminated_by="repeated_error",
                                model=model,
                                tier=resp.tier,
                                turn_start_ms=turn_start_ms,
                                prompt_tokens=cumulative_prompt_tokens,
                                completion_tokens=cumulative_completion_tokens,
                                errors=turn_errors,
                                extra={"repeated_error_tool": tc_name},
                            )
                        last_error_key = error_key
                    else:
                        last_error_key = None

                # Always append a tool result so the assistant tool_calls stay valid.
                history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_content,
                })

            if any_malformed:
                continue

        return self._result(
            response=(
                f"I reached the maximum number of reasoning steps ({self.max_iterations}) "
                f"and stopped. Here is what I have so far:\n\n"
                + "[no partial response]"
            ),
            iterations=iterations,
            total_tool_calls=total_tool_calls,
            all_tool_names=all_tool_names,
            terminated_by="max_iterations",
            model=model,
            tier=last_tier,
            turn_start_ms=turn_start_ms,
            prompt_tokens=cumulative_prompt_tokens,
            completion_tokens=cumulative_completion_tokens,
            errors=turn_errors,
        )

    def _result(
        self,
        *,
        response: str,
        iterations: int,
        total_tool_calls: int,
        all_tool_names: list[str],
        terminated_by: str,
        model: str,
        tier: str,
        turn_start_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        errors: list[str],
        extra: dict[str, Any] | None = None,
    ) -> ToolLoopResult:
        elapsed_ms = int((time.monotonic() * 1000) - turn_start_ms)
        metadata: dict[str, Any] = {
            "model": model,
            "tier": tier,
            "_obs": {
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "errors": errors,
            },
        }
        if extra:
            metadata.update(extra)
        return ToolLoopResult(
            response=response,
            iterations=iterations,
            tool_calls_made=total_tool_calls,
            tool_names=all_tool_names,
            terminated_by=terminated_by,
            metadata=metadata,
        )

    def _build_openai_tools(self) -> list[dict[str, Any]]:
        result = []
        for desc in self.registry.tool_descriptions():
            result.append(tool_def(
                name=desc["name"],
                description=desc["description"],
                parameters=desc.get("parameters", {"type": "object", "properties": {}}),
            ))
        return result

    @staticmethod
    def _extract_first_arg(args_str: str) -> str:
        try:
            obj = json.loads(args_str)
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str):
                        return v
                    return str(v)
        except (json.JSONDecodeError, ValueError):
            pass
        return args_str[:60]


__all__ = ["ToolLoop", "ToolLoopResult"]
