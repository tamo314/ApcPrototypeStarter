#!/usr/bin/env python3
"""Minimal Codex -> (Claude Code | Antigravity | Codex) research orchestration loop.

The planner is Codex CLI. The implementation executor is selected in config.json.
No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = APP_DIR / "config.json"
PLANNER_SCHEMA = APP_DIR / "planner_schema.json"
PLANNER_PROMPT_FILE = APP_DIR / "prompts" / "planner.md"
EXECUTOR_PROMPT_FILE = APP_DIR / "prompts" / "executor.md"
CONTEXT_FILE = APP_DIR / "AGENTS.md"
INITIAL_TASK_FILE = APP_DIR / "research" / "INITIAL_TASK.md"
RUNTIME_DIR = APP_DIR / ".orchestrator"
RUNS_DIR = RUNTIME_DIR / "runs"
CODEX_SCRATCH_DIR = RUNTIME_DIR / "codex_scratch"
STATE_FILE = RUNTIME_DIR / "state.json"
HISTORY_FILE = RUNTIME_DIR / "history.jsonl"


class OrchestratorError(RuntimeError):
    pass


@dataclass
class CommandResult:
    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


@dataclass
class State:
    iteration: int
    phase: str
    current_task: str
    executor: str
    last_executor_report_path: str | None = None
    done_reason: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OrchestratorError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise OrchestratorError(f"Required file not found: {path}") from exc


def resolve_project_dir(config: dict[str, Any], config_path: Path) -> Path:
    raw = config.get("project_dir", "..")
    project_dir = Path(raw)
    if not project_dir.is_absolute():
        project_dir = (config_path.parent / project_dir).resolve()
    else:
        project_dir = project_dir.resolve()
    if not project_dir.is_dir():
        raise OrchestratorError(f"project_dir does not exist or is not a directory: {project_dir}")
    return project_dir


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    required_top = ["executor", "max_iterations", "codex", "claude", "antigravity"]
    missing = [key for key in required_top if key not in config]
    if missing:
        raise OrchestratorError(f"Missing config keys: {', '.join(missing)}")

    executor = config["executor"]
    if executor not in {"claude", "antigravity", "codex"}:
        raise OrchestratorError('config.executor must be "claude", "antigravity", or "codex"')

    if int(config["max_iterations"]) < 1:
        raise OrchestratorError("max_iterations must be >= 1")

    config.setdefault("executor_timeout_seconds", 7200)
    config.setdefault("planner_timeout_seconds", 600)
    config.setdefault("max_result_chars_for_planner", 60000)
    config.setdefault("recent_history_items_for_planner", 4)
    config.setdefault("stop_on_executor_error", True)

    for tool in ("codex", "claude", "antigravity"):
        if not isinstance(config[tool], dict) or not config[tool].get("command"):
            raise OrchestratorError(f"config.{tool}.command is required")
        config[tool].setdefault("extra_args", [])
        config[tool].setdefault("model", "")
        if not isinstance(config[tool]["extra_args"], list):
            raise OrchestratorError(f"config.{tool}.extra_args must be an array")
        if config[tool]["model"] is None:
            config[tool]["model"] = ""
        if not isinstance(config[tool]["model"], str):
            raise OrchestratorError(f"config.{tool}.model must be a string")

    config["antigravity"].setdefault("dangerously_skip_permissions", False)
    if not isinstance(config["antigravity"]["dangerously_skip_permissions"], bool):
        raise OrchestratorError("config.antigravity.dangerously_skip_permissions must be true or false")

    # Codex is always the planner, and can optionally also be the implementation
    # executor. Keep executor-specific policy separate from the planner's
    # read-only invocation.
    config["codex"].setdefault("executor_model", "")
    config["codex"].setdefault("executor_sandbox", "workspace-write")
    config["codex"].setdefault("executor_approval_policy", "never")
    config["codex"].setdefault("executor_extra_args", [])
    if not isinstance(config["codex"]["executor_model"], str):
        raise OrchestratorError("config.codex.executor_model must be a string")
    if config["codex"]["executor_sandbox"] not in {"read-only", "workspace-write", "danger-full-access"}:
        raise OrchestratorError(
            'config.codex.executor_sandbox must be "read-only", "workspace-write", or "danger-full-access"'
        )
    if config["codex"]["executor_approval_policy"] not in {"untrusted", "on-request", "never"}:
        raise OrchestratorError(
            'config.codex.executor_approval_policy must be "untrusted", "on-request", or "never"'
        )
    if not isinstance(config["codex"]["executor_extra_args"], list):
        raise OrchestratorError("config.codex.executor_extra_args must be an array")

    return config


def resolve_cli_command(command: str) -> str:
    """Resolve a CLI name to the actual executable/shim path.

    This is especially important on Windows: PowerShell and shutil.which()
    can resolve commands such as ``codex`` to ``codex.CMD``, while
    CreateProcess (used by subprocess) may fail when given only the bare
    command name. Passing the fully resolved shim path avoids WinError 2.
    """
    command_path = Path(command)
    if command_path.is_absolute() or command_path.parent != Path("."):
        return str(command_path)
    resolved = shutil.which(command)
    return resolved or command


def resolved_command(command: list[str]) -> list[str]:
    if not command:
        raise OrchestratorError("Cannot run an empty command")
    return [resolve_cli_command(command[0]), *command[1:]]


def _decode_subprocess_bytes(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        if os.name == "nt":
            try:
                return value.decode("cp932")
            except UnicodeDecodeError:
                pass
        return value.decode("utf-8", errors="replace")


def run_command(
    command: list[str],
    cwd: Path,
    timeout_seconds: int,
    stdin_text: str | None = None,
) -> CommandResult:
    command = resolved_command(command)
    start = time.monotonic()
    stdin_bytes = stdin_text.encode("utf-8") if stdin_text is not None else None
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            input=stdin_bytes,
            capture_output=True,
            text=False,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        return CommandResult(
            command=command,
            return_code=proc.returncode,
            stdout=_decode_subprocess_bytes(proc.stdout),
            stderr=_decode_subprocess_bytes(proc.stderr),
            duration_seconds=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_subprocess_bytes(exc.stdout)
        stderr = _decode_subprocess_bytes(exc.stderr)
        return CommandResult(
            command=command,
            return_code=124,
            stdout=stdout,
            stderr=stderr + f"\nTimed out after {timeout_seconds} seconds.",
            duration_seconds=time.monotonic() - start,
            timed_out=True,
        )


def clean_cli_name(command: str) -> str:
    return Path(command).name


def command_exists(command: str) -> bool:
    command_path = Path(command)
    if command_path.is_absolute() or command_path.parent != Path("."):
        return command_path.exists()
    return shutil.which(command) is not None


def version_probe(command: str) -> str:
    try:
        actual_command = resolve_cli_command(command)
        result = subprocess.run(
            [actual_command, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        text = (result.stdout or result.stderr).strip().splitlines()
        return text[0] if text else f"exit={result.returncode}"
    except Exception as exc:  # diagnostic only
        return f"version check failed: {exc}"


def _read_antigravity_settings_model() -> str | None:
    path = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model = data.get("model") if isinstance(data, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def _read_codex_config_model() -> str | None:
    path = Path.home() / ".codex" / "config.toml"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    # Only consider a top-level model assignment before the first TOML table.
    pattern = re.compile(r"model\s*=\s*[\"']([^\"']+)[\"']")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            break
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip()
    return None


def configured_model(
    config: dict[str, Any], tool: str, *, role: str = "default"
) -> tuple[str, str]:
    if tool == "codex" and role == "executor":
        executor_model = str(config["codex"].get("executor_model", "")).strip()
        if executor_model:
            return executor_model, "config.json codex.executor_model"

    explicit = str(config[tool].get("model", "")).strip()
    if explicit:
        return explicit, "config.json"
    if tool == "antigravity":
        detected = _read_antigravity_settings_model()
        if detected:
            return detected, "Antigravity settings.json"
    if tool == "codex":
        detected = _read_codex_config_model()
        if detected:
            return detected, "Codex config.toml"
    return "CLI default (not pinned)", "default"


def _model_args(config: dict[str, Any], tool: str, *, role: str = "default") -> list[str]:
    if tool == "codex" and role == "executor":
        model = str(config["codex"].get("executor_model", "")).strip()
        if not model:
            model = str(config["codex"].get("model", "")).strip()
    else:
        model = str(config[tool].get("model", "")).strip()
    return ["--model", model] if model else []


def _parse_codex_runtime_model(stdout: str, stderr: str) -> str | None:
    text = f"{stderr}\n{stdout}"
    match = re.search(r"(?mi)^\s*model\s*:\s*([^\r\n]+)", text)
    return match.group(1).strip() if match else None


def run_checks(config: dict[str, Any], project_dir: Path) -> bool:
    executor = config["executor"]
    required_commands = list(dict.fromkeys([config["codex"]["command"], config[executor]["command"]]))
    ok = True

    planner_model, planner_model_source = configured_model(config, "codex")
    executor_model, executor_model_source = configured_model(
        config, executor, role="executor" if executor == "codex" else "default"
    )

    print(f"Project directory : {project_dir}")
    print(f"Executor          : {executor}")
    print(f"Planner model     : {planner_model} [{planner_model_source}]")
    print(f"Executor model    : {executor_model} [{executor_model_source}]")
    print(f"Max iterations    : {config['max_iterations']}")
    print()

    for command in required_commands:
        exists = command_exists(command)
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {clean_cli_name(command)}", end="")
        if exists:
            print(f" — {version_probe(command)}")
        else:
            print()
            ok = False

    for path in [PLANNER_SCHEMA, PLANNER_PROMPT_FILE, EXECUTOR_PROMPT_FILE, CONTEXT_FILE, INITIAL_TASK_FILE]:
        exists = path.is_file()
        print(f"[{'OK' if exists else 'MISSING'}] {path.relative_to(APP_DIR)}")
        ok = ok and exists

    if executor == "antigravity":
        print()
        if config["antigravity"].get("dangerously_skip_permissions"):
            print("[WARNING] Antigravity auto-approval is ENABLED (--dangerously-skip-permissions).")
            print("          The agent can run commands and modify files without confirmation.")
        else:
            print("Antigravity note: agy -p is non-interactive. For host-side code changes, set")
            print("Tool Permission to 'always-proceed' in /permissions or /config, OR explicitly")
            print("set antigravity.dangerously_skip_permissions=true in config.json.")

    if executor == "codex":
        print()
        print(
            "Codex executor policy: "
            f"sandbox={config['codex']['executor_sandbox']}, "
            f"approval={config['codex']['executor_approval_policy']}"
        )
        if config["codex"]["executor_sandbox"] == "danger-full-access":
            print("[WARNING] Codex executor has danger-full-access to the host environment.")
        print("Note: planner and executor are separate Codex invocations; planner remains read-only.")

    return ok


def build_executor_prompt(task: str, context: str) -> str:
    base = read_text(EXECUTOR_PROMPT_FILE)
    return (
        f"{base}\n\n"
        "=== RESEARCH CONTEXT ===\n"
        f"{context}\n\n"
        "=== CURRENT TASK ===\n"
        f"{task}\n"
    )


def normalize_executor_output(executor: str, raw_stdout: str, raw_stderr: str) -> str:
    if executor == "claude":
        try:
            parsed = json.loads(raw_stdout)
            if isinstance(parsed, dict):
                for key in ("structured_output", "result", "content"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                    if value is not None and key == "structured_output":
                        return json.dumps(value, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

    output = raw_stdout.strip()
    if not output and raw_stderr.strip():
        output = raw_stderr.strip()
    return output


def run_executor(
    config: dict[str, Any],
    project_dir: Path,
    task: str,
    iteration: int,
) -> tuple[CommandResult, str, Path]:
    executor = config["executor"]
    tool_config = config[executor]
    context = read_text(CONTEXT_FILE)
    prompt = build_executor_prompt(task, context)

    codex_final_path: Path | None = None
    stdin_text: str | None = None

    if executor == "claude":
        model_args = _model_args(config, executor)
        command = [tool_config["command"], *model_args, *tool_config["extra_args"], "-p", prompt]
        model_name, model_source = configured_model(config, executor)
    elif executor == "antigravity":
        model_args = _model_args(config, executor)
        permission_args = (
            ["--dangerously-skip-permissions"]
            if tool_config.get("dangerously_skip_permissions")
            else []
        )
        command = [
            tool_config["command"],
            *permission_args,
            *model_args,
            *tool_config["extra_args"],
            "-p",
            prompt,
        ]
        model_name, model_source = configured_model(config, executor)
    else:  # codex executor
        CODEX_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        codex_final_path = CODEX_SCRATCH_DIR / f"executor_{iteration:04d}_final.txt"
        if codex_final_path.exists():
            codex_final_path.unlink()

        approval = str(tool_config["executor_approval_policy"])
        sandbox = str(tool_config["executor_sandbox"])
        # --ask-for-approval is a global Codex flag, so it must appear before
        # the `exec` subcommand. Prompt text goes through stdin to avoid the
        # Windows command-line length limit.
        command = [
            tool_config["command"],
            "--ask-for-approval",
            approval,
            "exec",
            *_model_args(config, "codex", role="executor"),
            "--sandbox",
            sandbox,
            "--skip-git-repo-check",
            "--output-last-message",
            str(codex_final_path),
            *tool_config["executor_extra_args"],
            "-",
        ]
        stdin_text = prompt
        model_name, model_source = configured_model(config, "codex", role="executor")

    print(f"\n[{iteration}] Executor: {executor}")
    print(f"[{iteration}] Executor model: {model_name} [{model_source}]")
    if executor == "codex":
        print(
            f"[{iteration}] Codex policy: sandbox={tool_config['executor_sandbox']}, "
            f"approval={tool_config['executor_approval_policy']}"
        )
    print(f"[{iteration}] Running implementation task...")
    result = run_command(
        command,
        project_dir,
        int(config["executor_timeout_seconds"]),
        stdin_text=stdin_text,
    )

    if executor == "codex" and codex_final_path is not None and codex_final_path.is_file():
        normalized = codex_final_path.read_text(encoding="utf-8").strip()
        if not normalized:
            normalized = normalize_executor_output(executor, result.stdout, result.stderr)
    else:
        normalized = normalize_executor_output(executor, result.stdout, result.stderr)

    runtime_model = (
        _parse_codex_runtime_model(result.stdout, result.stderr)
        if executor == "codex"
        else None
    )
    effective_model = runtime_model or model_name
    effective_model_source = "Codex runtime banner" if runtime_model else model_source

    report_path = RUNS_DIR / f"{iteration:04d}_executor_{executor}.json"
    write_json(
        report_path,
        {
            "timestamp": utc_now(),
            "iteration": iteration,
            "executor": executor,
            "model": effective_model,
            "model_source": effective_model_source,
            "task": task,
            "command": [
                "<prompt>" if arg == prompt else (clean_cli_name(arg) if i == 0 else arg)
                for i, arg in enumerate(command)
            ],
            "prompt_transport": "stdin" if stdin_text is not None else "argv",
            "return_code": result.return_code,
            "timed_out": result.timed_out,
            "duration_seconds": round(result.duration_seconds, 3),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "normalized_report": normalized,
        },
    )

    if runtime_model and runtime_model != model_name:
        print(f"[{iteration}] Executor runtime model: {runtime_model}")
    print(f"[{iteration}] Executor exit={result.return_code}, {result.duration_seconds:.1f}s")
    print(f"[{iteration}] Saved: {report_path.relative_to(APP_DIR)}")
    return result, normalized, report_path


def load_recent_history(limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not HISTORY_FILE.is_file():
        return []
    lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    items: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                items.append(item)
        except json.JSONDecodeError:
            continue
    return items


def make_planner_prompt(
    config: dict[str, Any],
    iteration: int,
    executor: str,
    task: str,
    executor_result: CommandResult,
    executor_report: str,
) -> str:
    planner_rules = read_text(PLANNER_PROMPT_FILE)
    context = read_text(CONTEXT_FILE)
    max_chars = int(config["max_result_chars_for_planner"])
    truncated_report = executor_report[-max_chars:]
    history = load_recent_history(int(config["recent_history_items_for_planner"]))

    history_for_prompt = [
        {
            "iteration": item.get("iteration"),
            "executor": item.get("executor"),
            "task": item.get("task"),
            "executor_return_code": item.get("executor_return_code"),
            "planner_status": item.get("planner", {}).get("status") if isinstance(item.get("planner"), dict) else None,
            "planner_analysis": item.get("planner", {}).get("analysis") if isinstance(item.get("planner"), dict) else None,
            "next_task": item.get("planner", {}).get("next_task") if isinstance(item.get("planner"), dict) else None,
        }
        for item in history
    ]

    return (
        f"{planner_rules}\n\n"
        "=== RESEARCH CONTEXT ===\n"
        f"{context}\n\n"
        "=== RECENT HISTORY ===\n"
        f"{json.dumps(history_for_prompt, ensure_ascii=False, indent=2)}\n\n"
        "=== LATEST ITERATION ===\n"
        f"Iteration: {iteration}\n"
        f"Executor: {executor}\n"
        f"Executor return code: {executor_result.return_code}\n"
        f"Executor timed out: {executor_result.timed_out}\n"
        "Task sent to executor:\n"
        f"{task}\n\n"
        "Executor report:\n"
        f"{truncated_report}\n"
    )


def run_planner(
    config: dict[str, Any],
    iteration: int,
    executor: str,
    task: str,
    executor_result: CommandResult,
    executor_report: str,
) -> tuple[dict[str, Any], Path]:
    CODEX_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    final_path = CODEX_SCRATCH_DIR / f"planner_{iteration:04d}_final.json"
    if final_path.exists():
        final_path.unlink()

    prompt = make_planner_prompt(
        config, iteration, executor, task, executor_result, executor_report
    )
    codex_cfg = config["codex"]
    planner_model, planner_model_source = configured_model(config, "codex")
    command = [
        codex_cfg["command"],
        "exec",
        *_model_args(config, "codex"),
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--output-schema",
        str(PLANNER_SCHEMA),
        "--output-last-message",
        str(final_path),
        *codex_cfg["extra_args"],
        "-",
    ]

    print(f"[{iteration}] Planner: codex")
    print(f"[{iteration}] Planner model: {planner_model} [{planner_model_source}]")
    # Feed the planner prompt through UTF-8 stdin instead of argv.  This avoids
    # Windows .cmd/cmd.exe command-line length limits for large executor reports.
    result = run_command(
        command,
        CODEX_SCRATCH_DIR,
        int(config["planner_timeout_seconds"]),
        stdin_text=prompt,
    )
    if result.return_code != 0:
        diagnostic = (result.stderr or result.stdout).strip()
        raise OrchestratorError(
            f"Codex planner failed with exit code {result.return_code}.\n{diagnostic}"
        )
    if not final_path.is_file():
        raise OrchestratorError(
            "Codex completed but --output-last-message file was not created. "
            "Run `codex exec --help` and verify your Codex CLI is current."
        )

    final_text = final_path.read_text(encoding="utf-8").strip()
    try:
        decision = json.loads(final_text)
    except json.JSONDecodeError as exc:
        raise OrchestratorError(
            f"Codex final message was not valid JSON despite --output-schema: {exc}\n{final_text}"
        ) from exc

    if decision.get("status") not in {"continue", "done"}:
        raise OrchestratorError(f"Invalid planner status: {decision.get('status')!r}")
    if not isinstance(decision.get("analysis"), str):
        raise OrchestratorError("Planner response missing string field: analysis")
    if not isinstance(decision.get("next_task"), str):
        raise OrchestratorError("Planner response missing string field: next_task")
    if decision["status"] == "continue" and not decision["next_task"].strip():
        raise OrchestratorError("Planner returned continue with an empty next_task")

    runtime_model = _parse_codex_runtime_model(result.stdout, result.stderr)
    effective_model = runtime_model or planner_model

    planner_log = RUNS_DIR / f"{iteration:04d}_planner_codex.json"
    write_json(
        planner_log,
        {
            "timestamp": utc_now(),
            "iteration": iteration,
            "model": effective_model,
            "model_source": "Codex runtime banner" if runtime_model else planner_model_source,
            "return_code": result.return_code,
            "duration_seconds": round(result.duration_seconds, 3),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "decision": decision,
        },
    )

    if runtime_model and runtime_model != planner_model:
        print(f"[{iteration}] Planner runtime model: {runtime_model}")
    print(f"[{iteration}] Planner decision: {decision['status']}")
    print(f"[{iteration}] Analysis: {decision['analysis']}")
    if decision["status"] == "continue":
        print(f"[{iteration}] Next task: {decision['next_task']}")
    return decision, planner_log


def save_state(state: State) -> None:
    write_json(STATE_FILE, asdict(state))


def load_state(config: dict[str, Any], reset: bool) -> State:
    if reset and RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CODEX_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    if STATE_FILE.is_file():
        raw = load_json(STATE_FILE)
        try:
            state = State(**raw)
        except TypeError as exc:
            raise OrchestratorError(f"Invalid state file {STATE_FILE}: {exc}") from exc
        if state.executor != config["executor"]:
            raise OrchestratorError(
                "The existing run was started with executor "
                f"{state.executor!r}, but config now selects {config['executor']!r}. "
                "Use --reset to start a new run after switching executors."
            )
        return state

    initial_task = read_text(INITIAL_TASK_FILE)
    state = State(
        iteration=1,
        phase="executor_pending",
        current_task=initial_task,
        executor=config["executor"],
    )
    save_state(state)
    return state


def load_saved_executor_report(path_text: str) -> tuple[CommandResult, str]:
    path = Path(path_text)
    if not path.is_absolute():
        path = (APP_DIR / path).resolve()
    data = load_json(path)
    result = CommandResult(
        command=[],
        return_code=int(data.get("return_code", 1)),
        stdout=str(data.get("stdout", "")),
        stderr=str(data.get("stderr", "")),
        duration_seconds=float(data.get("duration_seconds", 0)),
        timed_out=bool(data.get("timed_out", False)),
    )
    report = str(data.get("normalized_report", ""))
    return result, report


def orchestrate(config: dict[str, Any], project_dir: Path, reset: bool, once: bool) -> None:
    state = load_state(config, reset=reset)
    max_iterations = int(config["max_iterations"])

    if state.phase == "done":
        print(f"Run already completed: {state.done_reason or 'done'}")
        print("Use --reset to start over.")
        return

    while state.iteration <= max_iterations:
        iteration = state.iteration
        task = state.current_task

        if state.phase == "executor_pending":
            result, report, report_path = run_executor(config, project_dir, task, iteration)
            state.last_executor_report_path = str(report_path.relative_to(APP_DIR))

            if result.return_code != 0 and bool(config["stop_on_executor_error"]):
                # Keep the task pending so a rerun retries the same task after the
                # user fixes the underlying problem. The failed report remains on disk.
                state.phase = "executor_pending"
                save_state(state)
                raise OrchestratorError(
                    f"Executor failed at iteration {iteration} (exit={result.return_code}). "
                    f"Inspect {report_path}. After fixing the cause, rerun to retry the same task."
                )

            state.phase = "planner_pending"
            save_state(state)
        elif state.phase == "planner_pending":
            if not state.last_executor_report_path:
                raise OrchestratorError("State is planner_pending but no executor report is recorded")
            result, report = load_saved_executor_report(state.last_executor_report_path)
        else:
            raise OrchestratorError(f"Unknown state phase: {state.phase}")

        decision, planner_log = run_planner(
            config,
            iteration,
            state.executor,
            task,
            result,
            report,
        )

        append_jsonl(
            HISTORY_FILE,
            {
                "timestamp": utc_now(),
                "iteration": iteration,
                "executor": state.executor,
                "task": task,
                "executor_return_code": result.return_code,
                "executor_report_path": state.last_executor_report_path,
                "planner_log_path": str(planner_log.relative_to(APP_DIR)),
                "planner": decision,
            },
        )

        if decision["status"] == "done":
            state.phase = "done"
            state.done_reason = decision["analysis"]
            save_state(state)
            print("\nOrchestration completed by planner decision.")
            return

        state.iteration += 1
        state.current_task = decision["next_task"].strip()
        state.phase = "executor_pending"
        state.last_executor_report_path = None
        save_state(state)

        if once:
            print("\n--once: completed one executor + planner cycle.")
            print(f"Next task is saved in {STATE_FILE.relative_to(APP_DIR)}")
            return

    state.phase = "done"
    state.done_reason = f"Reached max_iterations={max_iterations}"
    save_state(state)
    print(f"\nStopped: reached max_iterations={max_iterations}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Codex planner + configurable Claude Code / Antigravity / Codex executor orchestrator"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config.json (default: ./config.json next to this script)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration and CLI availability, then exit",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete orchestration state/logs and start a new run",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one executor + planner cycle and save the next task",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    try:
        config = normalize_config(load_json(config_path))
        project_dir = resolve_project_dir(config, config_path)

        if args.check:
            return 0 if run_checks(config, project_dir) else 2

        if not run_checks(config, project_dir):
            print("\nFix the missing prerequisites above before running.", file=sys.stderr)
            return 2

        orchestrate(config, project_dir, reset=args.reset, once=args.once)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. State was preserved; rerun the same command to resume.", file=sys.stderr)
        return 130
    except OrchestratorError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
