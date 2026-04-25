#!/usr/bin/env python3
"""
sync_values.py
--------------
1. Load old/new service values.yaml via git show -> compute every changed field
2. Call Gemini AI with a focused prompt -> it returns exact JSON patches
3. If Gemini fails (rate-limit/unavailable) -> deterministic fallback for
   known safe field mappings (replicaCount, image.tag, resources, autoscaling)
4. Apply patches at text level (preserves YAML comments and anchor aliases)
5. Write consolidated values.yaml; workflow diff-check handles the PR
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time

import requests
import yaml

# ---------------------------------------------------------------------------
# Service folder  ->  consolidated section key
# ---------------------------------------------------------------------------
SERVICE_KEY_MAP = {
    "platform-service-audit-trail":          "auditTrail",
    "platform-service-channel":              "channel",
    "platform-service-message-store":        "messageStore",
    "platform-service-user-management":      "userManagement",
    "platform-service-configuration":        "configuration",
    "platform-service-notification-gateway": "notificationGateway",
    "platform-service-batch":                "batch",
    "platform-service-host":                 "host",
    "platform-service-lookup":               "lookup",
    "platform-service-error-management":     "errorManagement",
    "raast-service-cas":                     "raastServiceCas",
}

# Explicit safe mappings: service flat-path -> (context_keyword, consolidated_leaf)
# context_keyword: the line immediately above the leaf (e.g. "requests", "limits")
#                  or "" to match the first occurrence anywhere in the section
FIELD_SYNC_MAP = {
    "replicaCount":                               ("",         "replicas"),
    "image.tag":                                  ("image",    "tag"),
    "image.repository":                           ("image",    "repository"),
    "image.pullPolicy":                           ("image",    "pullPolicy"),
    "resources.requests.memory":                  ("requests", "memory"),
    "resources.requests.cpu":                     ("requests", "cpu"),
    "resources.limits.memory":                    ("limits",   "memory"),
    "resources.limits.cpu":                       ("limits",   "cpu"),
    "autoscaling.enabled":                        ("autoscaling", "enabled"),
    "autoscaling.minReplicas":                    ("autoscaling", "minReplicas"),
    "autoscaling.maxReplicas":                    ("autoscaling", "maxReplicas"),
    "autoscaling.targetCPUUtilizationPercentage": ("autoscaling", "targetCPUUtilizationPercentage"),
    "autoscaling.targetMemoryUtilizationPercentage": ("autoscaling", "targetMemoryUtilizationPercentage"),
}

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL   = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
MAX_RETRIES  = 3
BASE_BACKOFF = 10   # keep retries short so fallback runs quickly on rate-limit


def call_gemini(prompt: str, api_key: str) -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.05, "maxOutputTokens": 2048},
    }
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "")
                wait = min(
                    int(retry_after) if retry_after.isdigit() else BASE_BACKOFF * attempt,
                    60,
                )
                print(f"  [gemini] 429 (attempt {attempt}/{MAX_RETRIES}) — wait {wait}s")
                time.sleep(wait)
                last_err = Exception("429")
                continue
            if resp.status_code in (500, 502, 503, 504):
                wait = BASE_BACKOFF * attempt
                print(f"  [gemini] HTTP {resp.status_code} (attempt {attempt}/{MAX_RETRIES}) — wait {wait}s")
                time.sleep(wait)
                last_err = Exception(f"HTTP {resp.status_code}")
                continue
            resp.raise_for_status()
            data       = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise ValueError("No candidates returned by Gemini")
            text = (
                candidates[0].get("content", {})
                .get("parts", [{}])[0]
                .get("text", "").strip()
            )
            print(f"  [gemini] OK (attempt {attempt})")
            return text
        except requests.exceptions.Timeout:
            print(f"  [gemini] Timeout (attempt {attempt}/{MAX_RETRIES})")
            last_err = TimeoutError()
        except requests.exceptions.RequestException as exc:
            print(f"  [gemini] Network error: {exc}")
            last_err = exc
        except ValueError:
            raise
    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts: {last_err}")


def parse_json_list(text: str) -> list:
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE).strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# YAML diff helpers
# ---------------------------------------------------------------------------

def flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, list):
            out[key] = v   # store as-is; compare with ==
        else:
            out[key] = v
    return out


def load_yaml(service: str, ref: str) -> dict:
    r = subprocess.run(
        ["git", "show", f"{ref}:{service}/values.yaml"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {}
    try:
        return yaml.safe_load(r.stdout) or {}
    except yaml.YAMLError:
        return {}


def changed_fields(service: str) -> dict:
    """Return {flat_key: new_value} for every field that changed HEAD~1->HEAD."""
    old = flatten(load_yaml(service, "HEAD~1"))
    new = flatten(load_yaml(service, "HEAD"))
    if not new:
        return {}
    result = {k: v for k, v in new.items() if old.get(k) != v}
    print(f"  [diff] Changed fields ({len(result)}): {sorted(result.keys())}")
    print(f"  [diff] Values: { {k: result[k] for k in sorted(result.keys())} }")
    return result


# ---------------------------------------------------------------------------
# Text-level patching
# ---------------------------------------------------------------------------

def read_file(path: str) -> tuple[str, str]:
    """Read file preserving exact line endings. Returns (content, eol)."""
    with open(path, encoding="utf-8", newline="") as f:
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw, eol


def write_file(path: str, content: str) -> None:
    """Write file without any newline translation."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def section_lines(content: str, eol: str, key: str) -> tuple[int | None, int, list[str]]:
    """
    Split content by eol, find the top-level YAML block for <key>.
    Returns (start_idx, end_idx, all_lines).
    """
    lines = content.split(eol)
    start = next(
        (i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(key)}\s*:", ln)),
        None,
    )
    if start is None:
        return None, 0, lines

    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln and not ln[0].isspace() and ln[0] not in ("#", "-") and re.match(r"^[a-zA-Z_]", ln):
            end = i
            break
    return start, end, lines


def patch_leaf(sec: list[str], context: str, leaf: str, value: str) -> bool:
    """
    Find `leaf:` inside sec, optionally requiring `context:` to appear
    somewhere above it in the same indented block. Skips anchor aliases (*).
    Returns True if patched.
    """
    leaf_pat = re.compile(rf"^(\s*{re.escape(leaf)}\s*:\s*)(.+)$")

    # Build candidate line indices where the leaf matches
    candidates = [i for i, ln in enumerate(sec) if leaf_pat.match(ln)]
    if not candidates:
        print(f"  [patch] '{leaf}' not found in section — skipped")
        return False

    chosen = candidates[0]   # default: first occurrence

    if context:
        # Find the candidate that is preceded (somewhere above, within indented block)
        # by a line matching the context keyword
        ctx_pat = re.compile(rf"^\s*{re.escape(context)}\s*:")
        for ci in candidates:
            # Walk backward from ci to find context within the same block
            indent = len(sec[ci]) - len(sec[ci].lstrip())
            for j in range(ci - 1, -1, -1):
                ln = sec[j]
                if not ln.strip() or ln.strip().startswith("#"):
                    continue
                ln_indent = len(ln) - len(ln.lstrip())
                if ln_indent < indent and ctx_pat.match(ln):
                    chosen = ci
                    break
                if ln_indent < indent:
                    break   # left the block without finding context

    m = leaf_pat.match(sec[chosen])
    cur = m.group(2).strip()
    if cur.startswith("*"):
        print(f"  [patch] Skipping anchor alias: {sec[chosen].strip()!r}")
        return False

    # Format the value correctly
    is_plain = bool(re.match(r"^-?[\d.]+$", value)) or value.lower() in ("true", "false", "null")
    already_quoted = value.startswith(('"', "'"))
    v = value if (is_plain or already_quoted) else f'"{value}"'

    sec[chosen] = m.group(1) + v
    print(f"  [patch] {leaf}: {cur!r} -> {v!r}")
    return True


# ---------------------------------------------------------------------------
# Sync one service
# ---------------------------------------------------------------------------

def sync_service(svc: str, con_key: str, content: str, eol: str, api_key: str) -> tuple[str, bool]:
    print(f"\n{'='*60}")
    print(f"Service  : {svc}")
    print(f"Chart key: {con_key}")
    print(f"{'='*60}")

    changes = changed_fields(svc)
    if not changes:
        print("  No changes — skipping")
        return content, False

    start, end, lines = section_lines(content, eol, con_key)
    if start is None:
        print(f"  ERROR: '{con_key}' block not found in consolidated values.yaml")
        return content, False

    sec = list(lines[start:end])
    section_text = "\n".join(sec)
    modified = False

    # ------------------------------------------------------------------
    # Path 1 — Gemini: send ALL changed fields, let AI decide what to sync
    # ------------------------------------------------------------------
    ai_patches: list[dict] = []
    if api_key:
        changes_text = "\n".join(f"  {k}: {v!r}" for k, v in sorted(changes.items()))
        prompt = textwrap.dedent(f"""
            You are a Kubernetes Helm expert performing a values sync.

            Service chart changed ({svc}):
            {changes_text}

            Consolidated section '{con_key}' (current state):
            ```yaml
            {section_text}
            ```

            Task: determine which changed fields should be mirrored into the
            consolidated section and return the exact patches.

            Rules (strictly follow all):
            1. Only sync a field if the consolidated section has a clearly
               corresponding key (even if the name differs, e.g. replicaCount
               vs replicas, service.port vs the port inside service.ports[]).
            2. NEVER touch a key whose current value starts with * (YAML anchor).
            3. Do NOT sync values that are intentionally different (health ports,
               namespaces, Dapr internal ports, etc.).
            4. Return ONLY a JSON array — no prose, no markdown:
               [{{"path": "leaf_key_as_in_consolidated", "value": "new_value"}}]
               Use the leaf key name exactly as it appears in the consolidated section.
            5. If nothing should change, return exactly: []
        """).strip()

        try:
            raw = call_gemini(prompt, api_key)
            ai_patches = parse_json_list(raw)
            print(f"  [gemini] patches: {ai_patches}")
        except Exception as exc:
            print(f"  [gemini] unavailable — using deterministic fallback: {exc}")

    # ------------------------------------------------------------------
    # Path 2 — Deterministic fallback (explicit safe field map, no ambiguity)
    # ------------------------------------------------------------------
    if not ai_patches:
        print("  [fallback] Applying deterministic field map")
        for flat_key, new_val in changes.items():
            mapping = FIELD_SYNC_MAP.get(flat_key)
            if mapping is None:
                print(f"  [fallback] '{flat_key}' not in FIELD_SYNC_MAP — skipped")
                continue
            context, leaf = mapping
            ai_patches.append({
                "path":    leaf,
                "context": context,
                "value":   str(new_val),
            })
        print(f"  [fallback] Patches to apply: {ai_patches}")

    # ------------------------------------------------------------------
    # Apply patches
    # ------------------------------------------------------------------
    for p in ai_patches:
        leaf    = str(p.get("path", "")).strip()
        value   = str(p.get("value", "")).strip()
        context = str(p.get("context", "")).strip()   # may be absent from Gemini patches
        if not leaf:
            continue
        if patch_leaf(sec, context, leaf, value):
            modified = True

    if modified:
        lines = lines[:start] + sec + lines[end:]
        content = eol.join(lines)
        # Preserve trailing newline if original had one
        if not content.endswith(eol):
            content += eol

    return content, modified


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key  = os.environ.get("GEMINI_API_KEY", "").strip()
    services = os.environ.get("CHANGED_SERVICES", "").strip().split()
    con_path = os.environ.get("CONSOLIDATED_VALUES_PATH",
                              "deploy-consolidated/CHART/values.yaml")

    if not services or services == [""]:
        print("No services detected — done.")
        sys.exit(0)

    mode = "Gemini AI + deterministic fallback" if api_key else "deterministic only"
    print(f"Mode     : {mode}")
    print(f"Services : {services}")
    print(f"Chart    : {con_path}")

    try:
        content, eol = read_file(con_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: {con_path} not found")

    print(f"EOL      : {'CRLF' if eol == chr(13)+chr(10) else 'LF'}")

    any_modified = False
    failed: list[str] = []

    for svc in services:
        key = SERVICE_KEY_MAP.get(svc)
        if not key:
            print(f"\nWARNING: '{svc}' has no entry in SERVICE_KEY_MAP — skipping")
            continue
        try:
            content, modified = sync_service(svc, key, content, eol, api_key)
            if modified:
                any_modified = True
        except Exception as exc:
            print(f"ERROR syncing '{svc}': {exc}")
            failed.append(svc)

    if any_modified:
        write_file(con_path, content)
        print(f"\nWrote: {con_path}")
    else:
        print("\nNo changes — consolidated chart already up to date.")

    if failed:
        print(f"WARNING: failed services: {failed}")


if __name__ == "__main__":
    main()
