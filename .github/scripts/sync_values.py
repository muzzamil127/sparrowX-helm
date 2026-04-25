#!/usr/bin/env python3
"""
sync_values.py — Helm values sync (YAML-diff primary, Gemini optional)

Primary path (no API key needed):
  1. Load service values.yaml at HEAD~1 and HEAD via git show
  2. Flatten both to dot-notation dicts, compute changed fields
  3. Map field names via FIELD_REMAP table (e.g. replicaCount -> replicas)
  4. Patch the consolidated values.yaml in-place (preserves comments/anchors)

Optional Gemini path:
  - For fields NOT in SYNCABLE_FIELDS, call Gemini if GEMINI_API_KEY is set
  - 429 / timeout -> exponential back-off, then skip gracefully (non-blocking)
  - Gemini failure never prevents the deterministic sync from completing

Exit codes:
  0  always — let the workflow diff-check decide whether a PR is needed
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from typing import Any

import requests
import yaml

# ---------------------------------------------------------------------------
# Service folder (sparrowX-helm) -> consolidated section key (sparrowX-deploy)
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

# Service field name -> consolidated field name (when names differ)
FIELD_REMAP = {
    "replicaCount": "replicas",
}

# Fields that are always safe to sync deterministically (no AI needed)
SYNCABLE_FIELDS = {
    "replicaCount",
    "image.tag",
    "image.repository",
    "image.pullPolicy",
    "resources.requests.memory",
    "resources.requests.cpu",
    "resources.limits.memory",
    "resources.limits.cpu",
    "autoscaling.enabled",
    "autoscaling.minReplicas",
    "autoscaling.maxReplicas",
    "autoscaling.targetCPUUtilizationPercentage",
    "autoscaling.targetMemoryUtilizationPercentage",
}

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
MAX_RETRIES  = 4
BASE_BACKOFF = 20   # seconds for first retry
MAX_BACKOFF  = 60   # cap per wait


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict to {dot.notation.key: value}."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            result.update(flatten_dict(v, key))
        else:
            result[key] = v
    return result


def load_values_at(service: str, ref: str) -> dict:
    """Return parsed values.yaml for <service> at git ref, or {} on error."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{service}/values.yaml"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        return yaml.safe_load(result.stdout) or {}
    except yaml.YAMLError as exc:
        print(f"  Warning: YAML parse error ({ref}): {exc}")
        return {}


def compute_changes(service: str) -> dict:
    """
    Return {flat_key: new_value} for every field that changed
    between HEAD~1 and HEAD in <service>/values.yaml.
    """
    old = load_values_at(service, "HEAD~1")
    new = load_values_at(service, "HEAD")

    if not new:
        print(f"  Could not load {service}/values.yaml at HEAD -- skipping")
        return {}

    old_flat = flatten_dict(old) if old else {}
    new_flat  = flatten_dict(new)

    changes = {}
    for key, val in new_flat.items():
        if old_flat.get(key) != val:
            changes[key] = val

    return changes


# ---------------------------------------------------------------------------
# Consolidated values.yaml patching (text-level to preserve comments/anchors)
# ---------------------------------------------------------------------------

def section_bounds(content: str, key: str):
    """
    Return (start_line_idx, end_line_idx) of the top-level YAML block '<key>:'.
    end is exclusive (slice-style). Returns (None, 0) if not found.
    """
    lines = content.splitlines()
    start = next(
        (i for i, line in enumerate(lines)
         if re.match(rf"^{re.escape(key)}\s*:", line)),
        None,
    )
    if start is None:
        return None, 0

    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if (
            line
            and not line[0].isspace()
            and line[0] not in ("#", "-")
            and re.match(r"^[a-zA-Z_]", line)
        ):
            end = i
            break

    return start, end


def apply_patch(section_lines: list, leaf: str, new_value: Any) -> bool:
    """
    Find the first line matching '<leaf>: ...' in section_lines and update it.
    - Skips YAML anchor references (value starts with *).
    - Quotes string values that are not plain scalars.
    Returns True if a patch was applied.
    """
    pat = re.compile(rf"^(\s*{re.escape(leaf)}\s*:\s*)(.+)$")

    for i, line in enumerate(section_lines):
        m = pat.match(line)
        if not m:
            continue

        cur = m.group(2).strip()
        if cur.startswith("*"):
            print(f"  Skipping YAML anchor: {line.strip()!r}")
            continue

        v = str(new_value).strip()
        is_plain = (
            bool(re.match(r"^-?[\d.]+$", v))
            or v.lower() in ("true", "false", "null")
        )
        already_quoted = v.startswith('"') or v.startswith("'")
        if not (is_plain or already_quoted):
            v = f'"{v}"'

        section_lines[i] = m.group(1) + v
        print(f"  Patched [{leaf}]: {cur!r} -> {v!r}")
        return True

    print(f"  Key '{leaf}' not found in section -- patch skipped")
    return False


# ---------------------------------------------------------------------------
# Gemini (optional AI path for unmapped fields)
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, api_key: str) -> str:
    """
    Call gemini-2.0-flash with exponential back-off on 429 / 5xx / timeout.
    Raises RuntimeError after exhausting all retries.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.05, "maxOutputTokens": 1024},
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
                retry_after = resp.headers.get("Retry-After")
                wait = min(
                    int(retry_after) if retry_after else BASE_BACKOFF * attempt,
                    MAX_BACKOFF,
                )
                print(f"  [Gemini] 429 (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s")
                time.sleep(wait)
                last_err = Exception("429 rate-limit")
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = min(BASE_BACKOFF * attempt, MAX_BACKOFF)
                print(f"  [Gemini] HTTP {resp.status_code} (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s")
                time.sleep(wait)
                last_err = Exception(f"HTTP {resp.status_code}")
                continue

            resp.raise_for_status()
            data       = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                block = data.get("promptFeedback", {}).get("blockReason", "unknown")
                raise ValueError(f"No candidates (blockReason={block})")

            text = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            print(f"  [Gemini] OK (attempt {attempt}) -- preview: {text[:120]!r}")
            return text

        except requests.exceptions.Timeout:
            last_err = TimeoutError("request timed out")
            print(f"  [Gemini] Timeout (attempt {attempt}/{MAX_RETRIES})")

        except requests.exceptions.RequestException as exc:
            last_err = exc
            print(f"  [Gemini] Network error (attempt {attempt}/{MAX_RETRIES}): {exc}")

        except ValueError:
            raise  # non-retryable

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts: {last_err}")


def extract_json(text: str) -> list:
    """Extract a JSON array from model response (handles markdown fences)."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip()).strip()

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
# Per-service sync
# ---------------------------------------------------------------------------

def sync_service(svc: str, key: str, content: str, api_key: str) -> tuple:
    """
    Sync one service into `content`.
    Returns (updated_content, was_modified).
    """
    print(f"\n{'='*60}")
    print(f"Processing: {svc}  ->  key: '{key}'")
    print(f"{'='*60}")

    changes = compute_changes(svc)
    if not changes:
        print("  No field changes detected -- skipping")
        return content, False

    print(f"  Changed fields: {sorted(changes.keys())}")

    start, end = section_bounds(content, key)
    if start is None:
        print(f"  ERROR: section '{key}' not found in consolidated values.yaml")
        return content, False

    lines     = content.splitlines()
    sec_lines = list(lines[start:end])
    modified  = False

    # ------------------------------------------------------------------
    # Phase 1: deterministic sync for known fields (no API call needed)
    # ------------------------------------------------------------------
    for flat_key, new_val in changes.items():
        if flat_key not in SYNCABLE_FIELDS:
            continue
        leaf = FIELD_REMAP.get(flat_key, flat_key.split(".")[-1])
        if apply_patch(sec_lines, leaf, new_val):
            modified = True

    # ------------------------------------------------------------------
    # Phase 2: Gemini for unmapped / unknown fields (optional, non-blocking)
    # ------------------------------------------------------------------
    unmapped = {k: v for k, v in changes.items() if k not in SYNCABLE_FIELDS}
    if unmapped and api_key:
        section_text = "\n".join(sec_lines)
        prompt = textwrap.dedent(f"""
            You are a Kubernetes Helm expert.
            These fields changed in the service chart '{svc}'.
            Decide which should be synced to the consolidated section '{key}'.

            Changed fields (dot-notation -> new value):
            {json.dumps(unmapped, indent=2, default=str)}

            Current consolidated section:
            ```yaml
            {section_text}
            ```

            Rules:
            1. Only sync fields with a clear, unambiguous equivalent.
            2. NEVER touch a line whose value starts with * (YAML anchor aliases).
            3. Do NOT sync values intentionally different (ports, namespaces, etc.).
            4. Output ONLY a JSON array: [{{"path": "dotted.key", "value": "new_value"}}]
            5. Return exactly [] if nothing should be synced.
            No markdown fences, no explanations.
        """).strip()

        try:
            raw     = call_gemini(prompt, api_key)
            patches = extract_json(raw)
            print(f"  [Gemini] patches: {patches}")
            for p in patches:
                leaf = str(p.get("path", "")).split(".")[-1].strip()
                val  = str(p.get("value", "")).strip()
                if leaf and apply_patch(sec_lines, leaf, val):
                    modified = True
        except Exception as exc:
            print(f"  [Gemini] Skipped (non-blocking): {exc}")

    # ------------------------------------------------------------------
    # Rebuild content if anything changed
    # ------------------------------------------------------------------
    if modified:
        lines   = lines[:start] + sec_lines + lines[end:]
        content = "\n".join(lines)
        if not content.endswith("\n"):
            content += "\n"

    return content, modified


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key  = os.environ.get("GEMINI_API_KEY", "").strip()
    services = os.environ.get("CHANGED_SERVICES", "").strip().split()
    con_path = os.environ.get(
        "CONSOLIDATED_VALUES_PATH",
        "deploy-consolidated/CHART/values.yaml",
    )

    if not services or services == [""]:
        print("No services detected -- nothing to sync.")
        sys.exit(0)

    if not api_key:
        print("INFO: GEMINI_API_KEY not set -- deterministic-only mode (no AI for unmapped fields)")
    else:
        print("INFO: GEMINI_API_KEY present -- AI path enabled for unmapped fields")

    print(f"Services to sync : {services}")
    print(f"Consolidated path: {con_path}")

    try:
        with open(con_path, encoding="utf-8") as fh:
            content = fh.read()
    except FileNotFoundError:
        sys.exit(f"ERROR: consolidated values file not found at '{con_path}'")

    any_modified = False
    failed_svcs  = []

    for svc in services:
        key = SERVICE_KEY_MAP.get(svc)
        if not key:
            print(f"\nWARNING: '{svc}' not in SERVICE_KEY_MAP -- skipping")
            continue
        try:
            content, modified = sync_service(svc, key, content, api_key)
            if modified:
                any_modified = True
        except Exception as exc:
            print(f"  ERROR processing '{svc}': {exc}")
            failed_svcs.append(svc)

    if any_modified:
        with open(con_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"\nUpdated: {con_path}")
    else:
        print("\nNo changes written -- consolidated values.yaml is already up to date.")

    if failed_svcs:
        print(f"\nWARNING: Services with errors (skipped): {failed_svcs}")


if __name__ == "__main__":
    main()
