#!/usr/bin/env python3
"""
sync_values.py — AI-powered Helm values sync (Gemini primary, deterministic fallback)

Flow for each changed service:
  1. Load values.yaml at HEAD~1 and HEAD via `git show`  -> compute ALL changed fields
  2. Gemini receives the changed fields + consolidated section for that service
     and returns exact JSON patches to apply  (primary path)
  3. If Gemini is unavailable/rate-limited, a deterministic fallback applies
     fields whose names match directly or exist in FIELD_REMAP  (fallback path)
  4. Patches are applied at the text level so YAML comments/anchors are preserved
  5. The consolidated values.yaml is written; the workflow diff-check and PR step
     decide whether anything changed and whether a PR is needed

Gemini rate-limit strategy:
  - Prompts are minimal (only changed fields + the relevant section, not full file)
  - Respects Retry-After header on 429 responses
  - Up to MAX_RETRIES attempts with capped exponential back-off
  - On final failure the deterministic fallback still runs, so the sync is never
    completely blocked by API quota issues
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
# Repo mapping
# ---------------------------------------------------------------------------
SERVICE_KEY_MAP: dict[str, str] = {
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

# Deterministic fallback: service field name -> consolidated field name (when different)
FIELD_REMAP: dict[str, str] = {
    "replicaCount": "replicas",
    "appPort":      "appPort",   # dapr
    "appId":        "appId",
}

# ---------------------------------------------------------------------------
# Gemini config
# ---------------------------------------------------------------------------
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
MAX_RETRIES    = 5
BASE_BACKOFF   = 15   # seconds
MAX_BACKOFF    = 90   # cap


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Recursively flatten nested dict to {dot.notation.key: value}."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif isinstance(v, list):
            # Store lists as JSON strings so they compare cleanly
            out[key] = json.dumps(v, default=str)
        else:
            out[key] = v
    return out


def load_yaml_at(service: str, ref: str) -> dict:
    """Parse values.yaml for <service> at git <ref>. Returns {} on any error."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{service}/values.yaml"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        return yaml.safe_load(result.stdout) or {}
    except yaml.YAMLError as exc:
        print(f"  [yaml] Parse error at {ref}: {exc}")
        return {}


def compute_all_changes(service: str) -> dict:
    """
    Return {flat_key: new_value} for EVERY field that changed between
    HEAD~1 and HEAD in <service>/values.yaml. No filter applied here.
    """
    old = load_yaml_at(service, "HEAD~1")
    new = load_yaml_at(service, "HEAD")

    if not new:
        print(f"  [diff] Cannot load {service}/values.yaml at HEAD — skipping")
        return {}

    old_flat = flatten_dict(old) if old else {}
    new_flat  = flatten_dict(new)

    return {k: v for k, v in new_flat.items() if old_flat.get(k) != v}


# ---------------------------------------------------------------------------
# Text-level patching (preserves YAML comments and anchors)
# ---------------------------------------------------------------------------

def detect_line_ending(content: str) -> str:
    """Return '\\r\\n' if content uses CRLF, else '\\n'."""
    return "\r\n" if "\r\n" in content else "\n"


def section_bounds(lines: list[str], key: str) -> tuple[int | None, int]:
    """
    Return (start, end) line indices of the top-level YAML block '<key>:'.
    end is exclusive. Returns (None, 0) if not found.
    """
    start = next(
        (i for i, line in enumerate(lines)
         if re.match(rf"^{re.escape(key)}\s*:", line)),
        None,
    )
    if start is None:
        return None, 0

    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if (
            ln
            and not ln[0].isspace()
            and ln[0] not in ("#", "-")
            and re.match(r"^[a-zA-Z_]", ln)
        ):
            end = i
            break
    return start, end


def patch_section(sec_lines: list[str], leaf: str, new_value: Any) -> bool:
    """
    Find the FIRST line in sec_lines matching '<leaf>: <anything>' and update it.
    - Skips lines whose current value starts with * (YAML anchor alias).
    - Quotes string values that aren't plain scalars.
    Returns True if a line was patched.
    """
    pat = re.compile(rf"^(\s*{re.escape(leaf)}\s*:\s*)(.+)$")

    for i, line in enumerate(sec_lines):
        m = pat.match(line)
        if not m:
            continue
        cur = m.group(2).strip()
        if cur.startswith("*"):
            print(f"  [patch] Skipping anchor: {line.strip()!r}")
            continue

        v = str(new_value).strip()
        is_plain = bool(re.match(r"^-?[\d.]+$", v)) or v.lower() in ("true", "false", "null")
        already_quoted = v.startswith(('"', "'"))
        if not (is_plain or already_quoted):
            v = f'"{v}"'

        sec_lines[i] = m.group(1) + v
        print(f"  [patch] {leaf}: {cur!r} -> {v!r}")
        return True

    print(f"  [patch] Key '{leaf}' not found in section — skipped")
    return False


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, api_key: str) -> str:
    """
    Call Gemini with exponential back-off on 429/5xx/timeout.
    Returns the model's text. Raises RuntimeError after all retries.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     0.05,
            "maxOutputTokens": 2048,
        },
    }
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=60,
            )

            if resp.status_code == 429:
                # Honour Retry-After when present; otherwise use backoff
                retry_after = resp.headers.get("Retry-After")
                wait = min(
                    int(retry_after) if retry_after and retry_after.isdigit()
                    else BASE_BACKOFF * (2 ** (attempt - 1)),
                    MAX_BACKOFF,
                )
                print(f"  [gemini] 429 rate-limit (attempt {attempt}/{MAX_RETRIES}) — waiting {wait}s")
                time.sleep(wait)
                last_err = Exception("429 rate-limit")
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
                print(f"  [gemini] HTTP {resp.status_code} (attempt {attempt}/{MAX_RETRIES}) — waiting {wait}s")
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
            print(f"  [gemini] OK (attempt {attempt}) — {len(text)} chars, preview: {text[:100]!r}")
            return text

        except requests.exceptions.Timeout:
            last_err = TimeoutError("Gemini request timed out")
            print(f"  [gemini] Timeout (attempt {attempt}/{MAX_RETRIES})")

        except requests.exceptions.RequestException as exc:
            last_err = exc
            print(f"  [gemini] Network error (attempt {attempt}/{MAX_RETRIES}): {exc}")

        except ValueError:
            raise  # non-retryable

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts: {last_err}")


def extract_json_list(text: str) -> list:
    """Extract a JSON array from model text (handles markdown fences)."""
    # Strip fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text.strip(), flags=re.MULTILINE).strip()

    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass

    # Fall back: first [...] block in the response
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return []


def gemini_patches(
    svc: str, key: str, changes: dict, section_text: str, api_key: str
) -> list[dict]:
    """
    Ask Gemini which fields to sync and exactly how.
    Returns a list of {"path": "dotted.key", "value": "new_value"} dicts.
    """
    changes_yaml = "\n".join(f"  {k}: {v!r}" for k, v in changes.items())

    prompt = textwrap.dedent(f"""
        You are a Kubernetes Helm expert performing a values sync.

        A developer changed these fields in the individual service chart '{svc}':
        ```
        {changes_yaml}
        ```

        The consolidated Helm chart uses a unified values.yaml.
        Below is the current YAML section for '{key}' in that file:
        ```yaml
        {section_text}
        ```

        Your job: decide which of the changed fields should be mirrored into the
        consolidated section, find the correct key path, and return the new value.

        Rules (strictly follow all):
        1. Only sync a field if there is a clear, unambiguous corresponding key in the
           consolidated section. When field NAMES differ (e.g. replicaCount vs replicas,
           service.port vs service.ports[].port), map them intelligently.
        2. NEVER touch any key whose current value starts with * — those are YAML anchor
           aliases and must remain untouched.
        3. Do NOT sync values that are intentionally different between the two charts
           (e.g. different namespaces, health-check paths, Dapr internal ports).
        4. For array fields (like service.ports), update only the scalar leaf that
           best matches the changed value (e.g. the `port` key inside the first entry).
        5. Return ONLY a JSON array — no prose, no markdown fences:
           [{{"path": "leaf_key", "value": "new_value"}}]
           Use the LEAF key name as it appears in the consolidated section, not the
           full dotted path.
        6. If nothing should be synced, return exactly: []
    """).strip()

    raw     = call_gemini(prompt, api_key)
    patches = extract_json_list(raw)
    print(f"  [gemini] Suggested patches: {patches}")
    return patches


# ---------------------------------------------------------------------------
# Deterministic fallback (no API)
# ---------------------------------------------------------------------------

def deterministic_patches(changes: dict, sec_lines: list[str]) -> list[dict]:
    """
    For each changed field, try an exact leaf-name match in the section.
    Also applies FIELD_REMAP for known renames.
    Returns patches in the same format as gemini_patches.
    """
    section_keys: set[str] = set()
    key_pat = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:")
    for line in sec_lines:
        m = key_pat.match(line)
        if m:
            section_keys.add(m.group(1))

    patches = []
    for flat_key, new_val in changes.items():
        leaf = flat_key.split(".")[-1]
        # Try remapped name first, then leaf name
        for candidate in [FIELD_REMAP.get(leaf), leaf]:
            if candidate and candidate in section_keys:
                patches.append({"path": candidate, "value": str(new_val)})
                break
    return patches


# ---------------------------------------------------------------------------
# Per-service entry point
# ---------------------------------------------------------------------------

def sync_service(
    svc: str, key: str, content: str, eol: str, api_key: str
) -> tuple[str, bool]:
    """
    Apply all relevant patches for one service to `content`.
    Returns (updated_content, was_modified).
    """
    print(f"\n{'='*62}")
    print(f"Service  : {svc}")
    print(f"Cons. key: {key}")
    print(f"{'='*62}")

    changes = compute_all_changes(svc)
    if not changes:
        print("  No field changes detected — nothing to sync")
        return content, False

    print(f"  Detected {len(changes)} changed field(s): {sorted(changes)}")

    lines = content.split(eol)   # split on exact line ending to preserve it
    start, end = section_bounds(lines, key)
    if start is None:
        print(f"  ERROR: section '{key}' not found in consolidated values.yaml")
        return content, False

    sec_lines = list(lines[start:end])
    section_text = "\n".join(sec_lines)
    modified = False

    # ------------------------------------------------------------------
    # 1. Primary path: Gemini decides what to sync and how
    # ------------------------------------------------------------------
    patches: list[dict] = []
    if api_key:
        try:
            patches = gemini_patches(svc, key, changes, section_text, api_key)
        except Exception as exc:
            print(f"  [gemini] Unavailable — falling back to deterministic: {exc}")

    # ------------------------------------------------------------------
    # 2. Fallback path: deterministic exact-name matching
    #    Also runs when Gemini returned nothing (empty patches)
    # ------------------------------------------------------------------
    if not patches:
        print("  [fallback] Running deterministic field matching")
        patches = deterministic_patches(changes, sec_lines)
        print(f"  [fallback] Patches: {patches}")

    # ------------------------------------------------------------------
    # 3. Apply patches
    # ------------------------------------------------------------------
    for p in patches:
        leaf = str(p.get("path", "")).strip()
        val  = str(p.get("value", "")).strip()
        if not leaf:
            continue
        if patch_section(sec_lines, leaf, val):
            modified = True

    # ------------------------------------------------------------------
    # 4. Rebuild content (preserving original line endings)
    # ------------------------------------------------------------------
    if modified:
        lines   = lines[:start] + sec_lines + lines[end:]
        content = eol.join(lines)
        if not content.endswith(eol):
            content += eol

    return content, modified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key  = os.environ.get("GEMINI_API_KEY", "").strip()
    services = os.environ.get("CHANGED_SERVICES", "").strip().split()
    con_path = os.environ.get(
        "CONSOLIDATED_VALUES_PATH",
        "deploy-consolidated/CHART/values.yaml",
    )

    if not services or services == [""]:
        print("No services detected — nothing to sync.")
        sys.exit(0)

    mode = "AI + deterministic fallback" if api_key else "deterministic only (no GEMINI_API_KEY)"
    print(f"Mode             : {mode}")
    print(f"Services to sync : {services}")
    print(f"Consolidated path: {con_path}")

    try:
        with open(con_path, encoding="utf-8") as fh:
            content = fh.read()
    except FileNotFoundError:
        sys.exit(f"ERROR: consolidated values file not found at '{con_path}'")

    # Detect and preserve original line endings
    eol = detect_line_ending(content)
    print(f"Line endings     : {'CRLF' if eol == chr(13)+chr(10) else 'LF'}")

    any_modified = False
    failed_svcs: list[str] = []

    for svc in services:
        key = SERVICE_KEY_MAP.get(svc)
        if not key:
            print(f"\nWARNING: '{svc}' not in SERVICE_KEY_MAP — skipping")
            continue
        try:
            content, modified = sync_service(svc, key, content, eol, api_key)
            if modified:
                any_modified = True
        except Exception as exc:
            print(f"  ERROR processing '{svc}': {exc}")
            failed_svcs.append(svc)

    if any_modified:
        with open(con_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        print(f"\nUpdated: {con_path}")
    else:
        print("\nNo changes written — consolidated values.yaml is already up to date.")

    if failed_svcs:
        print(f"\nWARNING: Services with errors (skipped): {failed_svcs}")
    # Always exit 0 — the workflow diff-check decides if a PR is needed
    sys.exit(0)


if __name__ == "__main__":
    main()
