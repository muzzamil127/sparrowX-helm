#!/usr/bin/env python3
"""
sync_values.py
==============
Called by the GitHub Actions workflow in sparrowX-deploy.

For each service whose values.yaml changed, this script:
  1. Reads the git diff
  2. Extracts the corresponding section from consolidated values.yaml
  3. Calls the Gemini API to produce JSON patch operations
  4. Applies those patches safely (preserving YAML anchors and comments)
  5. Writes the updated consolidated values.yaml back to disk

Environment variables (set by the workflow):
  GEMINI_API_KEY          – Google Gemini API key
  CHANGED_SERVICES        – space-separated list of service folder names
  CONSOLIDATED_VALUES_PATH – path to consolidated values.yaml
  GITHUB_SHA              – source commit SHA (for logging)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from typing import Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# SERVICE → CONSOLIDATED KEY MAPPING
# Add an entry here whenever a new service is onboarded.
# Key   = folder name in sparrowX-deploy
# Value = top-level YAML key in consolidated values.yaml
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_KEY_MAP: dict[str, str] = {
    "platform-service-audit-trail":         "auditTrail",
    "platform-service-channel":             "channel",
    "platform-service-message-store":       "messageStore",
    "platform-service-user-management":     "userManagement",
    "platform-service-configuration":       "configuration",
    "platform-service-notification-gateway": "notificationGateway",
    "platform-service-batch":               "batch",
    "platform-service-host":                "host",
    "platform-service-lookup":              "lookup",
    "platform-service-error-management":    "errorManagement",
    "raast-service-cas":                    "raastServiceCas",
    # Add more services here as needed
}

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash-latest:generateContent"
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_git_diff(service_path: str) -> str:
    """Return the unified diff for <service>/values.yaml between HEAD~1 and HEAD."""
    result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--", f"{service_path}/values.yaml"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def extract_section_bounds(content: str, key: str) -> tuple[Optional[int], Optional[int]]:
    """
    Return (start_line_idx, end_line_idx) of the top-level YAML section for `key`.
    end_line_idx points to the line *before* the next top-level key (exclusive).
    Returns (None, None) if the key is not found.
    """
    lines = content.splitlines()
    start: Optional[int] = None

    for i, line in enumerate(lines):
        if re.match(rf"^{re.escape(key)}\s*:", line):
            start = i
            break

    if start is None:
        return None, None

    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        # Next top-level key: non-empty, non-comment, starts with a letter/underscore
        if line and not line[0].isspace() and line[0] not in ("#", "-"):
            if re.match(r"^[a-zA-Z_]", line):
                end = i
                break

    return start, end


def call_gemini(prompt: str, api_key: str) -> str:
    """Call the Gemini API and return the raw text of the first candidate."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.05,   # near-deterministic
            "maxOutputTokens": 4096,
        },
    }
    resp = requests.post(
        GEMINI_URL,
        params={"key": api_key},
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def extract_json_from_response(text: str) -> list[dict]:
    """
    Extract a JSON array from Gemini's response, tolerating markdown code fences.
    Returns a list of patch objects: [{path, value}, ...]
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        # Try to find a JSON array anywhere in the response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return []


def apply_patch(section_lines: list[str], path: str, new_value: str) -> list[str]:
    """
    Apply a single patch to a YAML section (as a list of lines).

    path  – dot-notation relative to the section root, e.g. "image.tag"
    value – the new scalar value (string representation)

    Only handles simple scalar replacements (string, number, bool).
    Skips lines that contain YAML anchors (*) to preserve them.
    """
    # Build the innermost key from the dot path
    parts = path.strip(".").split(".")
    leaf_key = parts[-1]

    # Indentation pattern: find the leaf key in the section
    # We look for:  <spaces><key>: <old_value>
    pattern = re.compile(
        rf"^(\s*{re.escape(leaf_key)}\s*:\s*)(.+)$"
    )

    for i, line in enumerate(section_lines):
        m = pattern.match(line)
        if m:
            current_value = m.group(2).strip()
            # Never touch lines referencing a YAML anchor
            if current_value.startswith("*"):
                print(f"  Skipping anchor reference: {line.strip()}")
                continue
            # Quote strings if needed
            if isinstance(new_value, str) and not re.match(r'^[\d.]+$', new_value) \
                    and new_value.lower() not in ("true", "false", "null") \
                    and not new_value.startswith('"'):
                formatted = f'"{new_value}"'
            else:
                formatted = str(new_value)
            section_lines[i] = m.group(1) + formatted
            print(f"  Patched [{leaf_key}]: {current_value!r} → {formatted!r}")
            return section_lines

    print(f"  Warning: could not locate key '{leaf_key}' in section — skipping patch.")
    return section_lines


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    changed_services_str = os.environ.get("CHANGED_SERVICES", "").strip()
    consolidated_path = os.environ.get(
        "CONSOLIDATED_VALUES_PATH", "helm-consolidated/CHART/values.yaml"
    )

    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)

    if not changed_services_str:
        print("No changed services — nothing to do.")
        sys.exit(0)

    changed_services = changed_services_str.split()
    print(f"Services to sync: {changed_services}")

    consolidated_content = read_file(consolidated_path)
    file_modified = False

    for service in changed_services:
        print(f"\n{'─' * 60}")
        print(f"Processing: {service}")

        consolidated_key = SERVICE_KEY_MAP.get(service)
        if not consolidated_key:
            print(
                f"  WARNING: '{service}' is not in SERVICE_KEY_MAP.\n"
                f"  Add it to .github/scripts/sync_values.py and re-run."
            )
            continue

        diff = get_git_diff(service)
        if not diff:
            print(f"  No diff found — skipping.")
            continue

        print(f"  Diff:\n{textwrap.indent(diff, '    ')}")

        try:
            single_values = read_file(f"{service}/values.yaml")
        except FileNotFoundError:
            print(f"  ERROR: {service}/values.yaml not found.")
            continue

        start, end = extract_section_bounds(consolidated_content, consolidated_key)
        if start is None:
            print(f"  ERROR: Key '{consolidated_key}' not found in consolidated values.yaml.")
            continue

        lines = consolidated_content.splitlines()
        section_lines = lines[start:end]
        section_text = "\n".join(section_lines)

        # ── Build Gemini prompt ──────────────────────────────────────────────
        prompt = textwrap.dedent(f"""
            You are a Kubernetes Helm expert.

            A developer changed the `values.yaml` for a single-service Helm chart.
            Your job is to identify which fields in the consolidated Helm chart need to be updated.

            ## Context
            - Service folder: `{service}`
            - Consolidated chart key: `{consolidated_key}`

            ## Git diff (what changed in single service values.yaml)
            ```diff
            {diff}
            ```

            ## Current consolidated section for `{consolidated_key}`
            ```yaml
            {section_text}
            ```

            ## Instructions
            1. Analyse every `+` line in the diff (added/changed lines).
            2. For each changed field, find its **equivalent** inside the consolidated section.
               Note: field names and structure may differ between the two files (e.g. `replicaCount` in single = `replicas` in consolidated).
            3. Do NOT sync fields that are intentionally different (e.g. port numbers that differ, environment-specific settings).
            4. Do NOT modify lines whose value starts with `*` (YAML anchor aliases).
            5. Return a JSON array of patch operations. Each item must have:
               - `"path"` – dot-notation path **relative to the `{consolidated_key}` root** (e.g. `"image.tag"`)
               - `"value"` – the new value as a string
            6. If there is nothing to sync, return an empty array `[]`.
            7. Return ONLY the JSON array — no explanation, no markdown, no prose.

            Example output:
            [
              {{"path": "image.tag", "value": "v0.0.5"}},
              {{"path": "replicas", "value": "2"}}
            ]
        """).strip()

        print(f"\n  Calling Gemini API...")
        try:
            raw_response = call_gemini(prompt, api_key)
        except requests.RequestException as exc:
            print(f"  ERROR: Gemini API call failed: {exc}")
            continue

        print(f"  Gemini response:\n{textwrap.indent(raw_response, '    ')}")

        patches = extract_json_from_response(raw_response)
        if not patches:
            print(f"  Gemini returned no patches — no changes needed for {service}.")
            continue

        print(f"  Applying {len(patches)} patch(es)...")
        for patch in patches:
            path = patch.get("path", "")
            value = patch.get("value", "")
            if not path:
                continue
            section_lines = apply_patch(list(section_lines), path, str(value))

        # Rebuild the full file content
        new_lines = lines[:start] + section_lines + lines[end:]
        consolidated_content = "\n".join(new_lines)
        if not consolidated_content.endswith("\n"):
            consolidated_content += "\n"
        file_modified = True
        print(f"  Sync complete for: {service}")

    if file_modified:
        write_file(consolidated_path, consolidated_content)
        print(f"\n{'═' * 60}")
        print(f"consolidated values.yaml updated: {consolidated_path}")
    else:
        print(f"\nNo changes written — consolidated values.yaml is unchanged.")


if __name__ == "__main__":
    main()
