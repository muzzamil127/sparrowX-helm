#!/usr/bin/env python3
"""
sync_values.py — Gemini-powered Helm values sync
Syncs changed fields from sparrowX-helm service charts → sparrowX-deploy consolidated chart.

Fixes applied:
  - Upgraded to gemini-2.0-flash (stable, free tier)
  - Exponential back-off retry on 429 / 5xx / timeout (up to 6 attempts)
  - Guards against empty/blocked Gemini candidates
  - Never raises unexpectedly; failed services are warned but don't abort others
  - Exits 0 always (so the workflow diff-check decides whether a PR is needed)
    except when GEMINI_API_KEY is missing (hard failure).
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time

import requests

# ---------------------------------------------------------------------------
# Service folder name (sparrowX-helm) -> consolidated YAML key (sparrowX-deploy)
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

# gemini-2.0-flash — stable, generous free-tier quota, fast
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

MAX_RETRIES  = 6
BASE_BACKOFF = 15   # seconds for first retry
MAX_BACKOFF  = 120  # cap per wait


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_git_diff(service: str) -> str:
    """Return the unified diff for <service>/values.yaml between HEAD~1 and HEAD."""
    result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--", f"{service}/values.yaml"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def call_gemini(prompt: str, api_key: str) -> str:
    """
    Call the Gemini API with exponential back-off on rate-limit / server errors.
    Returns the model's text response.
    Raises RuntimeError after exhausting all retries.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.05,
            "maxOutputTokens": 4096,
        },
    }

    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=90,
            )

            # Transient / rate-limit errors — back off and retry
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
                print(
                    f"  [Gemini] HTTP {resp.status_code} on attempt {attempt}/{MAX_RETRIES}."
                    f" Retrying in {wait}s ..."
                )
                time.sleep(wait)
                last_exc = Exception(f"HTTP {resp.status_code}")
                continue

            resp.raise_for_status()  # any other 4xx is a hard failure

            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                raise ValueError(
                    f"Gemini returned no candidates (blockReason={block_reason})"
                )

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
            wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            print(
                f"  [Gemini] Timeout on attempt {attempt}/{MAX_RETRIES}."
                f" Retrying in {wait}s ..."
            )
            time.sleep(wait)
            last_exc = TimeoutError("Gemini request timed out")

        except requests.exceptions.RequestException as exc:
            wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            print(
                f"  [Gemini] Network error on attempt {attempt}/{MAX_RETRIES}: {exc}."
                f" Retrying in {wait}s ..."
            )
            time.sleep(wait)
            last_exc = exc

        except ValueError:
            raise  # non-retryable logic error — let caller handle it

    raise RuntimeError(
        f"Gemini failed after {MAX_RETRIES} attempts."
        + (f" Last error: {last_exc}" if last_exc else "")
    )


def extract_json(text: str) -> list:
    """
    Extract a JSON array from the model response.
    Handles markdown code-fences and surrounding prose.
    Returns an empty list on failure.
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip()).strip()

    # Try direct parse
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass

    # Fall back: find the first [...] block in the text
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return []


def section_bounds(content: str, key: str):
    """
    Return (start_index, end_index) of the top-level YAML block '<key>:'.
    end_index is exclusive (slice-style).
    Returns (None, 0) if the key is not found.
    """
    lines = content.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(rf"^{re.escape(key)}\s*:", line)
        ),
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


def apply_patch(section_lines: list, path: str, new_value: str) -> list:
    """
    Update the line matching the leaf key of <path> inside <section_lines>.
    - Skips YAML anchor references (value starts with *).
    - Quotes string values that are not plain scalars.
    - Only patches the first matching key.
    Returns the (possibly modified) list.
    """
    leaf = path.strip(".").split(".")[-1]
    pat  = re.compile(rf"^(\s*{re.escape(leaf)}\s*:\s*)(.+)$")

    for i, line in enumerate(section_lines):
        m = pat.match(line)
        if not m:
            continue

        cur = m.group(2).strip()

        # Never touch YAML anchor references
        if cur.startswith("*"):
            print(f"  Skipping YAML anchor: {line.strip()!r}")
            continue

        v = str(new_value).strip()

        is_plain_number = bool(re.match(r"^-?[\d.]+$", v))
        is_plain_bool   = v.lower() in ("true", "false")
        is_plain_null   = v.lower() == "null"
        already_quoted  = v.startswith('"') or v.startswith("'")

        if not (is_plain_number or is_plain_bool or is_plain_null or already_quoted):
            v = f'"{v}"'

        section_lines[i] = m.group(1) + v
        print(f"  Patched [{leaf}]: {cur!r} -> {v!r}")
        return section_lines  # one patch per leaf per call

    print(f"  Warning: key '{leaf}' not found in section -- patch skipped")
    return section_lines


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

    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY is not set.")

    if not services or services == [""]:
        print("No services detected -- nothing to sync.")
        sys.exit(0)

    print(f"Services to sync : {services}")
    print(f"Consolidated path: {con_path}")

    try:
        with open(con_path, encoding="utf-8") as fh:
            content = fh.read()
    except FileNotFoundError:
        sys.exit(f"ERROR: consolidated values file not found at '{con_path}'")

    modified     = False
    failed_svcs  = []

    for svc in services:
        key = SERVICE_KEY_MAP.get(svc)
        if not key:
            print(f"\nWARNING: '{svc}' has no entry in SERVICE_KEY_MAP -- skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing : {svc}  ->  consolidated key: '{key}'")
        print(f"{'='*60}")

        diff = get_git_diff(svc)
        if not diff:
            print(f"  No diff found for {svc} -- skipping")
            continue

        start, end = section_bounds(content, key)
        if start is None:
            print(f"  ERROR: section '{key}' not found in consolidated values.yaml -- skipping")
            failed_svcs.append(svc)
            continue

        lines   = content.splitlines()
        section = "\n".join(lines[start:end])

        prompt = textwrap.dedent(f"""
            You are a Kubernetes Helm expert.
            A developer changed values.yaml in a single-service Helm chart called '{svc}'.
            Your task is to sync only the relevant changes into the consolidated chart section '{key}'.

            ## Git diff (changes made in the service chart)
            ```diff
            {diff}
            ```

            ## Current consolidated section for '{key}'
            ```yaml
            {section}
            ```

            Rules you MUST follow:
            1. Only sync fields that have a clear, unambiguous equivalent in the consolidated section.
            2. NEVER change a line whose YAML value starts with * (those are YAML anchor aliases).
            3. Do NOT sync values that are intentionally different between the two charts
               (e.g. different ports, different namespaces, different replica counts set for scaling).
            4. Field names may differ between charts (e.g. replicaCount <-> replicas, image.tag <-> tag).
            5. Focus primarily on: image tags, replica counts, resource limits/requests, env vars.

            Output format -- return ONLY a JSON array, nothing else:
            [{{"path": "dotted.key.path", "value": "new_value"}}]

            If there is nothing relevant to sync, return exactly: []
            Do NOT include markdown fences, explanations, or any other text.
        """).strip()

        try:
            raw = call_gemini(prompt, api_key)
        except Exception as exc:
            print(f"  ERROR calling Gemini for '{svc}': {exc}")
            failed_svcs.append(svc)
            continue

        patches = extract_json(raw)
        print(f"  Patches received: {patches}")

        if not patches:
            print("  No patches to apply.")
            continue

        sec_lines = list(lines[start:end])
        for patch in patches:
            path  = str(patch.get("path",  "")).strip()
            value = str(patch.get("value", "")).strip()
            if not path:
                print(f"  Skipping patch with empty path: {patch}")
                continue
            sec_lines = apply_patch(sec_lines, path, value)

        lines   = lines[:start] + sec_lines + lines[end:]
        content = "\n".join(lines)
        if not content.endswith("\n"):
            content += "\n"
        modified = True

    # -----------------------------------------------------------------------
    if modified:
        with open(con_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"\nUpdated: {con_path}")
    else:
        print("\nNo changes written -- consolidated values.yaml is already up to date.")

    if failed_svcs:
        print(f"\nWARNING: Services with errors (skipped): {failed_svcs}")
        # Exit 0 so the workflow can still create a PR for services that DID succeed.


if __name__ == "__main__":
    main()
