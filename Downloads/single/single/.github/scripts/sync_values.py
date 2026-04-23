#!/usr/bin/env python3
import json, os, re, subprocess, sys, textwrap
import requests

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

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash-latest:generateContent"
)

def get_git_diff(service):
    r = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--", f"{service}/values.yaml"],
        capture_output=True, text=True
    )
    return r.stdout.strip()

def call_gemini(prompt, api_key):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.05, "maxOutputTokens": 4096}
    }
    r = requests.post(GEMINI_URL, params={"key": api_key}, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

def extract_json(text):
    text = re.sub(r'^```(?:json)?\n?', '', text.strip())
    text = re.sub(r'\n?```$', '', text.strip()).strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except Exception:
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return []

def section_bounds(content, key):
    lines = content.splitlines()
    start = next(
        (i for i, l in enumerate(lines) if re.match(rf'^{re.escape(key)}\s*:', l)),
        None
    )
    if start is None:
        return None, None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l and not l[0].isspace() and l[0] not in ('#', '-') and re.match(r'^[a-zA-Z_]', l):
            end = i
            break
    return start, end

def apply_patch(section_lines, path, new_value):
    leaf = path.strip('.').split('.')[-1]
    pat = re.compile(rf'^(\s*{re.escape(leaf)}\s*:\s*)(.+)$')
    for i, line in enumerate(section_lines):
        m = pat.match(line)
        if m:
            cur = m.group(2).strip()
            if cur.startswith('*'):
                print(f"  Skipping anchor: {line.strip()}")
                continue
            v = str(new_value)
            if (not re.match(r'^[\d.]+$', v) and
                    v.lower() not in ('true', 'false', 'null') and
                    not v.startswith('"')):
                v = f'"{v}"'
            section_lines[i] = m.group(1) + v
            print(f"  Patched [{leaf}]: {cur!r} -> {v!r}")
            return section_lines
    print(f"  Warning: key '{leaf}' not found in section")
    return section_lines

def main():
    api_key  = os.environ.get("GEMINI_API_KEY", "").strip()
    services = os.environ.get("CHANGED_SERVICES", "").strip().split()
    con_path = os.environ.get("CONSOLIDATED_VALUES_PATH", "helm-consolidated/CHART/values.yaml")

    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY not set")
    if not services:
        sys.exit(0)

    content  = open(con_path).read()
    modified = False

    for svc in services:
        key = SERVICE_KEY_MAP.get(svc)
        if not key:
            print(f"WARNING: '{svc}' not in SERVICE_KEY_MAP - skipping")
            continue

        diff = get_git_diff(svc)
        if not diff:
            print(f"No diff for {svc} - skipping")
            continue

        single = open(f"{svc}/values.yaml").read()
        start, end = section_bounds(content, key)
        if start is None:
            print(f"ERROR: '{key}' not found in consolidated values.yaml")
            continue

        lines   = content.splitlines()
        section = "\n".join(lines[start:end])

        prompt = textwrap.dedent(f"""
            You are a Kubernetes Helm expert.
            A developer changed values.yaml in single-service chart '{svc}'.
            Sync the relevant changes to consolidated chart key '{key}'.

            ## Git diff
            ```diff
            {diff}
            ```

            ## Current consolidated section for '{key}'
            ```yaml
            {section}
            ```

            Rules:
            - Only sync fields with a clear equivalent in the consolidated section
            - Never change lines whose value starts with * (YAML anchors)
            - Do not change intentionally different values (e.g. different ports)
            - Field names may differ (e.g. replicaCount -> replicas)

            Return ONLY a JSON array: [{{"path":"image.tag","value":"v0.0.5"}}]
            Return [] if nothing to sync.
        """).strip()

        print(f"\nProcessing {svc} -> {key}")
        try:
            resp = call_gemini(prompt, api_key)
        except Exception as e:
            print(f"  Gemini error: {e}")
            continue

        patches = extract_json(resp)
        if not patches:
            print("  No patches needed")
            continue

        sec_lines = lines[start:end]
        for p in patches:
            sec_lines = apply_patch(list(sec_lines), p.get("path", ""), str(p.get("value", "")))

        lines   = lines[:start] + sec_lines + lines[end:]
        content = "\n".join(lines)
        if not content.endswith("\n"):
            content += "\n"
        modified = True

    if modified:
        open(con_path, "w").write(content)
        print(f"\nUpdated: {con_path}")
    else:
        print("\nNo changes written.")

if __name__ == "__main__":
    main()
