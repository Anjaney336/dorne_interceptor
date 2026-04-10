import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

old_loop = r'''            if "targets" in live_snapshot:
                for t_entry in live_snapshot["targets"]:
                    backend_live_targets[t_entry["target"]] = t_entry'''

new_loop = r'''            if "targets" in live_snapshot:
                for t_entry in live_snapshot["targets"]:
                    if isinstance(t_entry, dict):
                        t_name = t_entry.get("target", t_entry.get("name", "Unknown"))
                        backend_live_targets[t_name] = t_entry'''

content = content.replace(old_loop, new_loop)

with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Success")
