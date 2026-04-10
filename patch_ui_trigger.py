import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# We want to force the replay to build and playback if user clicks standard simulation buttons.
# Searching for: controls["run_clicked"]
# We will inject conditions so that hitting "Run Validation" or "Start Mission" also triggers the replay.

old_logic = r'''    should_build_replay = (
        controls["run_clicked"]
        or controls["backend_start"]
        or controls["backend_preflight"]
        or controls["backend_validate"]
        or "day8_replay" in st.session_state
    )'''

new_logic = r'''    should_build_replay = (
        controls["run_clicked"]
        or controls["backend_start"]
        or controls["backend_preflight"]
        or controls["backend_validate"]
        or "day8_replay" in st.session_state
    )

    # USER FEEDBACK: "everytime i run the optimization i wanna see this and work on it"
    # Force frontend animation to run visually if validation or start is clicked!
    if controls["backend_validate"] or controls["backend_start"]:
        replay_requested = True
        controls["animate_frontend"] = True
'''

content = content.replace(old_logic, new_logic)

with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
