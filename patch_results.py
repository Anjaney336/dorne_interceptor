import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

old_func = r'''def _build_day8_target_results_frame(
    replay: MissionReplay,
    validation_report: MonteCarloSummary | dict[str, Any] | None,
    upto_index: int | None = None,
) -> pd.DataFrame:
    if len(replay.frames) == 0:
        return pd.DataFrame(
            columns=[
                "target",
                "status",
                "threat_level",
                "distance_m",
                "spoofing_active",
                "drift_rate_mps",
                "spoof_offset_m",
                "innovation_m",
                "innovation_gate",
                "estimated_error_m",
                "packet_dropped",
                "spoofing_detected",
                "jammed",
                "ekf_success_rate",
                "ekf_mean_miss_distance_m",
            ]
        )'''

new_func = r'''def _build_day8_target_results_frame(
    replay: MissionReplay,
    validation_report: MonteCarloSummary | dict[str, Any] | None,
    upto_index: int | None = None,
) -> pd.DataFrame:
    if len(replay.frames) == 0:
        if "backend_validation_frame" in st.session_state and st.session_state["backend_validation_frame"]:
            df = pd.DataFrame(st.session_state["backend_validation_frame"])
            for col in ["ekf_success_rate", "ekf_mean_miss_distance_m", "status", "threat_level", "distance_m", "spoofing_active", "drift_rate_mps", "spoof_offset_m", "innovation_m", "innovation_gate", "estimated_error_m", "packet_dropped", "spoofing_detected", "jammed"]:
                if col not in df.columns:
                    df[col] = 0.0
            return df
        return pd.DataFrame(
            columns=[
                "target",
                "status",
                "threat_level",
                "distance_m",
                "spoofing_active",
                "drift_rate_mps",
                "spoof_offset_m",
                "innovation_m",
                "innovation_gate",
                "estimated_error_m",
                "packet_dropped",
                "spoofing_detected",
                "jammed",
                "ekf_success_rate",
                "ekf_mean_miss_distance_m",
            ]
        )'''

content = content.replace(old_func, new_func)

old_port = 'backend_port = int(st.number_input("FastAPI Port", min_value=1, max_value=65535, value=8765, step=1))'
new_port = 'backend_port = int(st.number_input("FastAPI Port", min_value=1, max_value=65535, value=8000, step=1))'

content = content.replace(old_port, new_port)

with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
