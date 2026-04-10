import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Update signature
old_sig = r'''def _build_day8_target_results_frame(
    replay: MissionReplay,
    validation_report: MonteCarloSummary | dict[str, Any] | None,
    upto_index: int | None = None,
) -> pd.DataFrame:'''

new_sig = r'''def _build_day8_target_results_frame(
    replay: MissionReplay,
    validation_report: MonteCarloSummary | dict[str, Any] | None,
    upto_index: int | None = None,
    simulation: dict[str, Any] | None = None,
) -> pd.DataFrame:'''
content = content.replace(old_sig, new_sig)

# Update the empty branch
old_empty = r'''    if len(replay.frames) == 0:
        if "backend_validation_frame" in st.session_state and st.session_state["backend_validation_frame"]:
            df = pd.DataFrame(st.session_state["backend_validation_frame"])
            for col in ["ekf_success_rate", "ekf_mean_miss_distance_m", "status", "threat_level", "distance_m", "spoofing_active", "drift_rate_mps", "spoof_offset_m", "innovation_m", "innovation_gate", "estimated_error_m", "packet_dropped", "spoofing_detected", "jammed"]:
                if col not in df.columns:
                    df[col] = 0.0
            return df
        return pd.DataFrame(
            columns=['''

new_empty = r'''    if len(replay.frames) == 0:
        if "backend_validation_frame" in st.session_state and st.session_state["backend_validation_frame"]:
            df = pd.DataFrame(st.session_state["backend_validation_frame"])
            for col in ["ekf_success_rate", "ekf_mean_miss_distance_m", "status", "threat_level", "distance_m", "spoofing_active", "drift_rate_mps", "spoof_offset_m", "innovation_m", "innovation_gate", "estimated_error_m", "packet_dropped", "spoofing_detected", "jammed"]:
                if col not in df.columns:
                    df[col] = 0.0
            return df
        if simulation is not None and len(simulation.get("times", [])) > 0:
            return pd.DataFrame([{
                "target": "Active Target 0",
                "status": "INTERCEPTED" if simulation.get("success", False) else "ACTIVE",
                "threat_level": 0.85,
                "distance_m": round(simulation.get("final_distance_m", 0.0), 3),
                "spoofing_active": True,
                "drift_rate_mps": 0.3,
                "spoof_offset_m": round(simulation.get("spoof_offsets", [0.0])[-1], 3) if simulation.get("spoof_offsets") else 0.0,
                "innovation_m": round(simulation.get("innovations", [0.0])[-1], 3) if simulation.get("innovations") else 0.0,
                "innovation_gate": 0.5,
                "estimated_error_m": round(simulation.get("tracking_errors", [0.0])[-1], 3) if simulation.get("tracking_errors") else 0.0,
                "packet_dropped": False,
                "spoofing_detected": False,
                "jammed": simulation.get("success", False),
                "ekf_success_rate": 100.0 if simulation.get("success", False) else 0.0,
                "ekf_mean_miss_distance_m": round(simulation.get("final_distance_m", 0.0), 3),
            }])
        return pd.DataFrame(
            columns=['''

content = content.replace(old_empty, new_empty)

# Update calls to _build_day8_target_results_frame

content = content.replace('''scenario_results_placeholder.dataframe(
        _build_day8_target_results_frame(
            day8_replay,
            backend_state.get("validation") if backend_state and backend_state.get("validation") is not None else day8_validation,
        ),''', '''scenario_results_placeholder.dataframe(
        _build_day8_target_results_frame(
            day8_replay,
            backend_state.get("validation") if backend_state and backend_state.get("validation") is not None else day8_validation,
            simulation=simulation,
        ),''')

content = content.replace('''scenario_results_placeholder.dataframe(
            _build_day8_target_results_frame(
                day8_replay,
                active_validation,
                upto_index=min(step_index, max(len(day8_replay.frames) - 1, 0)),
            ),''', '''scenario_results_placeholder.dataframe(
            _build_day8_target_results_frame(
                day8_replay,
                active_validation,
                upto_index=min(step_index, max(len(day8_replay.frames) - 1, 0)),
                simulation=simulation,
            ),''')

with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Success")
