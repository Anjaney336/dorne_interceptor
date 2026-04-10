import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

old_fallback = r'''        if simulation is not None and len(simulation.get("times", [])) > 0:
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
            }])'''

new_fallback = r'''        if simulation is not None and len(simulation.get("times", [])) > 0:
            control_state = st.session_state.get("live_control_state", {})
            num_targets = control_state.get("num_targets", 3)
            packet_loss = control_state.get("packet_loss_rate", 0.05) > 0.0
            spoof_active = control_state.get("noise_level", 0.45) > 0.2
            rows = []
            for i in range(num_targets):
                rows.append({
                    "target": f"Active Target {i}",
                    "status": "INTERCEPTED" if simulation.get("success", False) else "ACTIVE",
                    "threat_level": round(0.85 - (i * 0.05), 3),
                    "distance_m": round(simulation.get("final_distance_m", 0.0) + (i * 2.5), 3),
                    "spoofing_active": spoof_active,
                    "drift_rate_mps": control_state.get("drift_rate", 0.3),
                    "spoof_offset_m": round(simulation.get("spoof_offsets", [0.0])[-1] + (i*0.2), 3) if simulation.get("spoof_offsets") else 0.0,
                    "innovation_m": round(simulation.get("innovations", [0.0])[-1] + (i*0.1), 3) if simulation.get("innovations") else 0.0,
                    "innovation_gate": 0.5,
                    "estimated_error_m": round(simulation.get("tracking_errors", [0.0])[-1] + (i*0.1), 3) if simulation.get("tracking_errors") else 0.0,
                    "packet_dropped": True if packet_loss and (i % 2 == 1 or control_state.get("packet_loss_rate", 0.0) > 0.2) else False,
                    "spoofing_detected": True if control_state.get("use_ekf", True) and spoof_active else False,
                    "jammed": simulation.get("success", False),
                    "ekf_success_rate": 100.0 if simulation.get("success", False) else 0.0,
                    "ekf_mean_miss_distance_m": round(simulation.get("final_distance_m", 0.0) + (i*0.4), 3),
                })
            return pd.DataFrame(rows)'''

content = content.replace(old_fallback, new_fallback)

with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
