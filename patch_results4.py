import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Fix _build_day8_target_results_frame so it generates realistic rows and merges with backend validation
old_func_body = r'''    if len(replay.frames) == 0:
        if "backend_validation_frame" in st.session_state and st.session_state["backend_validation_frame"]:
            df = pd.DataFrame(st.session_state["backend_validation_frame"])
            for col in ["ekf_success_rate", "ekf_mean_miss_distance_m", "status", "threat_level", "distance_m", "spoofing_active", "drift_rate_mps", "spoof_offset_m", "innovation_m", "innovation_gate", "estimated_error_m", "packet_dropped", "spoofing_detected", "jammed"]:
                if col not in df.columns:
                    df[col] = 0.0
            return df
        if simulation is not None and len(simulation.get("times", [])) > 0:
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
            return pd.DataFrame(rows)
        return pd.DataFrame('''

new_func_body = r'''    if len(replay.frames) == 0:
        if simulation is not None and len(simulation.get("times", [])) > 0:
            control_state = st.session_state.get("live_control_state", {})
            num_targets = control_state.get("num_targets", 3)
            packet_loss = control_state.get("packet_loss_rate", 0.05) > 0.0
            spoof_active = control_state.get("noise_level", 0.45) > 0.2
            rows = []
            
            # Fetch backend per-target validation if available
            backend_target_metrics = {}
            if validation_report and isinstance(validation_report, dict) and "per_target_summary" in validation_report:
                for idx, t in enumerate(validation_report["per_target_summary"]):
                    backend_target_metrics[idx] = t
            elif "backend_validation" in st.session_state and "per_target_summary" in st.session_state["backend_validation"]:
                for idx, t in enumerate(st.session_state["backend_validation"]["per_target_summary"]):
                    backend_target_metrics[idx] = t

            for i in range(num_targets):
                target_name = backend_target_metrics.get(i, {}).get("target", f"Target_{i+1}")
                ekf_success = 100.0 if simulation.get("success", False) else 0.0
                ekf_miss = round(simulation.get("final_distance_m", 0.0) + (i*0.4), 3)
                
                # Overwrite with precise backend validation if available
                if i in backend_target_metrics:
                    ekf_success = round(backend_target_metrics[i].get("ekf_success_rate", 0) * 100, 1)
                    ekf_miss = round(backend_target_metrics[i].get("ekf_mean_miss_distance_m", 0), 3)
                
                rows.append({
                    "target": target_name,
                    "status": "INTERCEPTED" if ekf_success > 50.0 else "ACTIVE",
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
                    "jammed": ekf_success > 50.0,
                    "ekf_success_rate": ekf_success,
                    "ekf_mean_miss_distance_m": ekf_miss,
                })
            return pd.DataFrame(rows)
        return pd.DataFrame('''
content = content.replace(old_func_body, new_func_body)

# Fix _render_backend_status_panel to add ALL targets
old_ekf = r'''            <div id="svc-ekf" class="stage-card {'status-green' if snapshot.get('ekf_lock', False) else 'status-red'}"><strong>EKF Lock</strong><br/>{'ACTIVE' if snapshot.get('ekf_lock', False) else 'OFF'}<br/>Target: {snapshot.get('active_target', 'n/a')}</div>'''

new_ekf = r'''            <div id="svc-targets" class="stage-card" style="grid-column: span 2; box-shadow: 0 0 14px rgba(0, 229, 255, 0.40);">
              <strong>Active Threat Matrix</strong><br/>
              {', '.join(snapshot.get("validation", {}).get("threat_order", [])) if snapshot.get("validation", {}).get("threat_order") else (', '.join(validation.get("threat_order", [])) if validation and validation.get("threat_order") else 'Scanning Space...')}
            </div>'''
            
content = content.replace(old_ekf, new_ekf)


with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Success")
