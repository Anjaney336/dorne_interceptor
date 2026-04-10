import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Update _build_day8_target_results_frame to use live backend mathematical tracking payloads natively
old_func_body = r'''    if len(replay.frames) == 0:
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

new_func_body = r'''    if len(replay.frames) == 0:
        if simulation is not None and len(simulation.get("times", [])) > 0:
            control_state = st.session_state.get("live_control_state", {})
            num_targets = control_state.get("num_targets", 3)
            packet_loss = control_state.get("packet_loss_rate", 0.05) > 0.0
            spoof_active = control_state.get("noise_level", 0.45) > 0.2
            rows = []
            
            # Extract live targets from backend snapshot if present
            backend_live_targets = {}
            live_snapshot = st.session_state.get("backend_state", {}).get("snapshot", {})
            if "targets" in live_snapshot:
                for t_entry in live_snapshot["targets"]:
                    backend_live_targets[t_entry["target"]] = t_entry
                    
            # Fetch backend per-target validation if available
            backend_target_metrics = {}
            if validation_report and isinstance(validation_report, dict) and "per_target_summary" in validation_report:
                for idx, t in enumerate(validation_report["per_target_summary"]):
                    backend_target_metrics[idx] = t
            elif "backend_validation" in st.session_state and isinstance(st.session_state["backend_validation"], dict) and "per_target_summary" in st.session_state["backend_validation"]:
                for idx, t in enumerate(st.session_state["backend_validation"]["per_target_summary"]):
                    backend_target_metrics[idx] = t

            for i in range(num_targets):
                target_name = backend_target_metrics.get(i, {}).get("target", f"Target_{i+1}")
                ekf_success = 100.0 if simulation.get("success", False) else 0.0
                ekf_miss = round(simulation.get("final_distance_m", 0.0) + (i*0.4), 3)
                
                # Default logic bridging backend gaps
                status = "INTERCEPTED" if ekf_success > 50.0 else "ACTIVE"
                threat = round(0.85 - (i * 0.05), 3)
                dist = round(simulation.get("final_distance_m", 0.0) + (i * 2.5), 3)
                drift_m = control_state.get("drift_rate", 0.3)
                spoof_off = round(simulation.get("spoof_offsets", [0.0])[-1] + (i*0.2), 3) if simulation.get("spoof_offsets") else 0.0
                pack_drop = True if packet_loss and (i % 2 == 1 or control_state.get("packet_loss_rate", 0.0) > 0.2) else False
                
                if target_name in backend_live_targets:
                    t_live = backend_live_targets[target_name]
                    dist = round(t_live.get("distance_m", dist), 3)
                    threat = round(t_live.get("threat_level", threat), 4)
                    spoof_active = t_live.get("spoofing_active", spoof_active)
                    drift_m = round(t_live.get("drift_rate_mps", drift_m), 3)
                    spoof_off = round(t_live.get("spoof_offset_m", spoof_off), 3)
                    pack_drop = t_live.get("packet_dropped", pack_drop)

                # Overwrite with precise backend validation if available
                if i in backend_target_metrics:
                    ekf_success = round(backend_target_metrics[i].get("ekf_success_rate", 0) * 100, 1)
                    ekf_miss = round(backend_target_metrics[i].get("ekf_mean_miss_distance_m", 0), 3)
                    status = "INTERCEPTED" if ekf_success > 50.0 else "ACTIVE"
                
                rows.append({
                    "target": target_name,
                    "status": status,
                    "threat_level": threat,
                    "distance_m": dist,
                    "spoofing_active": spoof_active,
                    "drift_rate_mps": drift_m,
                    "spoof_offset_m": spoof_off,
                    "innovation_m": round(simulation.get("innovations", [0.0])[-1] + (i*0.1), 3) if simulation.get("innovations") else 0.0,
                    "innovation_gate": 0.5,
                    "estimated_error_m": round(simulation.get("tracking_errors", [0.0])[-1] + (i*0.1), 3) if simulation.get("tracking_errors") else 0.0,
                    "packet_dropped": pack_drop,
                    "spoofing_detected": True if control_state.get("use_ekf", True) and spoof_active else False,
                    "jammed": ekf_success > 50.0,
                    "ekf_success_rate": ekf_success,
                    "ekf_mean_miss_distance_m": ekf_miss,
                })
            return pd.DataFrame(rows)
        return pd.DataFrame('''
content = content.replace(old_func_body, new_func_body)

with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Success")
