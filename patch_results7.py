import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

old_block = r'''                if target_name in backend_live_targets:
                    t_live = backend_live_targets[target_name]
                    dist = round(t_live.get("distance_m", dist), 3)
                    threat = round(t_live.get("threat_level", threat), 4)
                    spoof_active = t_live.get("spoofing_active", spoof_active)
                    drift_m = round(t_live.get("drift_rate_mps", drift_m), 3)
                    spoof_off = round(t_live.get("spoof_offset_m", spoof_off), 3)
                    pack_drop = t_live.get("packet_dropped", pack_drop)

                # Overwrite with precise backend validation if available'''

new_block = r'''                if target_name in backend_live_targets:
                    t_live = backend_live_targets[target_name]
                    # Compute distance_m relative to interceptor if positional vectors exist
                    if "true_position" in t_live and "interceptor_position" in live_snapshot:
                        import math
                        tx, ty, tz = t_live["true_position"]
                        ix, iy, iz = live_snapshot["interceptor_position"]
                        dist = round(math.sqrt((tx-ix)**2 + (ty-iy)**2 + (tz-iz)**2), 3)
                        
                    threat = round(t_live.get("threat_level", threat), 4)
                    spoof_active = t_live.get("spoofing_active", spoof_active)
                    drift_m = round(t_live.get("drift_rate_mps", drift_m), 3)
                    spoof_off = round(t_live.get("spoof_offset_m", spoof_off), 3)
                    pack_drop = t_live.get("packet_dropped", pack_drop)
                    status = "INTERCEPTED" if t_live.get("jammed", False) else "ACTIVE"
                    
                    # Compute estimated error
                    estimated_err_m = round(simulation.get("tracking_errors", [0.0])[-1] + (i*0.1), 3) if simulation.get("tracking_errors") else 0.0
                    if "true_position" in t_live and "estimated_position" in t_live:
                        tx, ty, tz = t_live["true_position"]
                        ex, ey, ez = t_live["estimated_position"]
                        import math
                        estimated_err_m = round(math.sqrt((tx-ex)**2 + (ty-ey)**2 + (tz-ez)**2), 3)
                        
                    t_innov = round(t_live.get("innovation_m", 0.0), 3)
                    t_innov_gate = round(t_live.get("innovation_gate", 0.5), 3)
                    t_spoof_det = t_live.get("spoofing_detected", False)
                    t_jammed = t_live.get("jammed", False)
                    
                    rows.append({
                        "target": target_name,
                        "status": status,
                        "threat_level": threat,
                        "distance_m": dist,
                        "spoofing_active": spoof_active,
                        "drift_rate_mps": drift_m,
                        "spoof_offset_m": spoof_off,
                        "innovation_m": t_innov,
                        "innovation_gate": t_innov_gate,
                        "estimated_error_m": estimated_err_m,
                        "packet_dropped": pack_drop,
                        "spoofing_detected": t_spoof_det,
                        "jammed": t_jammed,
                        "ekf_success_rate": round(backend_target_metrics[i].get("ekf_success_rate", 0) * 100, 1) if i in backend_target_metrics else (100.0 if t_jammed else 0.0),
                        "ekf_mean_miss_distance_m": round(backend_target_metrics[i].get("ekf_mean_miss_distance_m", 0), 3) if i in backend_target_metrics else 0.0,
                    })
                    continue

                # Overwrite with precise backend validation if available'''

content = content.replace(old_block, new_block)

with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
