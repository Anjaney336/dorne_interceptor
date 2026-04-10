import os
import sys
import numpy as np
from pathlib import Path

os.chdir(r'C:\Users\hp\Downloads\dorne_interceptor')
sys.path.insert(0, r'C:\Users\hp\Downloads\dorne_interceptor\src')
from drone_interceptor.backend.mission_service import MissionConfig, MissionController

config = MissionConfig(
    num_targets=1,
    target_speed_mps=4.0,
    interceptor_speed_mps=20.0,
    drift_rate_mps=0.0,
    noise_level_m=0.10,
    telemetry_latency_ms=0.0,
    packet_loss_rate=0.0,
    max_steps=int(20.0 / 0.05),
    dt=0.05,
    use_ekf=True,
)
controller = MissionController(config, output_dir=Path(r'C:\tmp\di'))
controller.initialize_mission()
for step in range(400):
    controller.execute_step(step)
    t = controller.targets[0]
    err = np.linalg.norm(t.estimated_position - t.position)
    if err > 1.0 or t.spoofing_active:
        print('step', step, 'err', err, 'spoof', t.spoofing_active, 'innov', t.innovation_m)
        break
print('final err', np.linalg.norm(controller.targets[0].estimated_position - controller.targets[0].position))
print('mean rmse', np.mean([f.rmse_m for f in controller.telemetry_log]))
