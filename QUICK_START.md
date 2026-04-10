# Quick Start Guide: Drone Interceptor Mission Backend

## Starting the Backend Server

### 1. Activate Virtual Environment
```bash
cd c:\Users\hp\Downloads\dorne_interceptor
.\.venv\Scripts\activate
```

### 2. Start FastAPI Telemetry Server
```bash
python scripts/run_day8_telemetry_server.py
```

Or explicitly:
```bash
python -m uvicorn drone_interceptor.simulation.telemetry_api:app --host 127.0.0.1 --port 8765 --reload
```

The API will be available at: **http://127.0.0.1:8765**

### 3. Start Streamlit Dashboard (in another terminal)
```bash
cd c:\Users\hp\Downloads\dorne_interceptor
.venv\Scripts\streamlit run src/drone_interceptor/dashboard/app.py
```

Dashboard will be available at: **http://127.0.0.1:8501**

---

## Using the New Mission Endpoints

### Option 1: Using cURL (for testing)

#### Start a Mission
```bash
curl -X POST http://127.0.0.1:8765/mission/start/v2 \
  -H "Content-Type: application/json" \
  -d '{
    "num_targets": 3,
    "target_speed_mps": 6.0,
    "interceptor_speed_mps": 20.0,
    "drift_rate_mps": 0.3,
    "noise_level_m": 0.45,
    "telemetry_latency_ms": 80.0,
    "packet_loss_rate": 0.05,
    "guidance_gain": 6.0,
    "kill_radius_m": 10.26,
    "use_ekf": true,
    "use_anti_spoofing": true,
    "random_seed": 42
  }'
```

Returns:
```json
{
  "status": "started",
  "run_id": "run_a1b2c3d4e5f6",
  "mission_config": { ... }
}
```

#### Check Mission Status
```bash
curl http://127.0.0.1:8765/mission/run_a1b2c3d4e5f6/status
```

#### Get Mission Artifacts
```bash
curl http://127.0.0.1:8765/mission/run_a1b2c3d4e5f6/artifacts
```

---

### Option 2: Using Python Requests

```python
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8765"

# Start mission
payload = {
    "num_targets": 3,
    "target_speed_mps": 6.0,
    "interceptor_speed_mps": 20.0,
    "drift_rate_mps": 0.3,
    "noise_level_m": 0.45,
    "guide_gain": 6.0,
    "use_ekf": True,
    "use_anti_spoofing": True,
}

response = requests.post(f"{BASE_URL}/mission/start/v2", json=payload)
result = response.json()
run_id = result["run_id"]

print(f"Mission started: {run_id}")
print(f"Initial status: {result['status']}")

# Poll for completion
while True:
    status = requests.get(f"{BASE_URL}/mission/{run_id}/status").json()
    print(f"Mission status: {status['status']}")
    
    if status["status"] in ["complete", "failed"]:
        break
    
    time.sleep(2)  # Check every 2 seconds

# Get artifacts when complete
artifacts = requests.get(f"{BASE_URL}/mission/{run_id}/artifacts").json()
print(f"Artifacts: {json.dumps(artifacts, indent=2)}")

# Download CSV
csv_path = artifacts["artifacts"]["telemetry_csv"]["path"]
print(f"CSV telemetry: {csv_path}")

# Download MP4
mp4_path = artifacts["artifacts"]["fpv_video_mp4"]["path"]
print(f"FPV video: {mp4_path}")
```

---

### Option 3: Using the Streamlit Dashboard

1. **Open** http://127.0.0.1:8501
2. **Adjust Parameters** in the left sidebar:
   - Target Speed (m/s)
   - Drift Rate (m/s)
   - Noise Level (m)
   - Other mission parameters
3. **Click** "Run Live Simulation" button
4. **Monitor** real-time telemetry in the HUD
5. **View** artifacts section after completion
   - Download CSV for analysis
   - Play MP4 video in embedded player

---

## Parameter Guide

### Key Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `num_targets` | 1-10 | 3 | Number of targets to engage |
| `target_speed_mps` | 2-12 | 6 | Target velocity (m/s) |
| `interceptor_speed_mps` | 15-35 | 20 | Interceptor velocity (m/s) |
| `drift_rate_mps` | 0-1.0 | 0.3 | Constant GPS drift (m/s) |
| `noise_level_m` | 0-2.0 | 0.45 | Gaussian measurement noise (m) |
| `telemetry_latency_ms` | 0-500 | 80 | Network delay (ms) |
| `packet_loss_rate` | 0-1.0 | 0.05 | Packet loss probability |
| `guidance_gain` | 4-8 | 6.0 | PN constant (higher = faster turn) |
| `kill_radius_m` | 1-20 | 10.26 | Intercept success distance (m) |
| `use_ekf` | true/false | true | Enable Kalman filter |
| `use_anti_spoofing` | true/false | true | Enable spoofing detection |

### Common Scenarios

#### Conservative (Stable)
```json
{
  "drift_rate_mps": 0.2,
  "noise_level_m": 0.2,
  "packet_loss_rate": 0.0,
  "guidance_gain": 4.0
}
```

#### Nominal (Balanced)
```json
{
  "drift_rate_mps": 0.3,
  "noise_level_m": 0.45,
  "packet_loss_rate": 0.05,
  "guidance_gain": 6.0
}
```

#### Aggressive (High Stress)
```json
{
  "drift_rate_mps": 0.5,
  "noise_level_m": 1.0,
  "packet_loss_rate": 0.1,
  "guidance_gain": 7.0,
  "telemetry_latency_ms": 150
}
```

---

## Monitoring the Mission

### Real-time Metrics (from Dashboard HUD)
- **RMSE**: Root mean square error of EKF estimates (m)
- **Kill Probability**: Likelihood of success based on current distance
- **Active Stage**: Detection → Tracking → Interception → Complete
- **Target Status**: TRACKING / SPOOFED / JAMMED
- **Closing Velocity**: Rate of approach to target (m/s)
- **Detection FPS**: Real-time sensor update rate

### Post-Mission Analysis

#### CSV Fields Available
- `step`: Time step index
- `time_s`: Elapsed time (seconds)
- `stage`: Mission phase (Detection, Tracking, Interception, Complete)
- `rmse_m`: Estimation error
- `interceptor_*`: Interceptor position and velocity
- `Target_*_pos_*`: True target positions
- `Target_*_est_*`: EKF estimated positions
- `Target_*_meas_*`: Drifted measurements
- `Target_*_spoofed`: Boolean - was spoofing detected?
- `Target_*_jammed`: Boolean - was target taken out?

#### Video Content (MP4)
- Interceptor FPV perspective
- Overlay: telemetry, stage, target status
- 20 FPS playback
- ~500 KB per 20-second mission

---

## Troubleshooting

### Issue: "Connection refused" when calling endpoints
**Solution**: Verify FastAPI server is running
```bash
python -m uvicorn drone_interceptor.simulation.telemetry_api:app --host 127.0.0.1 --port 8765
```

### Issue: Mission runs but no artifacts are generated
**Solution**: Check `/outputs` directory permissions
```bash
ls -la c:\Users\hp\Downloads\dorne_interceptor\outputs
```

### Issue: High RMSE (> 1.0 meter)
**Possible causes**:
- Drift rate too high (try 0.3 instead of 0.5+)
- Noise level too high (try 0.45 instead of 1.0+)
- Guidance gain too low (try 6.0 instead of 4.0)
- EKF disabled (ensure `use_ekf=true`)

**Solution**: Adjust parameters and re-run

### Issue: Targets not being intercepted
**Possible causes**:
- Kill radius too small (default 10.26 m is fine)
- Interceptor speed too low (should be > target_speed_mps)
- Guidance gain too low (try increasing)

**Solution**: 
- Increase `guidance_gain` to 7.0-8.0
- Decrease `kill_radius_m` to 5.0
- Run with `use_anti_spoofing=false` if spoofing is active

---

## Advanced: Running Test Suite

To validate the implementation:

```bash
cd c:\Users\hp\Downloads\dorne_interceptor
.\.venv\Scripts\python.exe test_mission_backend.py
```

Expected output:
```
============================================================
Testing Drone Interceptor Mission Backend Implementation
============================================================

1. Testing MissionConfig...
✓ MissionConfig tests passed

2. Testing ProportionalNavigation...
✓ ProportionalNavigation test passed

...

✓ ALL TESTS PASSED
```

---

## File Locations

- **API Server**: `src/drone_interceptor/simulation/telemetry_api.py`
- **Mission Service**: `src/drone_interceptor/backend/mission_service.py`
- **Dashboard**: `src/drone_interceptor/dashboard/app.py`
- **Telemetry Storage**: `outputs/mission_telemetry_*.csv`
- **Video Storage**: `outputs/mission_fpv_*.mp4`
- **Run Metadata**: `outputs/run_registry/run_*.json`

---

## API Documentation

Full OpenAPI/Swagger documentation available at:
```
http://127.0.0.1:8765/docs
```

---

## Tips for Best Results

1. **Use EKF with Anti-Spoofing Enabled**
   - Provides ~2-3x better RMSE
   - Automatically detects and mitigates spoofing

2. **Tune Guidance Gain per Scenario**
   - Conservative: 4.0-5.0 (smooth, stable)
   - Nominal: 6.0-6.5 (balanced)
   - Aggressive: 7.0-8.0 (fast convergence)

3. **Validate with CSV + Video**
   - CSV for quantitative analysis (RMSE, distances)
   - Video for qualitative inspection (visual verification)

4. **Batch Multiple Runs**
   - Can run multiple missions in parallel
   - Each gets unique `run_id`
   - All artifacts stored separately

5. **Monitor Dashboard HUD**
   - Real-time RMSE tells you EKF health
   - Closing velocity should be > 0
   - Kill probability should approach 1.0 at intercept

---

## Support & Documentation

- **Detailed Implementation**: See `BACKEND_IMPLEMENTATION.md`
- **Architecture Overview**: See `IMPLEMENTATION_SUMMARY.md`
- **Test Examples**: See `test_mission_backend.py`

---

**Version**: 1.0  
**Last Updated**: April 7, 2026  
**Status**: Ready for Production
