import asyncio
from drone_interceptor.simulation.telemetry_api import MISSION_STATE

async def test():
    print("Testing backend _run_mission via async...")
    payload = {
        "num_targets": 3,
        "interceptor_speed_mps": 28.0,
        "drift_rate_mps": 0.3,
        "noise_std_m": 0.45,
        "latency_ms": 0.0,
        "packet_loss_rate": 0.0,
        "random_seed": 61,
        "max_steps": 20,
        "dt": 0.1,
        "kill_radius_m": 1.0,
        "guidance_gain": 4.2,
        "use_ekf": True,
        "use_ekf_anti_spoofing": True
    }
    await MISSION_STATE.set_snapshot({"status": "preparing"})
    await MISSION_STATE._run_mission(1, payload)
    
    final_payload = await MISSION_STATE.get_payload()
    print("Final status:", final_payload["snapshot"].get("status"))

if __name__ == "__main__":
    asyncio.run(test())
