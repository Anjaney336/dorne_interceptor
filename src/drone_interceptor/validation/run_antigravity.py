import os
from pathlib import Path
import matplotlib.pyplot as plt
from drone_interceptor.simulation.airsim_manager import AirSimMissionManager
from drone_interceptor.simulation.antigravity import AntigravityMissionManager

def run_validation_suite(iterations: int = 10):
    print(f"Running Normal Physics Validation ({iterations} iterations)...")
    normal_manager = AirSimMissionManager(connect=False)
    normal_results = normal_manager.run_monte_carlo_validation(iterations=iterations, num_targets=3, use_multiprocessing=True)
    
    print(f"Running Antigravity Physics Validation ({iterations} iterations)...")
    antigravity_manager = AntigravityMissionManager(connect=False)
    antigrav_results = antigravity_manager.run_monte_carlo_validation(iterations=iterations, num_targets=3, use_multiprocessing=True)
    
    # Generate Chart
    labels = ['Normal Physics\n(Standard Drag)', 'Antigravity Physics\n(0g / Amplified Z)']
    rates = [normal_results.ekf_success_rate, antigrav_results.ekf_success_rate]
    
    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, rates, color=['#1f77b4', '#9467bd'], width=0.5)
    plt.ylim(0, 1.1)
    plt.ylabel('EKF Intercept Success Rate')
    plt.title(f'Drone Interceptor Combat Success\n(Validation Iterations: {iterations}x)')
    
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate*100:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    plt.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "success_rate_vs_scenario.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Validation complete! Plot saved to: {plot_path}")

if __name__ == "__main__":
    # The prompt explicitly asked to run 10x Validation Suite.
    run_validation_suite(iterations=10)
