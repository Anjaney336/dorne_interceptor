# Architecture Notes

The scaffold follows a standard autonomy pipeline:

1. Simulation advances the world state.
2. Perception converts observations into detections.
3. Tracking estimates the target state.
4. Prediction rolls the target state forward over a short horizon.
5. Planning selects an intercept waypoint.
6. Control generates velocity commands for the interceptor.

Each subsystem is isolated behind a simple class interface so researchers can replace placeholder models with learned or model-based implementations without restructuring the project.

