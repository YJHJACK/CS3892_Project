
oTTo Autonomous Vehicle Group Projects with VehicleSim

This project is built on top of VehicleSim, a lightweight and modular simulation framework for developing and testing autonomous vehicle systems.

We implemented a basic autonomous driving pipeline inside the example_project folder, which includes path planning, localization, and perception. The path planning module generates a drivable trajectory using BFS to find a route through the simulated road network. The localization module uses an Extended Kalman Filter (EKF) to estimate the vehicle’s position and velocity by fusing noisy GPS and IMU data. The perception module detects and tracks nearby vehicles using camera data and a particle filter for robust pose estimation.

Testing scripts are provided in the test directory, including evaluation setups for both localization and perception. These tests help validate the accuracy and stability of each module independently.

Note: While the algorithms for localization and perception have been verified to work correctly, we observed issues related to channel communication during integration. Specifically, after disabling the ground truth data, the system sometimes fails to retrieve valid sensor input due to OS-level synchronization delays. These problems are not due to flaws in the algorithms themselves, but rather from lower-level communication timing.
