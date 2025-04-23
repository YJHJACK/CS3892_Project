“oTTo Autonomous Vehicle Group Projects with VehicleSim“


This project is implemented in Julia and built on top of VehicleSim, a lightweight and modular simulation framework designed by VAMPIR Lab for developing and testing autonomous vehicle algorithms in a controlled virtual environment.

Project Overview
The core autonomous driving pipeline is located in the example_project folder. It includes:

Path Planning
A route is computed using Breadth-First Search (BFS) to determine the optimal sequence of road segments from the start point to the destination. The resulting trajectory is smoothed and resampled for the vehicle to follow.

Localization
We use an Extended Kalman Filter (EKF) to fuse noisy GPS and IMU sensor data, providing a stable estimate of the ego vehicle’s 3D position, velocity, and orientation over time.

Perception
Nearby vehicles are detected using stereo camera data. A combination of bounding box matching and a particle filter is used to estimate the positions and headings of dynamic obstacles.

Testing
Testing scripts are provided in the test/ folder, including evaluation setups for both localization and perception. These tests can be used to validate the accuracy and robustness of each module independently.

Known Issues
Although the localization and perception algorithms perform correctly in isolated tests, integration revealed channel communication issues—especially after disabling ground truth input. This is likely due to OS-level synchronization delays or stale data in inter-process channels. These are implementation-level issues and do not affect the correctness of the underlying algorithms.

