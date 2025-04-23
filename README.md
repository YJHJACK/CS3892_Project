# oTTo Autonomous Vehicle Group Project with VehicleSim

## Platform

This project is implemented in **Julia** and built on top of [VehicleSim](https://github.com/VAMPIR-Lab/VehicleSim), a lightweight and modular simulation framework developed by VAMPIR Lab for testing autonomous vehicle algorithms in a controlled virtual environment.

---

## Project Overview

The core autonomous driving pipeline is located in the `example_project/` folder and includes the following modules:

### Path Planning

We use **Breadth-First Search (BFS)** to compute an optimal sequence of road segments from a given start location to a designated destination.  
The resulting path is then smoothed and resampled into a sequence of waypoints for trajectory tracking.

### Localization

An **Extended Kalman Filter (EKF)** is implemented to fuse noisy GPS and IMU measurements. It produces a stable estimate of the ego vehicle’s 3D position, linear velocity, and orientation (in quaternion form).  
This module alternates between:


### Perception

Dynamic obstacles (e.g., other vehicles) are detected using stereo camera data. A hybrid approach combining:

- **Bounding box matching**, and  
- **Particle filtering**

is used to estimate the 2D positions and headings of nearby vehicles.

### Testing

Testing scripts are provided under the `test/` folder. They include:

- Localization accuracy evaluation against recorded ground truth  
- Perception accuracy evaluation against recorded ground truth

These tests validate module functionality in isolation and can be extended for integration-level validation.

---

## Known Issues

While both the localization and perception modules work correctly under controlled conditions and standalone tests, integration testing has revealed potential communication issues across channels.  
These appear particularly when ground truth is disabled and likely stem from:

- OS-level thread scheduling  
- Channel delays or stale data

These problems are implementation-level and do not affect the algorithmic correctness of the EKF or particle filter logic.  
We are actively refining the data synchronization strategy to ensure consistent inter-module communication.



