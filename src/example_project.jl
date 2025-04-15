using DataStructures
using StaticArrays
using Interpolations
using PyCall
using VehicleSim
using Sokets
using LinearAlgebra

struct PIDController
    kp::Float64
    ki::Float64
    kd::Float64
    integral::Float64
    last_error::Float64
end

struct MyLocalizationType
    valid::Bool                         # indicates if the estimate is valid
    position::SVector{3,Float64}        # x, y, z coordinates
    yaw::Float64                        # heading angle
    velocity::SVector{3,Float64}        # velocity vector
    covariance::Matrix{Float64}         # uncertainty matrix
    timestamp::Float64                  # time of the estimate
end

export MyLocalizationType

struct Detected_Obj
    id::int  # id for each object
    bbox::NTuple{4, Float64}           # Bounding box: (x_min, y_min, x_max, y_max).
    confidence::Float64                # Confidence score from the CNN.
    classification::String
    position::SVector{2, Float64}      # Estimated 2D position (from EKF fusion).
    velocity::SVector{2, Float64}      # Estimated 2D velocity (if available).
end

struct Particle
    angle::Float64            # angle
    loc::SVector{2,Float64}   # pos
    v::Float64                # velocity
    w::Float64                # particle weight
end

struct MyPerceptionType
    timestamp::Int
    Detected_Obj::Float64
end

mutable struct ObjectEKF
    x::Vector{Float64}      # State: [x, y, vx, vy]
    P::Matrix{Float64}      # Covariance
    Q::Matrix{Float64}      # Process noise
    R::Matrix{Float64}      # Measurement noise
end

# Update function: updates PID
function PID_update!(pid::PIDController, error, dt)
    pid.integral += error * dt
    derivative = (error - pid.last_error) / dt
    pid.last_error = error
    return pid.kp * error + pid.ki * pid.integral + pid.kd * derivative
end

# Helper function: prevents collision
function collision_avoidance_control(ego_state, perception_state, planned_velocity)
    # Define a safety distance (in meters)
    safety_distance = 8.0
    
    # Initialize the adjusted velocity with the planned value
    adjusted_velocity = planned_velocity
    
    # Assume perception_state.obstacles is a vector of obstacles, each with a 'pos' field.
    # Here, we check only obstacles that are ahead (in the vehicle's forward direction).
    for obs in perception_state.obstacles
        # Compute the vector from ego to obstacle
        relative_vec = obs.pos - ego_state.pos
        
        # Project the relative vector on the vehicle's heading (using ego_state.yaw)
        # Compute the unit vector in the direction of ego heading:
        forward = SVector(cos(ego_state.yaw), sin(ego_state.yaw))
        # Calculate how far ahead the obstacle is:
        distance_ahead = dot(relative_vec, forward)
        
        # If the obstacle is ahead and closer than the safety distance:
        if distance_ahead > 0 && distance_ahead < safety_distance
            # Reduce the target velocity (for instance, to zero or a low speed)
            adjusted_velocity = min(adjusted_velocity, 0.0)
        end
    end
    
    return adjusted_velocity
end

# Lane generator: generates a list of lanes (least number) that the ego will pass in order from the beginning to the destination
function least_lane_path(all_segs::Dict{Int,RoadSegment}, start_id::Int, goal_id::Int)
    # Use a queue for a breadth-first search.
    queue = Queue{Int}()
    enqueue!(queue, start_id)
    
    # Dictionary to record the predecessor for each segment.
    came_from = Dict{Int,Int}()
    
    # Set to track visited segments.
    visited = Set{Int}([start_id])
    
    while !isempty(queue)
        current = dequeue!(queue)
        if current == goal_id
            # Reconstruct the path by walking backwards.
            path = [current]
            while haskey(came_from, current)
                current = came_from[current]
                push!(path, current)
            end
            return reverse(path)
        end
        
        # For each neighbor reachable from this segment.
        for neighbor in all_segs[current].children
            if neighbor ∉ visited
                push!(visited, neighbor)
                came_from[neighbor] = current
                enqueue!(queue, neighbor)
            end
        end
    end
    error("No path found from segment $start_id to segment $goal_id")
end

function decision_making(localization_state_channel, perception_state_channel, map, target_road_segment_id, socket)
    # Setup a PID controller for steering.
    pid = PIDController(2.0, 0.1, 0.5, 0.0, 0.0)
    dt = 0.1  # time step in seconds

    # For simplicity, we assume the current segment is known.
    current_segment_id = 1

    # Compute the route and generate a reference trajectory.
    route = least_lane_path(map, current_segment_id, target_road_segment_id)
    route_segments = [map[id] for id in route]
    trajectory = generate_trajectory_plan_with_curvature(route_segments, 200)

    while true
        # Get the latest localization state.
        loc = fetch(localization_state_channel)
        # If the localization estimate is not valid, wait.
        if !loc.valid
            sleep(dt)
            continue
        end
        
        # Extract the current 2D position from the 3D position.
        current_pos = loc.position[1:2]
        
        # Get the latest perception state.
        perc = fetch(perception_state_channel)
        
        # Choose a lookahead point on the trajectory.
        lookahead_idx = min(length(trajectory), 10)
        desired_point = trajectory[lookahead_idx]
        
        # Compute error vector (only in 2D).
        error_vec = desired_point - current_pos
        distance_error = norm(error_vec)
        desired_heading = atan(error_vec[2], error_vec[1])
        heading_error = desired_heading - loc.yaw
        
        # Compute the steering command using the PID controller.
        steering_command = update!(pid, heading_error, dt)
        
        # Compute a planned target velocity (for instance, proportional to the distance error).
        planned_velocity = 1.0 * distance_error
        
        # Adjust the planned velocity based on collision avoidance.
        safe_velocity = collision_avoidance_control(loc, perc, planned_velocity)
        
        # Form the control command (steering, velocity, flag).
        cmd = (steering_command, safe_velocity, true)
        serialize(socket, cmd)
        
        sleep(dt)
    end
end

function process_gt(
        gt_channel,
        shutdown_channel,
        localization_state_channel,
        perception_state_channel)

    while true
        fetch(shutdown_channel) && break

        fresh_gt_meas = []
        while isready(gt_channel)
            meas = take!(gt_channel)
            push!(fresh_gt_meas, meas)
        end

        # process the fresh gt_measurements to produce localization_state and
        # perception_state
        
        take!(localization_state_channel)
        put!(localization_state_channel, new_localization_state_from_gt)
        
        take!(perception_state_channel)
        put!(perception_state_channel, new_perception_state_from_gt)
    end
end

# -------------------------
# Helper Functions
# -------------------------
function normalize_angle(angle::Float64)
    while angle > π
        angle -= 2π
    end
    while angle < -π
        angle += 2π
    end
    angle
end

# EKF Prediction: predict state and covariance after Δt
function ekf_predict(x::Vector{Float64}, P::Matrix{Float64}, Δt::Float64, Q::Matrix{Float64})
    x_pred = f(x, Δt)               # Process model function
    F = Jac_x_f(x, Δt)              # Jacobian of process model
    P_pred = F * P * F' + Q         # Covariance prediction
    return x_pred, P_pred
end

# EKF Update: update state using measurement z
function ekf_update(x::Vector{Float64}, P::Matrix{Float64}, z::Vector{Float64},
                    h::Function, H::Function, R::Matrix{Float64})
    z_pred = h(x)                   # Predicted measurement
    y = z - z_pred                  # Measurement residual
    if length(y) >= 3
        y[3] = normalize_angle(y[3])
    end
    Hx = H(x)                       # Jacobian of measurement model
    S = Hx * P * Hx' + R            # Residual covariance
    K = P * Hx' * inv(S)            # Kalman gain
    x_new = x + K * y               # State update
    P_new = (I - K * Hx) * P        # Covariance update
    return x_new, P_new
end

# -------------------------
# Main Localization Loop (EKF)
# -------------------------
"""
    localization(meas_channel, state_vec_channel; dt_default=0.1)

Main loop:
- Reads measurements from meas_channel,
- Predicts state using Δt and performs EKF updates for each GPS measurement,
- Sends updated state vector **and updated covariance matrix** as a tuple to state_vec_channel.
"""
function localization(meas_channel::Channel{MeasurementMessage},
                      state_vec_channel::Channel{Tuple{Vector{Float64}, Matrix{Float64}}};
                      dt_default=0.1)
    # State order: [position (3); quaternion (4); velocity (3); angular velocity (3)]
    x_est = vcat([0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0])
    # Initial covariance: different uncertainty for each part
    P_est = Diagonal([1.0, 1.0, 1.0,    # Position
                      0.1, 0.1, 0.1, 0.1, # Quaternion
                      0.5, 0.5, 0.5,      # Velocity
                      0.1, 0.1, 0.1])     # Angular velocity
    
    # Process noise covariance Q
    Q = Diagonal([0.3, 0.3, 0.3,           # Position noise
                  0.005, 0.005, 0.005, 0.005,  # Quaternion noise
                  0.2, 0.2, 0.2,           # Velocity noise
                  0.02, 0.02, 0.02])        # Angular velocity noise
    
    # GPS measurement noise covariance R
    R_gps = Diagonal([0.5, 0.5, 0.05])
    
    last_time = time()
    
    while true
        # Get message from measurement channel
        meas_msg = take!(meas_channel)
        meas_time = isempty(meas_msg.measurements) ? time() : meas_msg.measurements[1].time
        Δt = meas_time - last_time
        if Δt <= 0
            Δt = dt_default
        end
        last_time = meas_time
        
        # Prediction step
        x_pred, P_pred = ekf_predict(x_est, P_est, Δt, Q)
        
        x_upd = x_pred
        P_upd = P_pred
        # Update for each GPS measurement
        for m in meas_msg.measurements
            if m isa GPSMeasurement
                z = [m.lat, m.long, m.heading]
                x_upd, P_upd = ekf_update(x_upd, P_upd, z, h_gps, Jac_h_gps, R_gps)
            end
        end
        
        # Update state estimate and covariance
        x_est = x_upd
        P_est = P_upd
        
        put!(state_vec_channel, (x_est, P_est))
        sleep(0.001)
    end
end

# -------------------------
# Helper: State Conversion
# -------------------------
"""
    convert_state(x, P, t) -> MyLocalizationType

Converts the state vector x (length 13) to MyLocalizationType,
extracts position, yaw, velocity and adds timestamp t.
Assumes extract_yaw_from_quaternion is defined.
"""
function convert_state(x::Vector{Float64}, P::Matrix{Float64}, t::Float64)::MyLocalizationType
    pos = SVector{3,Float64}(x[1:3])
    yaw = extract_yaw_from_quaternion(x[4:7])
    vel = SVector{3,Float64}(x[8:10])
    return MyLocalizationType(true, pos, yaw, vel, P, t)
end

# -------------------------
# Wrapper: localize
# -------------------------
"""
    localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel; dt_default=0.1)

Wrapper function:
- Reads measurements from gps_channel and imu_channel to create MeasurementMessage,
- Sends the message to an internal meas_channel,
- Calls the main localization loop,
- Converts the output state vector **and covariance matrix** to MyLocalizationType and sends it to localization_state_channel,
- Monitors shutdown_channel to exit.
"""
function localize(
    gps_channel::Channel{GPSMeasurement},
    imu_channel::Channel{IMUMeasurement},
    localization_state_channel::Channel{MyLocalizationType},
    shutdown_channel::Channel{Bool};
    dt_default=0.1
)
    meas_channel = Channel{MeasurementMessage}(32)
    state_vec_channel = Channel{Tuple{Vector{Float64}, Matrix{Float64}}}(32)
    @async localization(meas_channel, state_vec_channel; dt_default=dt_default)
    
    while true
        if isready(shutdown_channel) && take!(shutdown_channel)
            close(meas_channel)
            close(state_vec_channel)
            break
        end
        
        # Collect GPS measurements (extend for IMU if needed)
        measurements = GPSMeasurement[]
        while isready(gps_channel)
            push!(measurements, take!(gps_channel))
        end
        
        if !isempty(measurements)
            # Create MeasurementMessage (using vehicle_id=1, target_segment=0 as an example)
            meas_msg = MeasurementMessage(1, 0, measurements)
            put!(meas_channel, meas_msg)
        end
        
        # If a new state vector is available, convert and output it
        if isready(state_vec_channel)
            x_vec, P_vec = take!(state_vec_channel)
            state = convert_state(x_vec, P_vec, time())
            put!(localization_state_channel, state)
        end
        
        sleep(0.005)
    end
end

#given quaternion, location, bboxes from 2 cams, return estimated location region
function estimate_location_from_2_bboxes(ego_quaternion::SVector{4,Float64},
                                  ego_position::SVector{3,Float64},
                                  T_body_camrot1, T_body_camrot2,
                                  true_bboxes_cam1::Vector{NTuple{4,Float64}},
                                  true_bboxes_cam2::Vector{NTuple{4,Float64}})
    quat_loc_bboxerror_list = []
    base_yaw = quaternion_to_yaw(ego_quaternion)
    for dyaw in range(-0.3, 0.3, length=5)   
        for dx in range(-1.0, 1.0, length=5) 
            for dy in range(-1.0, 1.0, length=5)
                # construction
                candidate_yaw = base_yaw + dyaw
                candidate_quat = yaw_to_quaternion(candidate_yaw)
                candidate_loc = SVector(ego_position[1] + dx,
                                        ego_position[2] + dy,
                                        ego_position[3])

                # 2. predict bbox on canmera besed on pose
                pred_bboxes_cam1 = predict_bboxes_from_pose(candidate_quat, candidate_loc, T_body_camrot1)
                pred_bboxes_cam2 = predict_bboxes_from_pose(candidate_quat, candidate_loc, T_body_camrot2)

                # 3. calculate errors
                cam1_error = bboxes_error(pred_bboxes_cam1, true_bboxes_cam1)
                cam2_error = bboxes_error(pred_bboxes_cam2, true_bboxes_cam2)
                total_error = cam1_error + cam2_error

                push!(quat_loc_bboxerror_list, (candidate_quat, candidate_loc, total_error))
            end
        end
    end

    # 5. find min error
    sort!(quat_loc_bboxerror_list, by = x->x[3])  # x[3] 即 total_error
    best_candidate = first(quat_loc_bboxerror_list)
    best_quat   = best_candidate[1]
    best_loc    = best_candidate[2]
    min_error   = best_candidate[3]

    margin = 1.0
    estimated_region = (best_loc[1] - margin, best_loc[2] - margin,
                        best_loc[1] + margin, best_loc[2] + margin)

    return best_quat, best_loc, min_error, estimated_region
end

function predict_bboxes(quat, loc, T_body_camrot, image_width::Int, image_height::Int, pixel_len::Float64)
    x_min = 100.0 + randn()*5
    y_min = 80.0 + randn()*5
    x_max = x_min + 50.0 + randn()*5
    y_max = y_min + 40.0 + randn()*5
    return [(x_min, y_min, x_max, y_max)]
end

function bboxes_error(pred_bboxes::Vector{NTuple{4,Float64}}, true_bboxes::Vector{NTuple{4,Float64}})
    p = pred_bboxes[1]
    t = true_bboxes[1]
    return abs(p[1]-t[1]) + abs(p[2]-t[2]) + abs(p[3]-t[3]) + abs(p[4]-t[4])
end

function initialize_particles(quat_loc_minerror_list; var_location=0.5,var_angle=pi/12,v_max=7.5,step_v=0.5,number_of_particles=1000)
    # assign particles
    avg_num = max(1, ceil(Integer, number_of_particles / length(quat_loc_minerror_list)))

    particles = Particle[]
    total_created = 0

    for (quat, loc, min_err) in quat_loc_minerror_list
        # 1. turn quaterion to yaw
        base_angle = quaternion_to_yaw(quat)

        #    mean = [ base_angle, loc[1], loc[2] ]
        #    cov  = diagm([var_angle^2, var_location^2, var_location^2])
        mu = [base_angle, loc[1], loc[2]]
        cov = Matrix{Float64}(I, 3, 3)
        cov[1,1] = var_angle^2
        cov[2,2] = var_location^2
        cov[3,3] = var_location^2
        multi_normal = MvNormal(mu, cov)

        # sample particles
        for i in 1:avg_num
            sample = rand(multi_normal)
            # sample[1] -> particle angle
            # sample[2] -> x
            # sample[3] -> y
            θ = sample[1]
            x = sample[2]
            y = sample[3]
            # random speed
            possible_v = collect(0:step_v:v_max)
            v = rand(possible_v)
            # equally assign initial weight
            w = 1.0 / number_of_particles

            push!(particles, Particle(θ, SVector(x,y), v, w))
            total_created += 1
            if total_created >= number_of_particles
                break
            end
        end

        if total_created >= number_of_particles
            break
        end
    end

    return particles
end

function update_particles(particles::Vector{Particle},
                          delta_t::Float64,
                          obj_bboxes_cam1::Vector{NTuple{4,Float64}},
                          obj_bboxes_cam2::Vector{NTuple{4,Float64}},
                          ego_orientation::Float64,
                          ego_position::SVector{2,Float64},
                          T_body_camrot1,
                          T_body_camrot2,
                          image_width::Int,
                          image_height::Int,
                          pixel_len::Float64)
    for i in 1:length(particles)
        p = particles[i]
        # update particle pos
        dx = p.v * cos(p.angle) * delta_t
        dy = p.v * sin(p.angle) * delta_t
        new_loc = p.loc + SVector(dx, dy)
        # add noise
        noise_std = 0.05
        new_loc += SVector(randn()*noise_std, randn()*noise_std)
        new_angle = p.angle + randn()*0.01
        
        # trajection
        quat_new = yaw_to_quaternion(new_angle)
        # predict bbox
        pred_bboxes_cam1 = predict_bboxes(quat_new, new_loc, T_body_camrot1, image_width, image_height, pixel_len)
        pred_bboxes_cam2 = predict_bboxes(quat_new, new_loc, T_body_camrot2, image_width, image_height, pixel_len)
        
        # calculate error
        error1 = bboxes_error(pred_bboxes_cam1, obj_bboxes_cam1)
        error2 = bboxes_error(pred_bboxes_cam2, obj_bboxes_cam2)
        total_error = error1 + error2
        
        # update likelihood
        sigma = 20.0  # pixels
        likelihood = exp(-0.5 * (total_error/sigma)^2)
        
        # update particles
        particles[i] = Particle(new_angle, new_loc, p.v, likelihood)
    end
    
    total_weight = sum(p -> p.w, particles)
    if total_weight > 0
        for i in 1:length(particles)
            p = particles[i]
            particles[i] = Particle(p.angle, p.loc, p.v, p.w / total_weight)
        end
    end

    return particles
end

function estimate_object_state(ego_state, obj_bboxes::Vector{NTuple{4,Float64}}; scale_factor=0.01)
    bbox = obj_bboxes[1]
    center_pixel = bbox_center(bbox)
    offset = SVector(center_pixel[1] * scale_factor, center_pixel[2] * scale_factor)
    
    # suppose ego_state: (ego_position, ego_orientation)
    ego_position = ego_state[1]  # SVector{3,Float64},
    ego_orientation = ego_state[2]

    # get 2D pos
    estimated_location = SVector(ego_position[1] + offset[1],
                                 ego_position[2] + offset[2],
                                 ego_position[3])
    
    # estimate target orientation
    bbox_width  = bbox[3] - bbox[1]
    bbox_height = bbox[4] - bbox[2]
    if bbox_width >= bbox_height
        estimated_orientation = ego_orientation
    else
        estimated_orientation = ego_orientation + (pi/2)
    end

    return estimated_orientation, estimated_location
end
    

#EKF perception function
function perception(cam_meas_channel, localization_state_channel, perception_state_channel, ekf, cnn_model; confidence_threshold=0.5)
    # set up stuff
    last_time = time()
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        true_bboxes_cam1 = haskey(cam_meas, :bboxes_cam1) ? cam_meas[:bboxes_cam1] : []
    true_bboxes_cam2 = haskey(cam_meas, :bboxes_cam2) ? cam_meas[:bboxes_cam2] : []

    # calculate target pos
    best_center, half_w, half_y, quat_loc_minerror_list =
        estimate_location_from_2_bboxes(ego_orientation, ego_position,
                                        T_body_camrot1, T_body_camrot2,
                                        vehicle_size, image_width, image_height, pixel_len,
                                        true_bboxes_cam1, true_bboxes_cam2;
                                        step=candidate_step)
    sorted_candidates = sort(quat_loc_minerror_list, by = x -> x[3])
    selected_candidates = sorted_candidates[1:min(10, length(sorted_candidates))]

    # initialize particles
    particles = initializa_particles(selected_candidates, vehicle_size;
                                     varangle = pi/12, var_location = 0.5, 
                                     max_v = 7.5, step_v = 0.5, number_of_particles = 1000)
    
    # estimate using bbox
    if !isempty(true_bboxes_cam1)
        obj_bboxes = [true_bboxes_cam1[1]]
        est_obj_ori, est_obj_loc = estimate_object_state(ego_state, obj_bboxes; scale_factor=pixel_len)
    else
        # if no target, default to ego
        est_obj_ori = ego_orientation
        est_obj_loc = ego_position
    end

    detected_object = DetectedObject(1, est_obj_ori, est_obj_loc, true_bboxes_cam1 != [] ? true_bboxes_cam1[1] : (0.0,0.0,0.0,0.0))
    
    # update particles
    delta_t = time() - last_time
    updated_particles = update_particles(particles, delta_t,
                                         true_bboxes_cam1, true_bboxes_cam2,
                                         ego_orientation, ego_position,
                                         T_body_camrot1, T_body_camrot2,
                                         image_width, image_height, pixel_len)

    margin = 1.0  
    estimated_region = (best_center[1]-margin, best_center[2]-margin,
                        best_center[1]+margin, best_center[2]+margin)
    perception_msg = MyPerceptionType(time(), [detected_object], estimated_region)

        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end
end

function decision_making(localization_state_channel, 
    perception_state_channel, 
    map, 
    target_road_segment_id, 
    socket)
    # Simple reactive control
    while true
        latest_localization_state = fetch(localization_state_channel)
        latest_perception_state = fetch(perception_state_channel)

        # Default commands
        steering_angle = 0.0
        target_vel = 0.0
        keep_driving = false

        # Only move if we are localized
        if latest_localization_state.field1 == 1
            # If we "see" something (basic perception cue), maybe slow down
            if latest_perception_state.field1 == 1
                target_vel = 1.0  # slow
                keep_driving = false  # stop and wait (or could swerve later)
            else
                target_vel = 5.0  # arbitrary cruising speed
                keep_driving = true
            end
        else
            @warn "Localization not ready, vehicle staying stopped"
        end

        # Dummy logic for future: use map + target_road_segment_id for planning

        cmd = (steering_angle, target_vel, keep_driving)
        serialize(socket, cmd)

        sleep(0.05)  # Send commands at 20Hz
    end
end


function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.city_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    #localization_state_channel = Channel{MyLocalizationType}(1)
    #perception_state_channel = Channel{MyPerceptionType}(1)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment
        ego_vehicle_id = measurement_msg.vehicle_id
        for meas in measurement_msg.measurements
            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)

    @async localize(gps_channel, imu_channel, localization_state_channel)
    @async perception(cam_channel, localization_state_channel, perception_state_channel)
    @async decision_making(localization_state_channel, perception_state_channel, map, socket)
end
