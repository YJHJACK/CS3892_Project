using DataStructures
using StaticArrays
using Interpolations
using VehicleSim
using Sockets
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

# export MyLocalizationType

struct Detected_Obj
    id::Int  # id for each object
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
    timestamp::Float64
    obstacles::Vector{Detected_Obj}
    estimated_region::NTuple{4,Float64}
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

function compute_segment_arc_points(seg::VehicleSim.RoadSegment, npts::Int=1)
    lb = seg.lane_boundaries[1]
    c  = lb.curvature
    pA = lb.pt_a
    pB = lb.pt_b

    if isapprox(c, 0.0; atol=1e-6)
        return [compute_segment_center_point(seg)]
    end

    lane_width = 5.0

    inside_rad = 1.0 / abs(c)
    middle_rad = inside_rad
    chord = pB - pA
    L = norm(chord)
    if L/2 > middle_rad
        @warn "Chord length is too long relative to the computed middle_rad. Returning endpoints."
        return [pA, 0.5*(pA+pB), pB]
    end

    mid = 0.5 * (pA + pB)

    h = sqrt(middle_rad^2 - (L/2)^2)

    n = SVector(-chord[2], chord[1]) / L

    center = c > 0 ? mid + h * n : mid - h * n

    θA = atan(pA[2] - center[2], pA[1] - center[1])
    θB = atan(pB[2] - center[2], pB[1] - center[1])
    span = θB - θA
    if c > 0
        if span < 0
            span += 2π
        end
    else
        if span > 0
            span -= 2π
        end
    end
    pts = SVector{2,Float64}[]
    for t in range(0.0, step=1.0, length=1)
        θ = θA + t * span
        push!(pts, center + middle_rad * SVector(cos(θ), sin(θ)))
    end

    return pts
end

function compute_segment_straight_points(seg::VehicleSim.RoadSegment)
    lb1 = seg.lane_boundaries[1]
    lb2 = seg.lane_boundaries[2]
    start_center = 0.5 * (lb1.pt_a + lb2.pt_a)
    end_center   = 0.5 * (lb1.pt_b + lb2.pt_b)
    return [start_center[1:2], end_center[1:2]]
end

function generate_trajectory_plan(segments::Vector{VehicleSim.RoadSegment})
    raw_pts = SVector{2,Float64}[]

    for (i, seg) in enumerate(segments)
        lb = seg.lane_boundaries[1]
        pts = if isapprox(lb.curvature, 0.0; atol=1e-6)
            compute_segment_straight_points(seg)
        else
            compute_segment_arc_points(seg, 1)
        end

        if i == 1
            append!(raw_pts, pts)
        else
            append!(raw_pts, pts[2:end])
        end        
    end

    xs = [p[1] for p in raw_pts]
    ys = [p[2] for p in raw_pts]

    plt = plot(xs, ys; st=:scatter, texts=1:length(xs))
    display(plt)

    N = length(raw_pts)
    tk = range(0, 1, length = N)
    xs = [p[1] for p in raw_pts]
    ys = [p[2] for p in raw_pts]
    sx = CubicSplineInterpolation(tk, xs)
    sy = CubicSplineInterpolation(tk, ys)

    N = length(raw_pts)
    cum_dists = zeros(Float64, N)
    for i in 2:N
        cum_dists[i] = cum_dists[i-1] + norm(raw_pts[i] - raw_pts[i-1])
    end
    total_length = cum_dists[end]
    
    num_samples = 200
    sample_dists = range(0, stop=total_length, length=num_samples)
    sampled_points = SVector{2,Float64}[]
    
    current_index = 1
    for d in sample_dists
        while current_index < N && cum_dists[current_index+1] < d
            current_index += 1
        end
        if current_index == N
            push!(sampled_points, raw_pts[end])
        else
            d0 = cum_dists[current_index]
            d1 = cum_dists[current_index+1]
            t = (d - d0) / (d1 - d0)
            pt = raw_pts[current_index] + t * (raw_pts[current_index+1] - raw_pts[current_index])
            push!(sampled_points, pt)
        end
    end
    return sampled_points
end

function find_current_segment(pos2d::SVector{2,Float64}, map::Dict{Int,VehicleSim.RoadSegment})
    for (seg_id, seg) in map
        if within_lane(pos2d, seg)
            return seg_id
        end
    end
    error("No segment found containing position $pos2d")
end

function decision_making(loc_ch, perc_ch,
                        map::Dict{Int,VehicleSim.RoadSegment},
                        target_seg::Int, sock)
    @info "==== Entering decision_making ===="
    dt = 0.01
    MAX_STEERING_RATE = 1.25
    MAX_SPEED = 4.0
    MAX_ACCEL = 5.0

    lookahead_base = 1.5
    lookahead_time = 2.0
    min_lookahead = 1.0
    max_lookahead = 6.0
    wheelbase = 3.0
    k_p_speed = 1.5
    k_i_speed = 0.15

    prev_steering_angle = 0.0
    integral = 0.0

    loc0 = fetch(loc_ch)
    p0 = SVector(loc0.position[1], loc0.position[2])
    s0 = find_current_segment(p0, map)
    route = least_lane_path(map, s0, target_seg)
    segs = [map[id] for id in route]
    traj = generate_trajectory_plan(segs)
    traj_points = [SVector{2}(p[1], p[2]) for p in traj]

    BRAKING_DISTANCE = 10.0
    STOP_DISTANCE = 8.0
    STOP_SPEED_THRESHOLD = 0.5
    is_braking = false
    while isopen(sock)
        loc = fetch(loc_ch)
        perc = fetch(perc_ch)
        if !loc.valid
            serialize(sock, (0.0, 0.0, true))
            sleep(dt)
            continue
        end

        current_pos = SVector(loc.position[1], loc.position[2])
        current_heading = loc.yaw
        current_speed = norm(SVector(loc.velocity[1], loc.velocity[2]))
        end_point = traj_points[end]
        distance_to_end = norm(current_pos - end_point)

        obstacle_detected = false
        safe_distance = 35.0

        for obj in perc.detections
            if obj.classification == "vehicle"
                rel_pos = obj.position - current_pos
                rel_dist = norm(rel_pos)

                angle_to_obj = atan(rel_pos[2], rel_pos[1])
                heading_error = atan(sin(angle_to_obj - current_heading), cos(angle_to_obj - current_heading))

                if abs(heading_error) < π/4 && rel_dist ≤ safe_distance
                    obstacle_detected = true
                    break
                end
            end
        end

        if obstacle_detected
            @info "Detected vehicle in front within $safe_distance meters. Slowing down..."
            target_speed = 0.0
            speed_error = target_speed - current_speed
            acceleration = 3*clamp(speed_error * 2.0, -MAX_ACCEL, 0.0)
            new_speed = max(current_speed + acceleration * dt *3, 0.0)
            serialize(sock, (0.0, new_speed, true))
            sleep(dt)
            continue
        end

        if !is_braking && distance_to_end ≤ BRAKING_DISTANCE
            @info "breaking"
            is_braking = true
        end

        if is_braking
            target_speed = if distance_to_end ≤ STOP_DISTANCE
                0.0
            else
                lerp(current_speed, 0.0, (BRAKING_DISTANCE - distance_to_end)/5.0)
            end

            speed_error = target_speed - current_speed
            acceleration = 1.5*clamp(speed_error * 2.0, -MAX_ACCEL, 0.0)
            new_speed = current_speed + acceleration * dt
            new_speed = max(new_speed, 0.0)

            serialize(sock, (0.0, 0.0, true))

            if new_speed ≤ STOP_SPEED_THRESHOLD
                serialize(sock, (0.0, 0.0, false))
                @info "stopped, end loop"
                break
            end

            sleep(dt)
            continue
        end
        distances = [norm(p - current_pos) for p in traj_points]
        closest_idx = argmin(distances)
        
        lookahead_dist = lookahead_base + current_speed * lookahead_time
        lookahead_dist = clamp(lookahead_dist, min_lookahead, max_lookahead)

        target_point = traj_points[closest_idx]
        cumulative_dist = 0.0
        for i in closest_idx:min(closest_idx+30, length(traj_points)-1)
            segment_length = norm(traj_points[i+1] - traj_points[i])
            if cumulative_dist + segment_length >= lookahead_dist
                ratio = (lookahead_dist - cumulative_dist) / segment_length
                target_point = traj_points[i] + ratio*(traj_points[i+1]-traj_points[i])
                break
            end
            cumulative_dist += segment_length
            target_point = traj_points[i+1]
        end

        vec_to_target = target_point - current_pos
        target_heading = atan(vec_to_target[2], vec_to_target[1])
        α = target_heading - current_heading
        α = atan(sin(α), cos(α))

        curvature = sin(α) / (lookahead_dist + 0.05*current_speed)
        desired_steering = atan(curvature * wheelbase)

        max_delta = MAX_STEERING_RATE * dt * 0.8
        steering_delta = desired_steering - prev_steering_angle
        clamped_delta = clamp(steering_delta, -max_delta, max_delta)
        current_steering = min(prev_steering_angle + clamped_delta, MAX_STEERING_RATE)
        prev_steering_angle = current_steering

        target_speed = MAX_SPEED
        speed_error = target_speed - current_speed
        integral += speed_error * dt
        integral = clamp(integral, -1.5, 1.5)

        acceleration = 5*(k_p_speed * speed_error + k_i_speed * integral)
        acceleration = clamp(acceleration, -MAX_ACCEL, MAX_ACCEL)
        new_speed = current_speed + acceleration * dt
        new_speed = clamp(new_speed, 0.0, MAX_SPEED)

        serialize(sock, (current_steering, new_speed, true))
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
    P_est = Diagonal([1.0, 1.0, 1.0,      # Position
                      0.1, 0.1, 0.1, 0.1, # Quaternion
                      0.5, 0.5, 0.5,      # Velocity
                      0.1, 0.1, 0.1])     # Angular velocity
    
    # Process noise covariance Q
    Q = Diagonal([0.3, 0.3, 0.3,               # Position noise
                  0.005, 0.005, 0.005, 0.005,  # Quaternion noise
                  0.2, 0.2, 0.2,               # Velocity noise
                  0.02, 0.02, 0.02])           # Angular velocity noise
    
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
                candidate_loc = SVector(ego_position[1] + dx, ego_position[2] + dy, ego_position[3])

                # 2. predict bbox on canmera besed on pose
                pred_bboxes_cam1 = predict_bboxes(candidate_quat, candidate_loc, T_body_camrot1)
                pred_bboxes_cam2 = predict_bboxes(candidate_quat, candidate_loc, T_body_camrot2)

                # 3. calculate errors
                cam1_error = bboxes_error(pred_bboxes_cam1, true_bboxes_cam1)
                cam2_error = bboxes_error(pred_bboxes_cam2, true_bboxes_cam2)
                total_error = cam1_error + cam2_error

                push!(quat_loc_bboxerror_list, (candidate_quat, candidate_loc, total_error))
            end
        end
    end

    # 5. find min error
    sort!(quat_loc_bboxerror_list, by = x->x[3])  
    best_candidate = first(quat_loc_bboxerror_list)
    best_quat   = best_candidate[1]
    best_loc    = best_candidate[2]
    min_error   = best_candidate[3]

    margin = 1.0
    estimated_region = (best_loc[1] - margin, best_loc[2] - margin,
    best_loc[1] + margin, best_loc[2] + margin)

    return best_quat, best_loc, min_error, estimated_region
end

function predict_bboxes(quat, loc, T_body_camrot, image_width=640, image_height=480, pixel_len=0.001)
    object_width = image_width * pixel_len   
    object_height = image_height * pixel_len   
    object_depth = 1.0                          
    object_size = SVector(object_width, object_height, object_depth)

    corners_world = compute_3d_corners(quat, loc, object_size)

    projected_points = Vector{Tuple{Int,Int}}()
    for Xh in corners_world
        X_cam_hom = T_body_camrot * Xh
        X_cam = X_cam_hom[1:3] 
        Z = X_cam[3]
        if Z <= 0
            continue  
        end

        proj_x = focal_len * X_cam[1] / Z
        proj_y = focal_len * X_cam[2] / Z
    
        px = convert_to_pixel(image_width, pixel_len, proj_x)
        py = convert_to_pixel(image_height, pixel_len, proj_y)
    
        push!(projected_points, (px, py))
    end

    if isempty(projected_points)
        return []
    end
    u_vals = [pt[1] for pt in projected_points]
    v_vals = [pt[2] for pt in projected_points]
    x_min = minimum(u_vals)
    y_min = minimum(v_vals)
    x_max = maximum(u_vals)
    y_max = maximum(v_vals)

    return [(x_min, y_min, x_max, y_max)]
end

function compute_3d_corners(quat, loc, object_size)
    w, h, d = object_size
    dx = w/2
    dy = h/2
    dz = d/2
    corners = [
        SVector(loc[1]-dx, loc[2]-dy, loc[3]-dz),
        SVector(loc[1]-dx, loc[2]-dy, loc[3]+dz),
        SVector(loc[1]-dx, loc[2]+dy, loc[3]-dz),
        SVector(loc[1]-dx, loc[2]+dy, loc[3]+dz),
        SVector(loc[1]+dx, loc[2]-dy, loc[3]-dz),
        SVector(loc[1]+dx, loc[2]-dy, loc[3]+dz),
        SVector(loc[1]+dx, loc[2]+dy, loc[3]-dz),
        SVector(loc[1]+dx, loc[2]+dy, loc[3]+dz)
    ]
    return [vcat(corner, 1.0) for corner in corners]
end

function convert_to_pixel(num_pixels, pixel_len, px)
    min_val = -pixel_len*num_pixels/2
    pix_id = cld(px - min_val, pixel_len)+1 |> Int
    return pix_id
end

function yaw_to_quaternion(yaw::Float64)
    half = yaw / 2
    return SVector(cos(half), 0.0, 0.0, sin(half))
end

function quaternion_to_yaw(q::SVector{4,Float64})
    return atan(2*(q[1]*q[4] + q[2]*q[3]), 1 - 2*(q[3]^2 + q[4]^2))
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

        # mean = [ base_angle, loc[1], loc[2] ]
        # cov  = diagm([var_angle^2, var_location^2, var_location^2])
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
    

#particle & bbox perception function
function perception(cam_meas_channel, localization_state_channel, perception_state_channel, shutdown_channel)
    # set up stuff
    last_time = time()
    while true
        if isready(shutdown_channel)
            break
        end
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        if isempty(fresh_cam_meas)
            sleep(0.005)
            continue
        end
        current_meas = fresh_cam_meas[end]
        true_bboxes_cam1 = haskey(current_meas, :bboxes_cam1) ? current_meas[:bboxes_cam1] : []
        true_bboxes_cam2 = haskey(current_meas, :bboxes_cam2) ? current_meas[:bboxes_cam2] : []

        # calculate target pos
        best_quat, best_loc, min_error, estimated_region =
            estimate_location_from_2_bboxes(ego_orientation, ego_position,
                                        T_body_camrot1, T_body_camrot2,
                                        true_bboxes_cam1, true_bboxes_cam2)
        sorted_candidates = sort!(quat_loc_bboxerror_list, by = x -> x[3])
        selected_candidates = sorted_candidates[1:min(10, length(sorted_candidates))]

        # initialize particles
        particles = initialize_particles(selected_candidates, vehicle_size;
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

        detected_object = Detected_Obj(1, true_bboxes_cam1 != [] ? true_bboxes_cam1[1] : (0.0,0.0,0.0,0.0),
                                        1.0, "vehicle", est_obj_loc[1:2], SVector(0.0,0.0))
    
        # update particles
        delta_t = time() - last_time
        last_time = time()
        updated_particles = update_particles(particles, delta_t,
                                         true_bboxes_cam1, true_bboxes_cam2,
                                         ego_orientation, ego_position,
                                         T_body_camrot1, T_body_camrot2,
                                         image_width, image_height, pixel_len)

        perception_msg = MyPerceptionType(time(), [detected_object], estimated_region)

        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
        sleep(0.01)
    end
end

function my_client(host::IPAddr = IPv4(0), port::Int = 4444)
    # Connect to the server
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.city_map()
    
    # Deserialize visualization information from the server
    msg = deserialize(socket)  # Visualization info
    @info msg

    # Create channels for incoming measurements
    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel  = Channel{GroundTruthMeasurement}(32)

    # Initialize target map segment and ego vehicle id (they will be overwritten by incoming messages)
    target_map_segment = 0
    ego_vehicle_id = 0

    # Define a simple isfull function to check if a channel is full (using its capacity)
    function isfull(ch::Channel)
        return length(ch) >= ch.capacity
    end

    # Start an asynchronous task to continuously read measurement messages from the socket
    @async begin
        while true
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
            if !received
                continue
            end

            # Update target map segment and ego vehicle id from the received message
            target_map_segment = measurement_msg.target_segment
            ego_vehicle_id = measurement_msg.vehicle_id

            # Dispatch measurements to the corresponding channels
            for meas in measurement_msg.measurements
                if meas isa GPSMeasurement
                    if !isfull(gps_channel)
                        put!(gps_channel, meas)
                    end
                elseif meas isa IMUMeasurement
                    if !isfull(imu_channel)
                        put!(imu_channel, meas)
                    end
                elseif meas isa CameraMeasurement
                    if !isfull(cam_channel)
                        put!(cam_channel, meas)
                    end
                elseif meas isa GroundTruthMeasurement
                    if !isfull(gt_channel)
                        put!(gt_channel, meas)
                    end
                end
            end
        end
    end

    # Create a shutdown channel for the localize function
    shutdown_channel = Channel{Bool}(1)

    # Launch asynchronous tasks for localization, perception, and decision making.
    @async localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel)
    # Note: perception function requires ekf and cnn_model; placeholders (nothing) are used here.
    @async perception(cam_channel, localization_state_channel, perception_state_channel, shutdown_channel)
    # Launch decision making with localization state, perception state, map segments, target segment, and socket.
    @async decision_making(localization_state_channel, perception_state_channel, map_segments, target_map_segment, socket)
end
