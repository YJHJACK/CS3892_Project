using DataStructures
using StaticArrays
using Interpolations
using VehicleSim
using LinearAlgebra
using Plots

mutable struct PIDController
    Kp::Float64
    Ki::Float64
    Kd::Float64
    integral::Float64
    prev_error::Float64

    function PIDController(Kp, Ki, Kd)
        new(Kp, Ki, Kd, 0.0, 0.0)
    end
end

function pid_update(pid::PIDController, error, dt)
    pid.integral += error * dt
    derivative = (error - pid.prev_error) / dt
    pid.prev_error = error
    return pid.Kp * error + pid.Ki * pid.integral + pid.Kd * derivative
end

export MyLocalizationType, Detected_Obj, MyPerceptionType

struct MyLocalizationType
    valid::Bool                         # indicates if the estimate is valid
    position::SVector{3,Float64}        # x, y, z coordinates
    yaw::Float64                        # heading angle
    velocity::SVector{3,Float64}        # velocity vector
    covariance::Matrix{Float64}         # uncertainty matrix
    timestamp::Float64                  # time of the estimate
end

struct Detected_Obj
    id::Int  # id for each object
    bbox::NTuple{4, Float64}           # Bounding box: (x_min, y_min, x_max, y_max).
    confidence::Float64                # Confidence score from the CNN.
    classification::String             # e.g., "vehicle", "pedestrian"
    position::SVector{2, Float64}      # Estimated 2D position (from EKF fusion).
    velocity::SVector{2, Float64}      # Estimated 2D velocity (if available).
    uncertainty::Matrix{Float64}       # Covariance of the position estimate.
    sensor_source::String              # e.g., "camera", "lidar"
end

struct Particle
    angle::Float64            # angle
    loc::SVector{2,Float64}   # pos
    v::Float64                # velocity
    w::Float64                # particle weight
end

struct MyPerceptionType
    timestamp::Float64
    detections::Vector{Detected_Obj}
end

function within_lane(pos::SVector{2,Float64}, seg::VehicleSim.RoadSegment)
    nb = length(seg.lane_boundaries)
    
    if nb ≥ 3
        # Use the second and third lane boundaries.
        A = seg.lane_boundaries[2].pt_a
        B = seg.lane_boundaries[2].pt_b
        C = seg.lane_boundaries[3].pt_a
        D = seg.lane_boundaries[3].pt_b
    elseif nb == 2
        A = seg.lane_boundaries[1].pt_a
        B = seg.lane_boundaries[1].pt_b
        C = seg.lane_boundaries[2].pt_a
        D = seg.lane_boundaries[2].pt_b
    else
        @warn "Segment has less than 2 lane boundaries!"
        return false
    end

    min_x = min(A[1], B[1], C[1], D[1])
    max_x = max(A[1], B[1], C[1], D[1])
    min_y = min(A[2], B[2], C[2], D[2])
    max_y = max(A[2], B[2], C[2], D[2])
    
    return (min_x <= pos[1] <= max_x) && (min_y <= pos[2] <= max_y)
end

function collision_avoidance_control(ego::MyLocalizationType,
    perc::MyPerceptionType,
    planned_vel::Float64)
    safety_distance = 8.0
    factor = 1.0
    forward = SVector(cos(ego.yaw), sin(ego.yaw))
    for obj in perc.detected_objs
        rel = obj.position - ego.position[1:2]
        dist_ahead = dot(rel, forward)
        if dist_ahead>0 && dist_ahead<safety_distance
            factor = min(factor, dist_ahead/safety_distance)
        end
    end
    return planned_vel*factor
end

function least_lane_path(all_segs::Dict{Int,VehicleSim.RoadSegment}, start_id::Int, goal_id::Int)
    queue = Queue{Int}()
    enqueue!(queue, start_id)
    came_from = Dict{Int,Int}()
    visited = Set{Int}([start_id])
    
    while !isempty(queue)
        current = dequeue!(queue)
        if current == goal_id
            path = [current]
            while haskey(came_from, current)
                current = came_from[current]
                push!(path, current)
            end
            return reverse(path)
        end
        
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

function compute_segment_straight_points(seg::VehicleSim.RoadSegment, target_id::Int)
    lb1 = seg.lane_boundaries[1]
    lb2 = seg.lane_boundaries[2]
    if(seg.id == target_id)
        lb1 = seg.lane_boundaries[2]
        lb2 = seg.lane_boundaries[3]
    end
    start_center = 0.5 * (lb1.pt_a + lb2.pt_a)
    end_center   = 0.5 * (lb1.pt_b + lb2.pt_b)
    t_vals = range(0.0, 1.0, 5)

    points = [ start_center .+ τ .* (end_center .- start_center) for τ in t_vals ]
    return points
end

# Generate a smooth trajectory from a list of segments.
# For each segment, if the curvature is negligible the segment is treated as straight;
# otherwise it is treated as curved.
function generate_trajectory_plan(segments::Vector{VehicleSim.RoadSegment}, target_id::Int)
    raw_pts = SVector{2,Float64}[]

    for (i, seg) in enumerate(segments)
        lb = seg.lane_boundaries[1]
        pts = if isapprox(lb.curvature, 0.0; atol=1e-6)
            compute_segment_straight_points(seg, target_id)
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

    N = length(raw_pts)
    cum_dists = zeros(Float64, N)
    for i in 2:N
        cum_dists[i] = cum_dists[i-1] + norm(raw_pts[i] - raw_pts[i-1])
    end
    total_length = cum_dists[end]
    
    num_samples = 500
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

function process_gt(gt_ch, sd_ch, loc_ch, perc_ch)
    while true
        fetch(sd_ch) && break
        b=[]; while isready(gt_ch); push!(b,take!(gt_ch)); end
        take!(loc_ch); put!(loc_ch, new_localization_state_from_gt)
        take!(perc_ch); put!(perc_ch, new_perception_state_from_gt)
    end
end

function normalize_angle(a::Float64)
    while a>π; a-=2π end
    while a<-π; a+=2π end
    return a
end

function ekf_predict(x::Vector{Float64}, P::AbstractMatrix{Float64}, Δt::Float64, Q::AbstractMatrix{Float64})
    x_pred = f(x, Δt)
    F      = Jac_x_f(x, Δt)
    P_pred = F*P*F' + Q
    return x_pred, P_pred
end

function ekf_update(x::Vector{Float64}, P::AbstractMatrix{Float64}, z::Vector{Float64}, h::Function, H::Function, R::AbstractMatrix{Float64})
    z_pred = h(x)
    y = z .- z_pred
    if length(y)≥3; y[3]=normalize_angle(y[3]); end
    Hx = H(x)
    S = Hx*P*Hx' + R
    K = P*Hx'*inv(S)
    x_new = x + K*y
    P_new = (I - K*Hx)*P
    return x_new, P_new
end

function localization(meas_ch::Channel{MeasurementMessage}, state_ch::Channel{Tuple{Vector{Float64},Matrix{Float64}}}; dt_default=0.1)
    x_est = vcat([0.0,0.0,0.0], [1.0,0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0])
    P_est = Diagonal([1.0,1.0,1.0, 0.1,0.1,0.1,0.1, 0.5,0.5,0.5, 0.1,0.1,0.1])
    Q     = Diagonal([0.3,0.3,0.3, 0.005,0.005,0.005,0.005, 0.2,0.2,0.2, 0.02,0.02,0.02])
    Rgps  = Diagonal([0.5,0.5,0.05])
    last_time = time()
    while true
        msg = take!(meas_ch)
        t = isempty(msg.measurements) ? time() : msg.measurements[1].time
        Δt = max(t - last_time, dt_default)
        last_time = t
        x_pred, P_pred = ekf_predict(x_est, P_est, Δt, Q)
        x_upd, P_upd = x_pred, P_pred
        for m in msg.measurements
            if m isa GPSMeasurement
                z = [m.lat, m.long, m.heading]
                x_upd, P_upd = ekf_update(x_upd, P_upd, z, h_gps, Jac_h_gps, Rgps)
            end
        end
        x_est, P_est = x_upd, P_upd
        put!(state_ch, (x_est, Matrix(P_est)))
        sleep(0.001)
    end
end

function convert_state(x::Vector{Float64}, P::Matrix{Float64}, t::Float64)::MyLocalizationType
    return MyLocalizationType(true,
        SVector(x[1:3]...),
        extract_yaw_from_quaternion(x[4:7]),
        SVector(x[8:10]...),
        P,
        t)
end

function localize(gps_ch::Channel{GPSMeasurement}, imu_ch::Channel{IMUMeasurement}, loc_ch::Channel{MyLocalizationType}, sd_ch::Channel{Bool}; dt_default=0.1)
    meas_ch  = Channel{MeasurementMessage}(32)
    state_ch = Channel{Tuple{Vector{Float64},Matrix{Float64}}}(32)
    errormonitor(@async localization(meas_ch, state_ch; dt_default=dt_default))
    while true
        if isready(sd_ch) && take!(sd_ch)
            close(meas_ch); close(state_ch); break
        end
        gps_msgs = GPSMeasurement[]
        while isready(gps_ch); push!(gps_msgs, take!(gps_ch)); end
        if !isempty(gps_msgs)
            put!(meas_ch, MeasurementMessage(1,0,gps_msgs))
        end
        if isready(state_ch)
            xv, Pv = take!(state_ch)
            put!(loc_ch, convert_state(xv, Pv, time()))
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
    focal_len=0.64
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

function bboxes_error(pred_bboxes, true_bboxes)
    if isempty(pred_bboxes) || isempty(true_bboxes)
        return Inf
    end
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
    
function get_cam_transform(camera_id::Int)
    R_cam_to_body = RotY(0.02)
    t_cam_to_body = [1.35,  1.7, 2.4]
    if camera_id == 2
        t_cam_to_body[2] = -1.7
    end
    return hcat(R_cam_to_body, t_cam_to_body)
end

function get_rotated_camera_transform()
    R = [ 0.0  0.0 1.0;
         -1.0  0.0 0.0;
          0.0 -1.0 0.0 ]
    t = zeros(3)
    [R t]
end

function multiply_transforms(T1, T2)
    T1f = [T1; [0 0 0 1.]] 
    T2f = [T2; [0 0 0 1.]]

    T = T1f * T2f
    T = T[1:3, :]
end

#particle & bbox perception function
function perception(cam_meas_channel, localization_state_channel, perception_state_channel, shutdown_channel)
    # set up stuff
    last_time = time()
    particles = Particle[]
    while true
        if isready(shutdown_channel)
            break
        end
        fresh_cam_meas = CameraMeasurement[]
        while isready(cam_meas_channel)
            push!(fresh_cam_meas, take!(cam_meas_channel))
        end

        if isempty(fresh_cam_meas)
            put!(perception_state_channel, MyPerceptionType(time(), Detected_Obj[]))
            sleep(0.005)
            continue
        end

        if isready(localization_state_channel)
            loc_state = take!(localization_state_channel)
            ego_orientation = loc_state.yaw
            ego_position    = loc_state.position
        else
            put!(perception_state_channel, MyPerceptionType(time(), Detected_Obj[]))
            sleep(0.005)
            continue
        end

        b1 = Vector{SVector{4,Int}}()
        b2 = Vector{SVector{4,Int}}()
        for fresh in fresh_cam_meas
            @info fresh
            if fresh.camera_id == 1
                append!(b1, fresh.bounding_boxes)
            else  
                append!(b2, fresh.bounding_boxes)
            end
        end

        # calculate target pos
        T1 = get_cam_transform(1)
        T2 = get_cam_transform(2)
        Tr = get_rotated_camera_transform()
        T_body_camrot1 = multiply_transforms(T1, Tr)
        T_body_camrot2 = multiply_transforms(T2, Tr)

        best_quat, best_loc, min_error, estimated_region =
            estimate_location_from_2_bboxes(ego_orientation, ego_position,
                                        T_body_camrot1, T_body_camrot2,
                                        b1, b2)
        sorted_candidates = sort!(quat_loc_bboxerror_list, by = x -> x[3])
        selected_candidates = sorted_candidates[1:min(10, length(sorted_candidates))]

        # initialize particles
        particles = initialize_particles(selected_candidates;
                                     varangle = pi/12, var_location = 0.5, 
                                     max_v = 7.5, step_v = 0.5, number_of_particles = 1000)

        if !isempty(b1)
            bbox = b1[1]
            est_obj_ori, est_obj_loc = estimate_object_state(ego_state, bbox; scale_factor=pixel_len)
        elseif !isempty(b2)
            bbox = b2[1]
            est_obj_ori, est_obj_loc = estimate_object_state(ego_state, bbox; scale_factor=pixel_len)
        else
            bbox = (0.0, 0.0, 0.0, 0.0)
            est_obj_ori = ego_orientation
            est_obj_loc = ego_position
        end

        x_mean = sum(p.loc[1] * p.w for p in particles)
        y_mean = sum(p.loc[2] * p.w for p in particles)
        center = SVector(x_mean, y_mean)

        cov_xx = sum(p.w * (p.loc[1] - x_mean)^2 for p in particles)
        cov_yy = sum(p.w * (p.loc[2] - y_mean)^2 for p in particles)
        cov_xy = sum(p.w * (p.loc[1] - x_mean)*(p.loc[2] - y_mean) for p in particles)
        pos_cov = [cov_xx cov_xy; cov_xy cov_yy]

        detected_object = Detected_Obj(
            1,
            bbox,
            1.0,
            "vehicle",
            center,
            SVector(0.0,0.0),     
            pos_cov,           
            "camera"
        )
    
        # update particles
        delta_t = time() - last_time
        last_time = time()
        updated_particles = update_particles(particles, delta_t,
                                         b1, b2,
                                         ego_orientation, ego_position,
                                         T_body_camrot1, T_body_camrot2,
                                         image_width, image_height, pixel_len)

        perception_msg = MyPerceptionType(time(), [detected_object])

        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_msg)
        sleep(0.01)
    end
end

function find_current_segment(pos2d::SVector{2,Float64}, map::Dict{Int,VehicleSim.RoadSegment})
    for (seg_id, seg) in map
        if within_lane(pos2d, seg)
            return seg_id
        end
    end
    return nothing
end

function find_target_point(traj_points, current_pos, lookahead_dist)
    isempty(traj_points) && error("Trajectory points are empty")
    
    distances = [norm(p - current_pos) for p in traj_points]
    closest_idx = argmin(distances)
    
    cumulative_dist = 0.0
    target_point = traj_points[closest_idx]
    
    for i in closest_idx:min(closest_idx+100, length(traj_points)-1)
        segment = traj_points[i+1] - traj_points[i]
        segment_length = norm(segment)
        
        if cumulative_dist + segment_length >= lookahead_dist
            ratio = (lookahead_dist - cumulative_dist) / segment_length
            return traj_points[i] + ratio * segment
        end
        
        cumulative_dist += segment_length
        target_point = traj_points[i+1]
    end
    return target_point
end

function decision_making(loc_ch, perc_ch, map::Dict{Int,VehicleSim.RoadSegment}, target_seg::Int, sock)

    dt = 0.01
    MAX_STEERING_RATE = 0.4
    MAX_SPEED = 4.0
    MAX_ACCEL = 5.0
    wheelbase = 2.0
    
    lookahead_base = 3.5
    lookahead_time = 1.5
    min_lookahead = 1.0
    max_lookahead = 6.0
    
    k_p_speed = 1.5
    k_i_speed = 0.15
    speed_pid = PIDController(k_p_speed, k_i_speed, 0.0)
    
    prev_steering_angle = 0.0
    stop_now = false
    stop_counter = 0.0
    is_braking = false
    is_steering_adjustment = false
    STEERING_ERROR_THRESHOLD = 0.05
    STEERING_RECOVER_THRESHOLD = 0.03
    
    loc0 = fetch(loc_ch)
    p0 = SVector(loc0.position[1], loc0.position[2])
    s0 = find_current_segment(p0, map)
    route = least_lane_path(map, s0, target_seg)
    segs = [map[id] for id in route]
    traj = generate_trajectory_plan(segs, target_seg)
    traj_points = [SVector{2}(p[1], p[2]) for p in traj]

    while isopen(sock)
        loc = fetch(loc_ch)
        perc = fetch(perc_ch)
        
        if !loc.valid
            serialize(sock, (0.0, 0.0, true))
            sleep(dt)
            continue
        end

        current_pos = SVector(loc.position[1], loc.position[2])
        current_seg_id = find_current_segment(current_pos, map)
        current_seg = nothing
        if current_seg_id ≠ nothing
            current_seg = map[current_seg_id]
        end
        current_heading = loc.yaw
        current_speed = norm(SVector(loc.velocity[1], loc.velocity[2]))
        
        #break if there is a car ahead
        # ========================================
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
            if new_speed < 0.05
                serialize(sock, (0.0, 0.0, true))
                @info "stop!"
                sleep(dt)
                continue
            end
            serialize(sock, (0.0, new_speed, true))
            sleep(dt)
            continue
        end

        # ========================================

        #break if there is a stop sign
        # ========================================
        if current_seg ≠ nothing && stop_counter == 0
            lb1 = current_seg.lane_boundaries[1]
            lb2 = current_seg.lane_boundaries[2]
            seg_end = 0.5 * (lb1.pt_b + lb2.pt_b)
            dist_to_seg_end = norm(current_pos - seg_end) 
            for i in 1:length(current_seg.lane_types)
                if current_seg.lane_types[i] == VehicleSim.stop_sign && dist_to_seg_end < 10.0
                    stop_now = true
                end
            end
        end

        if stop_now
            @info "stop sigh here, $current_speed"
            serialize(sock, (0.0, 0.0, true))
            if current_speed < 0.001
                @info "stop!"
                sleep(1)
                stop_now = false
                stop_counter = 10.0
            end
            continue
        end

        if stop_counter > 0
            stop_counter -= 0.01
        end
        
        # ========================================

        #break if we arrive
        # ========================================
        end_point = traj_points[end]
        distance_to_end = norm(current_pos - end_point)
        if !is_braking && distance_to_end ≤ 5.0
            is_braking = true
            @info "start to break"
        end

        if is_braking
            target_speed = if distance_to_end ≤ 5.0
                0.0
            else
                lerp(current_speed, 0.0, (5.0 - distance_to_end)/5.0)
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
        # ========================================
        
        lookahead_dist = 5.0
        target_point = find_target_point(traj_points, current_pos, lookahead_dist)
        
        vec = target_point - current_pos
        x_vehicle = cos(current_heading) * vec[1] + sin(current_heading) * vec[2]
        y_vehicle = -sin(current_heading) * vec[1] + cos(current_heading) * vec[2]
        target_alpha = atan(y_vehicle, x_vehicle)
        
        alpha = target_alpha
        alpha = alpha - 2π * floor((alpha + π) / (2π))
        
        # 转向状态机
        if !is_steering_adjustment && abs(alpha) > STEERING_ERROR_THRESHOLD
            @info "$(round(alpha, digits=3))"
            is_steering_adjustment = true
            speed_pid.integral = 0.0
        end
        
        if is_steering_adjustment
            target_speed = 0.0
            speed_error = target_speed - current_speed
            acceleration = clamp(speed_error * 5.0, -MAX_ACCEL, 0.0)
            new_speed = max(current_speed + acceleration * dt, 0.0)
            desired_steering = atan(2 * wheelbase * sin(alpha) / lookahead_dist)
            
            if abs(alpha) ≤ STEERING_RECOVER_THRESHOLD
                @info "转向调整完成"
                is_steering_adjustment = false
            end
        else
            target_speed = is_braking ? lerp(current_speed, 0.0, (10.0 - distance_to_end)/5.0) : MAX_SPEED
            speed_error = target_speed - current_speed
            acceleration = pid_update(speed_pid, speed_error, dt)
            acceleration = clamp(acceleration, -MAX_ACCEL, MAX_ACCEL)
            
            new_speed = clamp(current_speed + acceleration * dt, 0.0, MAX_SPEED)
            desired_steering = atan(2 * wheelbase * sin(alpha) / lookahead_dist)
        end

        steering_delta = desired_steering - prev_steering_angle
        clamped_delta = clamp(steering_delta, -MAX_STEERING_RATE*dt, MAX_STEERING_RATE*dt)
        current_steering = prev_steering_angle + clamped_delta
        prev_steering_angle = current_steering

        serialize(sock, (current_steering, new_speed, true))
        sleep(dt)
    end
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
