struct MyLocalizationType
    field1::Int
    field2::Float64
end

struct MyPerceptionType
    field1::Int
    field2::Float64
end

mutable struct ObjectEKF
    x::Vector{Float64}      # State: [x, y, vx, vy]
    P::Matrix{Float64}      # Covariance
    Q::Matrix{Float64}      # Process noise
    R::Matrix{Float64}      # Measurement noise
end

function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    while true
        fresh_gps_meas = []
        while isready(gps_channel)
            meas = take!(gps_channel)
            push!(fresh_gps_meas, meas)
        end
        fresh_imu_meas = []
        while isready(imu_channel)
            meas = take!(imu_channel)
            push!(fresh_imu_meas, meas)
        end

        # Process measurements - a simple placeholder using GPS data
        if !isempty(fresh_gps_meas)
            lat_sum = 0.0
            lon_sum = 0.0
            for m in fresh_gps_meas
                lat_sum += m.lat
                lon_sum += m.long
            end
            avg_lat = lat_sum / length(fresh_gps_meas)
            avg_lon = lon_sum / length(fresh_gps_meas)

            # Simple position estimation placeholder
            # Change to EKF in the future
            localization_state = MyLocalizationType(1, avg_lat + avg_lon)
        else
            localization_state = MyLocalizationType(0, 0.0)
        end

        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end 
end

function EKF_predict!(ekf::ObjectEKF, dt)
    F = [1 0 dt 0;
         0 1 0 dt;
         0 0 1 0;
         0 0 0 1]
    ekf.x = F * ekf.x
    ekf.P = F * ekf.P * F' + ekf.Q
end

function EKF_update!(ekf::ObjectEKF, z::Vector{Float64})
    H = [1 0 0 0;
         0 1 0 0]
    y = z - H * ekf.x
    S = H * ekf.P * H' + ekf.R
    K = ekf.P * H' * inv(S)
    ekf.x += K * y
    ekf.P = (I - K * H) * ekf.P
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    # set up stuff
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        latest_localization_state = fetch(localization_state_channel)
        
        # process bounding boxes / run ekf / do what you think is good
        # Simulated detection: assume bounding box center gives object (x, y)
        if !isempty(fresh_cam_meas)
            # Use the most recent camera measurement (e.g., Dict(:x, :y))
            meas = fresh_cam_meas[end]
            z = [meas[:x], meas[:y]]

            # Run EKF prediction and update
            dt = time() - last_time
            last_time = time()
            EKF_predict!(ekf, dt)
            EKF_update!(ekf, z)
        end

        # Create output state using estimated x-position and x-velocity
        estimated_x = ekf.x[1]
        estimated_vx = ekf.x[3]
        perception_state = MyPerceptionType(round(Int, estimated_x), estimated_vx)

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
