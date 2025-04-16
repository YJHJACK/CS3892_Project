using Sockets
using StaticArrays
using Logging
using VehicleSim

struct VehicleCommand
    steering_angle::Float64
    velocity::Float64
    controlled::Bool
end

function get_gt(msg::VehicleSim.MeasurementMessage, ego_id::Int)
    for m in msg.measurements
        if m isa VehicleSim.GroundTruthMeasurement && m.vehicle_id == ego_id
            return m
        end
    end
    error("No GroundTruthMeasurement for vehicle $ego_id")
end

function get_c()
    c = 'x'
    try
        ret = ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, true)
        ret == 0 || error("unable to switch to raw mode")
        c = read(stdin, Char)
        ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, false)
    catch e
    end
    c
end

function keyboard_client(host::IPAddr=IPv4(0), port=4444; v_step = 1.0, s_step = π/10)
    socket = Sockets.connect(host, port)
    (peer_host, peer_port) = getpeername(socket)
    msg = deserialize(socket) # Visualization info
    @info msg

    @async while isopen(socket)
        sleep(0.001)
        state_msg = deserialize(socket)
        measurements = state_msg.measurements
        num_cam = 0
        num_imu = 0
        num_gps = 0
        num_gt = 0
        for meas in measurements
            if meas isa GroundTruthMeasurement
                num_gt += 1
            elseif meas isa CameraMeasurement
                num_cam += 1
            elseif meas isa IMUMeasurement
                num_imu += 1
            elseif meas isa GPSMeasurement
                num_gps += 1
            end
        end
    end
    
    target_velocity = 0.0
    steering_angle = 0.0
    controlled = true
    
    client_info_string = 
        "********************
      Keyboard Control (manual mode)
      ********************
        -Press 'q' at any time to terminate vehicle.
        -Press 'i' to increase vehicle speed.
        -Press 'k' to decrease vehicle speed.
        -Press 'j' to increase steering angle (turn left).
        -Press 'l' to decrease steering angle (turn right)."
    @info client_info_string
    while controlled && isopen(socket)
        key = get_c()
        if key == 'q'
            # terminate vehicle
            controlled = false
            target_velocity = 0.0
            steering_angle = 0.0
            @info "Terminating Keyboard Client."
        elseif key == 'i'
            # increase target velocity
            target_velocity += v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'k'
            # decrease forward force
            target_velocity -= v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'j'
            # increase steering angle
            steering_angle += s_step
            @info "Target steering angle: $steering_angle"
        elseif key == 'l'
            # decrease steering angle
            steering_angle -= s_step
            @info "Target steering angle: $steering_angle"
        end
        cmd = (steering_angle, target_velocity, controlled)
        serialize(socket, cmd)
    end
end

function example_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = training_map()
    (; chevy_base) = load_mechanism()

    @async while isopen(socket)
        state_msg = deserialize(socket)
    end
   
    shutdown = false
    persist = true
    while isopen(socket)
        position = state_msg.q[5:7]
        @info position
        if norm(position) >= 100
            shutdown = true
            persist = false
        end
        cmd = (0.0, 2.5, persist, shutdown)
        serialize(socket, cmd) 
    end

end

function auto_client(host::IPAddr = IPv4(0), port::Int = 4444; ego_id::Int = 1)
    # Connect to the simulator server.
    sock = Sockets.connect(host, port)
    info_msg = deserialize(sock)
    @info "Connected to simulator:" info_msg=info_msg

    # Send an initial zero command.
    serialize(sock, (0.0, 0.0, true))
    @info "Sent initial zero command"

    # Create channels for incoming sensor measurements.
    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)

    # Create channels for output states:
    # loc_state_channel will receive the EKF localization output.
    # perc_state_channel will receive the perception state.
    loc_state_channel = Channel{MyLocalizationType}(32)
    perc_state_channel = Channel{MyPerceptionType}(32)

    # Launch an asynchronous task to continuously read MeasurementMessage from the socket 
    # and distribute sensor measurements into the corresponding channels.
    @async begin
        while isopen(sock)
            sleep(0.001)
            msg = try
                deserialize(sock)::VehicleSim.MeasurementMessage
            catch e
                @warn "AutoClient read loop: failed to deserialize, retrying..." error=e
                continue
            end
            for meas in msg.measurements
                if meas isa GPSMeasurement
                    put!(gps_channel, meas)
                elseif meas isa IMUMeasurement
                    put!(imu_channel, meas)
                elseif meas isa CameraMeasurement
                    put!(cam_channel, meas)
                end
                # Note: Do not use ground truth measurements for localization/perception.
            end
        end
    end

    # Create a shutdown channel for the localization and perception modules.
    shutdown_channel = Channel{Bool}(1)

    # Launch the EKF localization module.
    # 'localize' reads from gps_channel and imu_channel and outputs state into loc_state_channel.
    @async localize(gps_channel, imu_channel, loc_state_channel, shutdown_channel)

    # Launch the perception module.
    # 'perception' reads from cam_channel (and can use loc_state_channel) to update perc_state_channel.
    # Adjust parameters as needed (e.g. if perception requires ekf or cnn_model, supply them accordingly).
    @async perception(cam_channel, loc_state_channel, perc_state_channel, shutdown_channel)

    # Obtain the map and select a target road segment.
    local map_segments = VehicleSim.city_map()
    local targets = VehicleSim.identify_loading_segments(map_segments)
    @assert !isempty(targets) "No loading segments found in the map!"
    local tgt = rand(targets)
    @info "Driving to target segment:" target_segment=tgt

    # Launch the decision making module,
    # which will fetch the latest localization state from loc_state_channel (from EKF)
    # and perception state from perc_state_channel to generate control commands.
    @async decision_making(loc_state_channel, perc_state_channel, map_segments, tgt, sock)
end

function channel_full(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end
