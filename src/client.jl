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

function auto_client(host::IPAddr=IPv4(0), port::Int=4444; ego_id::Int=1)
    sock = Sockets.connect(host, port)
    info_msg = deserialize(sock)
    @info "Connected to simulator:" info_msg=info_msg

    serialize(sock, (0.0, 0.0, true))
    @info "Sent initial zero command"

    loc_ch  = Channel{MyLocalizationType}(32)
    perc_ch = Channel{MyPerceptionType}(32)

    errormonitor(@async begin
        while true
            if !isopen(sock)
                @warn "Socket closed; exiting auto_client read loop."
                break
            end

            msg = try
                deserialize(sock)::VehicleSim.MeasurementMessage
            catch e
                @warn "AutoClient read loop: failed to deserialize, retrying..." error=e
                continue
            end

            gt = nothing
            try
                gt = get_gt(msg, ego_id)
            catch e
                continue
            end

            if gt !== nothing
                if(channel_full(loc_ch))
                    take!(loc_ch)
                end

                loc = MyLocalizationType(
                    true,
                    gt.position,
                    VehicleSim.extract_yaw_from_quaternion(gt.orientation),
                    gt.velocity,
                    zeros(6,6),
                    gt.time
                )
                put!(loc_ch, loc)

                if(channel_full(perc_ch))
                    take!(perc_ch)
                end
                detections = Detected_Obj[]
                for m in msg.measurements
                    if m isa GroundTruthMeasurement && m.vehicle_id != ego_id
                        push!(detections, Detected_Obj(
                            m.vehicle_id,
                            (0.0, 0.0, 0.0, 0.0),
                            1.0,
                            "vehicle",
                            m.position[1:2],
                            m.velocity[1:2],
                            zeros(2,2),
                            "ground_truth"
                        ))
                    end
                end
                perc = MyPerceptionType(gt.time, detections)
                put!(perc_ch, perc)
            end
        end
        sleep(0.001)
    end)

    map = VehicleSim.city_map()
    targets = VehicleSim.identify_loading_segments(map)
    @assert !isempty(targets) "No loading segments found in the map!"
    tgt = rand(targets)
    @info "Driving to target segment:" target_segment=tgt

    @async decision_making(loc_ch, perc_ch, map, tgt, sock)
end

function channel_full(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end
