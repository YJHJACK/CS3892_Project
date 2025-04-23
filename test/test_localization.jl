using Serialization
using LinearAlgebra
using StaticArrays
using Base.Threads
using Plots
using Random
using JLD2

include("../src/example_project.jl")

function test_localization()
    println("Loading message buffer from message_buff.jld2 …")
    msg_buf = load("../../VehicleSim/message_buff.jld2", "msg_buf")
    println("Loaded $(length(msg_buf)) messages.")

    gps_ch = Channel{GPSMeasurement}(32)
    imu_ch = Channel{IMUMeasurement}(32)
    loc_ch = Channel{MyLocalizationType}(1)
    sd_ch  = Channel{Bool}(1)

    # initialize with invalid state
    put!(loc_ch, MyLocalizationType(
        false,
        SVector{3,Float64}(0.0,0.0,0.0),
        0.0,
        SVector{3,Float64}(0.0,0.0,0.0),
        I(13),
        0.0
    ))

    println("[Tester] Launching EKF localization thread.")
    @async localize(gps_ch, imu_ch, loc_ch, sd_ch; dt_default=0.1)

    times  = Float64[]
    errors = Float64[]

    for (i, msg) in enumerate(msg_buf)
        # clear outdated sensor buffers
        while isready(gps_ch); take!(gps_ch); end
        while isready(imu_ch); take!(imu_ch); end

        # feed new measurements
        for meas in msg.measurements
            if meas isa GPSMeasurement
                put!(gps_ch, meas)
            elseif meas isa IMUMeasurement
                put!(imu_ch, meas)
            end
        end

        # find ground truth for ego vehicle
        idx = findfirst(m -> (m isa GroundTruthMeasurement && m.vehicle_id == 1),
                        msg.measurements)
        if idx === nothing
            @warn "[Tester] msg $i has no GT → skip"
            continue
        end
        gt = msg.measurements[idx]

        # allow EKF thread to process
        sleep(0.01)

        # only take if EKF has produced an output
        if !isready(loc_ch)
            @warn "[Tester] msg $i 尚无 EKF 输出 → skip"
            continue
        end
        loc = take!(loc_ch)

        # compute error
        dx = loc.position[1] - gt.position[1]
        dy = loc.position[2] - gt.position[2]
        err = sqrt(dx^2 + dy^2)

        push!(times, gt.time)
        push!(errors, err)

        println("[Tester] msg $i | t=$(round(gt.time, digits=2))s | error=$(round(err, digits=3)) m")
    end

    println("[Tester] All messages processed → sending shutdown.")
    put!(sd_ch, true)
    sleep(0.5)

    println("[Tester] Plotting error curve …")
    plt = plot(times, errors,
        xlabel="Time (s)",
        ylabel="XY Position Error (m)",
        title="EKF Localization XY Error",
        marker=:circle,
        legend=false)
    savefig(plt, "ekf_localization_error.png")
    println("[Tester] Done. Saved ekf_localization_error.png")
end

test_localization()
