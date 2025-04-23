using Test, StaticArrays, VehicleSim, SU7AVStack, JLD2, Plots

gt_data = JLD2.load("test/message_buff.jld2")["msg_buf"]
pred_data = JLD2.load("test/perc_buff.jld2")["msg_buf"]

@test length(gt_data) == length(pred_data)

errors = Float64[]
timestamps = Float64[]

for (gt_msg, pred_msg) in zip(gt_data, pred_data)
    gt_objs = filter(m -> m isa VehicleSim.GroundTruthMeasurement && m.vehicle_id==2, #data for static vehicle, not for ego
                    msg.measurements)
    pred_objs = pred_msg.measurements

    if isempty(gt_objs) || isempty(pred_objs)
        push!(errors, 0.0)
        push!(timestamps, gt_msg.time)
        continue
    end

    gt_pos = gt_objs[1].position
    pred_pos = pred_objs[1].position

    error = norm(gt_pos[1:2] - pred_pos[1:2])  # XY-plane error
    push!(errors, error)
    push!(timestamps, gt_msg.time)
end

# ----------- Plot Error -----------
plot(timestamps, errors,
    xlabel="Time (s)",
    ylabel="Perception Error (m)",
    title="Perception Error Over Time",
    legend=false,
    marker=:circle,
    lw=2)

savefig("test/perception_error_plot.png")