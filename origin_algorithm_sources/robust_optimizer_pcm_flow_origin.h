// Original source excerpt:
//   /home/wurenche/Co_lio_orgin/src/Co-LRIO/include/robustOptimizer.h
//
// Purpose in the report:
//   Evidence for how PCM-filtered inter-robot loops become final-used backend
//   constraints. This file is not intended to compile; the ROS-free coursework
//   implementation is ../src/pcm_consistency.py and ../src/loop_matching_analysis.py.

int pendingInterRobotLoopThreshold() const
{
    return std::max(3, loop_num_threshold_);
}

void storeInterRobotLoopClosure(const LoopClosure& lc)
{
    inter_robot_loop_closures[lc.robot0].emplace_back(lc);
    adjacency_matrix(lc.robot0, lc.robot1) += 1;
    adjacency_matrix(lc.robot1, lc.robot0) += 1;
}

void addInterRobotLoopFactor(const LoopClosure& loop)
{
    auto this_noise_model =
        noiseModel::Diagonal::Variances(loop.measurement.covariance.diagonal());
    graph->addExpressionFactor(
        between(Pose3_(loop.symbol0), Pose3_(loop.symbol1)),
        loop.measurement.pose,
        this_noise_model);

    // Original log marker for final-used inter-robot loop:
    // "add inter-robot loop~ [a*][b*], fitness=..."
}

bool acceptInterRobotLoopClosures(
    const std::vector<LoopClosure>& accepted_loop_closures,
    map<gtsam::Symbol, gtsam::Symbol>& indexes)
{
    if (accepted_loop_closures.empty())
    {
        return false;
    }

    for (const auto& accepted_loop : accepted_loop_closures)
    {
        storeInterRobotLoopClosure(accepted_loop);
        indexes.emplace(make_pair(accepted_loop.symbol0, accepted_loop.symbol1));
        indexes.emplace(make_pair(accepted_loop.symbol1, accepted_loop.symbol0));
    }

    if (!connectRobotsIfNeeded(accepted_loop_closures.front()))
    {
        return false;
    }

    for (const auto& accepted_loop : accepted_loop_closures)
    {
        addInterRobotLoopFactor(accepted_loop);
    }

    return true;
}

bool handleInterRobotLoopWithPCM(const SharedFactor& sf)
{
    LoopClosure lc(sf.index_from, sf.index_to, sf.noise,
        Measurement(sf.pose_to, noise_model->covariance()));

    if (use_pcm_)
    {
        addPendingInterRobotLoopClosure(lc);
        auto pending_pair_loop_closures = getPendingInterRobotLoopClosures(lc);
        const auto min_clique_size = pendingInterRobotLoopThreshold();

        if (pending_pair_loop_closures.size() < min_clique_size)
        {
            // Pending loops are delayed until enough candidates exist.
            return false;
        }

        auto stored_pair_loop_closures = getStoredInterRobotLoopClosures(lc);
        vector<LoopClosure> loops_for_pcm = stored_pair_loop_closures;
        loops_for_pcm.insert(loops_for_pcm.end(),
            pending_pair_loop_closures.begin(),
            pending_pair_loop_closures.end());

        auto trajectories = buildPairwiseLocalTrajectories(lc.robot0, lc.robot1);
        auto inlier_indexes = outlier_reject->selectConsistentLoopClosureIndexes(
            loops_for_pcm, trajectories, min_clique_size);

        if (inlier_indexes.empty())
        {
            return false;
        }

        vector<LoopClosure> accepted_loop_closures;
        for (const auto& inlier_index : inlier_indexes)
        {
            if (inlier_index >= stored_pair_loop_closures.size())
            {
                accepted_loop_closures.emplace_back(loops_for_pcm.at(inlier_index));
            }
        }
        clearPendingInterRobotLoopClosures(lc.robot0, lc.robot1);

        return acceptInterRobotLoopClosures(accepted_loop_closures, indexes);
    }

    vector<LoopClosure> accepted_loop_closures;
    accepted_loop_closures.emplace_back(lc);
    return acceptInterRobotLoopClosures(accepted_loop_closures, indexes);
}
