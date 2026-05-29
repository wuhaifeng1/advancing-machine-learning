// Original source excerpt:
//   /home/wurenche/Co_lio_orgin/src/Co-LRIO/src/lidarOdometry.cpp
//
// Purpose in the report:
//   Evidence for the local GICP loop-verification stage. This file is not
//   intended to compile; the ROS-free coursework implementation is
//   ../src/gicp_evidence.py.

// Stage-one loop message handling:
//   noise=999 requests the source keyframe from robot0.
//   noise=888 carries the source keyframe to robot1 for local verification.
void LidarOdometry::loopClosureHandler(
    const std::shared_ptr<rclcpp::SerializedMessage> serialized_msg)
{
    // ...
    if (int(msg.noise) == 999 && msg.robot0 == params.id_)
    {
        // Publish source keyframe point cloud back to the verifier.
        loop_closure_msg->robot0 = msg.robot0;
        loop_closure_msg->key0 = msg.key0;
        loop_closure_msg->robot1 = msg.robot1;
        loop_closure_msg->key1 = msg.key1;
        loop_closure_msg->yaw_diff = msg.yaw_diff;
        loop_closure_msg->noise = 888.0;
        pcl::toROSMsg(*keyframe, loop_closure_msg->frame);
    }
    else if (int(msg.noise) == 888 && msg.robot1 == params.id_)
    {
        LoopClosure lc(msg.robot0, msg.key0, msg.robot1,
                       msg.key1, msg.yaw_diff, msg.frame);
        loop_closure_candidates.emplace_back(lc);
    }
}

std::pair<int, LoopClosure> LidarOdometry::calculateTransformation(
    const LoopClosure& lc_candidate)
{
    // Initial yaw from Scan Context sector shift.
    double initial_yaw_ = 0.0;
    if (loop_robot0 != loop_robot1)
    {
        if (params.descriptor_type_ == DescriptorType::ScanContext)
        {
            initial_yaw_ = (init_yaw) * 2 * M_PI / 60.0;
        }
        if (initial_yaw_ > M_PI)
        {
            initial_yaw_ -= 2 * M_PI;
        }

        auto init_pose = keyposes_snapshot[loop_key1];
        loop_pose0 = gtsam::Pose3(
            gtsam::Rot3::RzRyRx(init_pose.rotation().roll(),
                                init_pose.rotation().pitch(),
                                init_pose.rotation().yaw() + initial_yaw_),
            gtsam::Point3(init_pose.translation().x(),
                          init_pose.translation().y(),
                          init_pose.translation().z()));

        *scan_cloud = *lc_candidate.frame;
    }

    // Source is the candidate scan. Target is the local submap around key1.
    loop_closure_voxelgrid.setInputCloud(scan_cloud);
    loop_closure_voxelgrid.filter(*scan_cloud_filtered);
    loop_pose1 = keyposes_snapshot[loop_key1];
    *map_cloud = *surroundingMap(loop_key1, params.history_keyframe_search_num_,
                                 keyframes_snapshot, keyposes_snapshot);
    loop_closure_voxelgrid.setInputCloud(map_cloud);
    loop_closure_voxelgrid.filter(*map_cloud_filtered);

    // Cross-robot loop uses inter-robot GICP settings.
    if (loop_robot0 == loop_robot1)
    {
        gicp_loop.setMaxCorrespondenceDistance(params.intra_icp_max_correspondence_distance_);
        gicp_loop.setMaximumIterations(params.intra_icp_iterations_time_);
    }
    else
    {
        gicp_loop.setMaxCorrespondenceDistance(params.inter_icp_max_correspondence_distance_);
        gicp_loop.setMaximumIterations(params.inter_icp_iterations_time_);
    }

    gicp_loop.setInputSource(scan_cloud_filtered);
    gicp_loop.setInputTarget(map_cloud_filtered);
    gicp_loop.align(*unused_result, loop_pose0.matrix().cast<float>());

    const auto fitness = gicp_loop.getFitnessScore();
    if (gicp_loop.hasConverged() == false ||
        fitness > params.fitness_score_threshold_)
    {
        saveLoopDebugClouds(lc_result, "rejected", initial_yaw_,
            loop_pose0, failed_pose_from, loop_pose1,
            failed_pose_from.between(loop_pose1), target_keyframe_cloud,
            gicp_loop.hasConverged(), fitness,
            scan_cloud_filtered->size(), map_cloud_filtered->size());
        return make_pair(0, lc_result);
    }

    const auto pose_from = gtsam::Pose3(gicp_loop.getFinalTransformation().cast<double>());
    const auto pose_to = loop_pose1;

    Measurement measurement;
    measurement.pose = pose_from.between(pose_to);
    measurement.covariance = loop_noise->covariance();
    lc_result.measurement = measurement;
    lc_result.fitness_score = fitness;

    saveLoopDebugClouds(lc_result, "passed", initial_yaw_,
        loop_pose0, pose_from, pose_to, measurement.pose,
        target_keyframe_cloud, true, fitness,
        scan_cloud_filtered->size(), map_cloud_filtered->size());

    return make_pair(1, lc_result);
}

// Compact evidence exported by Co-LRIO and reused by the coursework:
//   source_raw.pcd
//   target_submap.pcd
//   source_init_in_target.pcd
//   source_gicp_in_target.pcd
//   loop_matches.csv with status, init_yaw, converged, fitness, and point counts.
