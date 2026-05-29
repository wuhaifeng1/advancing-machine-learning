// Original source excerpt:
//   /home/wurenche/Co_lio_orgin/src/Co-LRIO/include/outlierRejection.h
//
// Purpose in the report:
//   Evidence for the PCM pairwise consistency and maximum clique stage. This
//   file is not intended to compile; the ROS-free coursework implementation is
//   ../src/pcm_consistency.py.

class OutlierRejection
{
public:
    std::vector<int> selectConsistentLoopClosureIndexes(
        const std::vector<LoopClosure>& loop_closures_i_j,
        const gtsam::Values& trajectories,
        const int& min_clique_size)
    {
        std::vector<int> max_clique_data;
        if (loop_closures_i_j.size() < min_clique_size)
        {
            return max_clique_data;
        }

        auto consistency_matrix =
            computePairwiseConsistentMeasurementsMatrix(loop_closures_i_j, trajectories);
        printConsistencyGraph(consistency_matrix, consistency_matrix_file);

        FMC::CGraphIO gio;
        gio.readGraph(consistency_matrix_file);
        int max_clique_size = FMC::maxCliqueHeu(gio, max_clique_data);

        if (max_clique_size < min_clique_size)
        {
            max_clique_data.clear();
        }

        return max_clique_data;
    }

private:
    double computeConsistencyError(
        const Measurement& z_ij,
        const Measurement& z_lk,
        const Measurement& z_ik,
        const Measurement& z_jl)
    {
        // Compute z_ij + z_jl + z_lk - z_ik in Pose3 and evaluate a
        // Mahalanobis residual.
        Measurement mid_result;
        mid_result.pose = z_ij.pose.compose(z_jl.pose, Ha, Hb);
        mid_result.covariance =
            Ha * z_ij.covariance * Ha.transpose() +
            Hb * z_jl.covariance * Hb.transpose();

        Measurement mid_result2;
        mid_result2.pose = mid_result.pose.compose(z_lk.pose, Hc, Hd);
        mid_result2.covariance =
            Hc * mid_result.covariance * Hc.transpose() +
            Hd * z_lk.covariance * Hd.transpose();

        Measurement result;
        result.pose = z_ik.pose.between(mid_result2.pose, He, Hf);
        result.covariance =
            He * z_ik.covariance * He.transpose() +
            Hf * mid_result2.covariance * Hf.transpose();

        auto consistency_error = gtsam::Pose3::Logmap(result.pose);
        return std::sqrt(consistency_error.transpose() *
            result.covariance.inverse() * consistency_error);
    }

    Eigen::MatrixXi computePairwiseConsistentMeasurementsMatrix(
        const std::vector<LoopClosure>& robota_robotb_loop_closures,
        const gtsam::Values& trajectories)
    {
        Eigen::MatrixXi consistency_matrix(
            robota_robotb_loop_closures.size(),
            robota_robotb_loop_closures.size());
        consistency_matrix.setZero();

        for (auto iter = 0; iter < robota_robotb_loop_closures.size(); iter++)
        {
            auto i = robota_robotb_loop_closures[iter].symbol0;
            auto k = robota_robotb_loop_closures[iter].symbol1;
            auto z_ik = robota_robotb_loop_closures[iter].measurement;
            auto pose_i = trajectories.at<gtsam::Pose3>(i);
            auto pose_k = trajectories.at<gtsam::Pose3>(k);

            for (auto iter2 = 0; iter2 < robota_robotb_loop_closures.size(); iter2++)
            {
                auto j = robota_robotb_loop_closures[iter2].symbol0;
                auto l = robota_robotb_loop_closures[iter2].symbol1;
                auto z_jl = robota_robotb_loop_closures[iter2].measurement;
                auto pose_j = trajectories.at<gtsam::Pose3>(j);
                auto pose_l = trajectories.at<gtsam::Pose3>(l);

                Measurement z_ij, z_lk;
                z_ij.pose = pose_i.between(pose_j, Ha, Hb);
                z_ij.covariance =
                    Ha * odometry_covariance * Ha.transpose() +
                    Hb * odometry_covariance * Hb.transpose();
                z_lk.pose = pose_l.between(pose_k, Hc, Hd);
                z_lk.covariance =
                    Hc * odometry_covariance * Hc.transpose() +
                    Hd * odometry_covariance * Hd.transpose();

                auto dis = computeConsistencyError(z_ij, z_lk, z_ik, z_jl);
                consistency_matrix(iter, iter2) = dis < pcm_threshold_ ? 1 : 0;
            }
        }

        return consistency_matrix;
    }
};
