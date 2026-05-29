// Original source excerpts:
//   /home/wurenche/Co_lio_orgin/src/Co-LRIO/include/scanContextDescriptor.h
//   /home/wurenche/Co_lio_orgin/src/Co-LRIO/src/scanContextDescriptor.cpp
//
// Purpose in the report:
//   Evidence for the Scan Context candidate recall stage. This file is not
//   intended to compile; the ROS-free coursework implementation is
//   ../src/scan_context.py.

class ScanContextDescriptor : public ScanDescriptor
{
public:
    ScanContextDescriptor(
        int ring_num,
        int sector_num,
        int candidates_num,
        float distance_threshold,
        float max_radius,
        int exclude_recent_num,
        std::string directory);

    std::vector<float> makeDescriptor(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr scan);

    void saveDescriptorAndKey(
        const std::vector<float> descriptor,
        const int8_t& robot,
        const int& index);

    std::vector<std::tuple<int8_t, int, int>> detectLoopClosure(
        const int8_t& cur_robot,
        const int& cur_ptr);

private:
    Eigen::MatrixXf makeScancontext(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr scan_down);

    Eigen::MatrixXf makeRingkeyFromScancontext(
        const Eigen::MatrixXf& desc);

    Eigen::MatrixXf makeSectorkeyFromScancontext(
        const Eigen::MatrixXf& desc);

    std::pair<float, int> distanceBtnScanContext(
        const Eigen::MatrixXf& sc1,
        const Eigen::MatrixXf& sc2);
};

// Ring-sector descriptor: each bin stores max height shifted by +1.0.
Eigen::MatrixXf ScanContextDescriptor::makeScancontext(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr scan_down)
{
    Eigen::MatrixXf sc = Eigen::MatrixXf::Zero(pc_ring_num_, pc_sector_num_);

    float azim_angle, azim_range;
    int ring_idx, sctor_idx;
    for (auto pt : scan_down->points)
    {
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        azim_angle = xy2theta(pt.x, pt.y);
        if (azim_range > pc_max_radius_)
        {
            continue;
        }

        ring_idx = max(min(pc_ring_num_,
            int(ceil((azim_range / pc_max_radius_) * pc_ring_num_))), 1) - 1;
        sctor_idx = max(min(pc_sector_num_,
            int(ceil((azim_angle / 360.0) * pc_sector_num_))), 1) - 1;

        if (sc(ring_idx, sctor_idx) < pt.z + 1.0)
        {
            sc(ring_idx, sctor_idx) = pt.z + 1.0;
        }
    }

    return sc;
}

// Ring key is rowwise mean; sector key is columnwise mean.
Eigen::MatrixXf ScanContextDescriptor::makeRingkeyFromScancontext(
    const Eigen::MatrixXf& desc)
{
    Eigen::MatrixXf invariant_key(desc.rows(), 1);
    for(int row_idx = 0; row_idx < desc.rows(); row_idx++)
    {
        Eigen::MatrixXf curr_row = desc.row(row_idx);
        invariant_key(row_idx, 0) = curr_row.mean();
    }
    return invariant_key;
}

Eigen::MatrixXf ScanContextDescriptor::makeSectorkeyFromScancontext(
    const Eigen::MatrixXf& desc)
{
    Eigen::MatrixXf variant_key(1, desc.cols());
    for(int col_idx = 0; col_idx < desc.cols(); col_idx++)
    {
        Eigen::MatrixXf curr_col = desc.col(col_idx);
        variant_key(0, col_idx) = curr_col.mean();
    }
    return variant_key;
}

// Descriptor distance after sector shift alignment.
std::pair<float, int> ScanContextDescriptor::distanceBtnScanContext(
    const Eigen::MatrixXf& sc1,
    const Eigen::MatrixXf& sc2)
{
    Eigen::MatrixXf vkey_sc1 = makeSectorkeyFromScancontext(sc1);
    Eigen::MatrixXf vkey_sc2 = makeSectorkeyFromScancontext(sc2);
    int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);

    int SEARCH_RADIUS = round(0.5 * 0.1 * sc1.cols());
    std::vector<int> shift_idx_search_space { argmin_vkey_shift };
    for(int ii = 1; ii < SEARCH_RADIUS + 1; ii++)
    {
        shift_idx_search_space.push_back((argmin_vkey_shift + ii + sc1.cols()) % sc1.cols());
        shift_idx_search_space.push_back((argmin_vkey_shift - ii + sc1.cols()) % sc1.cols());
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

    int argmin_shift = 0;
    float min_sc_dist = 10000000;
    for(int num_shift: shift_idx_search_space)
    {
        Eigen::MatrixXf sc2_shifted = circshift(sc2, num_shift);
        float cur_sc_dist = distDirectSC(sc1, sc2_shifted);
        if(cur_sc_dist < min_sc_dist)
        {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }

    return make_pair(min_sc_dist, argmin_shift);
}

// Candidate retrieval: KNN on ring key, then full Scan Context distance.
std::vector<std::tuple<int8_t, int, int>> ScanContextDescriptor::detectLoopClosure(
    const int8_t& cur_robot,
    const int& cur_key)
{
    auto ringkey = scan_context_ringkey.at(cur_robot).col(cur_key);
    auto scan_context = scan_contexts.at(cur_robot).at(cur_key);

    // Build a history database. For the same robot, recent frames are excluded;
    // for other robots, all stored descriptors are searchable.
    // ...

    kdTree = Nabo::NNSearchF::createKDTreeLinearHeap(
        new_scan_context_ringkey, pc_ring_num_);
    kdTree->knn(ringkey, indices, distance, candidates_num_);

    for (const auto& indice : indices_vec)
    {
        auto scan_context_candidate = new_scan_contexts.at(indice);
        auto sc_dist_result = distanceBtnScanContext(scan_context, scan_context_candidate);
        float candidate_dis = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if(candidate_dis < distance_threshold_)
        {
            // Keep the best candidate for each robot.
            // Return: candidate_robot, candidate_key, yaw_shift.
        }
    }
}
