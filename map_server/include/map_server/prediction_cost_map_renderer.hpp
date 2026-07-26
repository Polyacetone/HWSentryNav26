#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include <map_server/object_tracker.hpp>
#include <map_server/utils.hpp>

namespace map_server {

class PredictionCostMapRenderer {
public:
    PredictionCostMapRenderer(
        int map_width,
        int map_height,
        int prediction_steps,
        const map_utils::MapInflationParams& inflation_params
    );

    std::vector<cv::Mat> render(
        const ObjectTracker::PredictionResult& prediction_result
    ) const;

private:
    cv::Mat inflate_local_footprint(const cv::Mat& footprint_mask) const;

    int map_width_;
    int map_height_;
    int prediction_steps_;
    int cutoff_radius_px_;
    map_utils::MapInflationParams inflation_params_;
};

} // namespace map_server
