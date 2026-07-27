#include <map_server/prediction_cost_map_renderer.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace map_server {

namespace {

void max_blit(const cv::Mat& source, cv::Mat& destination, const cv::Point& top_left) {
    CV_Assert(source.type() == CV_8UC1);
    CV_Assert(destination.type() == CV_8UC1);

    const cv::Rect destination_bounds(0, 0, destination.cols, destination.rows);
    const cv::Rect requested_region(top_left.x, top_left.y, source.cols, source.rows);
    const cv::Rect destination_region = requested_region & destination_bounds;
    if (destination_region.empty()) return;

    const cv::Rect source_region(
        destination_region.x - requested_region.x,
        destination_region.y - requested_region.y,
        destination_region.width,
        destination_region.height
    );
    const cv::Mat source_roi = source(source_region);
    cv::Mat destination_roi = destination(destination_region);
    cv::max(source_roi, destination_roi, destination_roi);
}

} // anonymous namespace

PredictionCostMapRenderer::PredictionCostMapRenderer(
    const int map_width,
    const int map_height,
    const int prediction_steps,
    const map_utils::MapInflationParams& inflation_params
):
    map_width_(map_width),
    map_height_(map_height),
    prediction_steps_(prediction_steps),
    cutoff_radius_px_(0),
    inflation_params_(inflation_params) {
    if (map_width_ <= 0 || map_height_ <= 0) {
        throw std::invalid_argument("prediction renderer map dimensions must be positive");
    }
    if (prediction_steps_ < 0) {
        throw std::invalid_argument("prediction renderer step count must be non-negative");
    }
    if (!std::isfinite(inflation_params_.resolution)
        || inflation_params_.resolution <= 0.0) {
        throw std::invalid_argument("prediction renderer resolution must be positive");
    }
    cutoff_radius_px_ = std::min(
        map_utils::enclosing_radius_cells(
            inflation_params_.cutoff_radius_m,
            inflation_params_.resolution
        ),
        std::max(map_width_, map_height_)
    );
}

std::vector<cv::Mat> PredictionCostMapRenderer::render(
    const ObjectTracker::PredictionResult& prediction_result
) const {
    if (prediction_result.static_fallback_mask.type() != CV_8UC1
        || prediction_result.static_fallback_mask.cols != map_width_
        || prediction_result.static_fallback_mask.rows != map_height_) {
        throw std::invalid_argument("prediction fallback mask does not match the map");
    }

    const cv::Mat static_cost_map = map_utils::inflate_cost_map_bounded(
        prediction_result.static_fallback_mask,
        inflation_params_
    );
    std::vector<cv::Mat> future_cost_maps(static_cast<size_t>(prediction_steps_));
    for (cv::Mat& cost_map : future_cost_maps) {
        cost_map = static_cost_map.clone();
    }

    for (const ObjectTracker::MotionPrediction& prediction
         : prediction_result.motion_predictions) {
        if (prediction.footprint_mask.type() != CV_8UC1) {
            throw std::invalid_argument("motion prediction footprint must be CV_8UC1");
        }
        if (prediction.future_centroids_px.size() != future_cost_maps.size()) {
            throw std::logic_error("motion prediction length does not match prediction_steps");
        }
        if (cv::countNonZero(prediction.footprint_mask) == 0) continue;

        const cv::Mat inflated_footprint = inflate_local_footprint(
            prediction.footprint_mask
        );
        // inflated_footprint 相对质心的左上角偏移：轮廓半径 + 膨胀光环半径。
        const int offset_x = prediction.footprint_mask.cols / 2 + cutoff_radius_px_;
        const int offset_y = prediction.footprint_mask.rows / 2 + cutoff_radius_px_;

        // 完整轮廓只膨胀一次，越界由 max_blit 裁剪，使出图轮廓的光环仍能写入图内。
        for (size_t step = 0; step < future_cost_maps.size(); ++step) {
            const Eigen::Vector2i& centroid = prediction.future_centroids_px[step];
            max_blit(
                inflated_footprint,
                future_cost_maps[step],
                {centroid.x() - offset_x, centroid.y() - offset_y}
            );
        }
    }
    return future_cost_maps;
}

cv::Mat PredictionCostMapRenderer::inflate_local_footprint(
    const cv::Mat& footprint_mask
) const {
    cv::Mat padded_mask;
    cv::copyMakeBorder(
        footprint_mask,
        padded_mask,
        cutoff_radius_px_,
        cutoff_radius_px_,
        cutoff_radius_px_,
        cutoff_radius_px_,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    return map_utils::inflate_cost_map(padded_mask, inflation_params_);
}

} // namespace map_server
