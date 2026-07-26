#include <map_server/object_tracker.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace map_server {

// ═══════════════ Hungarian Algorithm (Kuhn-Munkres, O(n³)) ═══════════════

static std::vector<int>
hungarian_solve(const std::vector<std::vector<double>>& cost, int n_rows, int n_cols, double gate) {
    if (n_rows == 0 || n_cols == 0) return std::vector<int>(static_cast<size_t>(n_rows), -1);

    const int n = std::max(n_rows, n_cols);
    const double BIG = gate + 1.0;
    const auto idx = [](int x) noexcept { return static_cast<size_t>(x); };

    // 1-indexed cost matrix, padded with BIG
    std::vector<std::vector<double>> a(
        idx(n + 1),
        std::vector<double>(idx(n + 1), BIG)
    );
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            a[idx(i + 1)][idx(j + 1)] =
                std::min(cost[idx(i)][idx(j)], BIG);
        }
    }

    std::vector<double> u(idx(n + 1), 0.0);
    std::vector<double> v(idx(n + 1), 0.0);
    std::vector<int> p(idx(n + 1), 0);
    std::vector<int> way(idx(n + 1), 0);

    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(idx(n + 1), 1e18);
        std::vector<bool> used(idx(n + 1), false);

        do {
            used[idx(j0)] = true;
            int i0 = p[idx(j0)];
            int j1 = 0;
            double delta = 1e18;
            for (int j = 1; j <= n; j++) {
                if (used[idx(j)]) continue;
                double cur = a[idx(i0)][idx(j)] - u[idx(i0)] - v[idx(j)];
                if (cur < minv[idx(j)]) {
                    minv[idx(j)] = cur;
                    way[idx(j)] = j0;
                }
                if (minv[idx(j)] < delta) {
                    delta = minv[idx(j)];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; j++) {
                if (used[idx(j)]) {
                    u[idx(p[idx(j)])] += delta;
                    v[idx(j)] -= delta;
                } else {
                    minv[idx(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[idx(j0)] != 0);

        do {
            int j1 = way[idx(j0)];
            p[idx(j0)] = p[idx(j1)];
            j0 = j1;
        } while (j0);
    }

    // Extract assignment, filter by gate
    std::vector<int> result(idx(n_rows), -1);
    for (int j = 1; j <= n; j++) {
        int row = p[idx(j)] - 1;
        int col = j - 1;
        if (row >= 0 && row < n_rows && col >= 0 && col < n_cols) {
            if (cost[idx(row)][idx(col)] < gate) {
                result[idx(row)] = col;
            }
        }
    }
    return result;
}

// ═══════════════ Constructor ═══════════════

ObjectTracker::ObjectTracker(int width, int height, double resolution, const ObjectTrackerParams& params):
    width_(width),
    height_(height),
    resolution_(resolution),
    params_(params) {}

// ═══════════════ Main Update ═══════════════

ObjectTracker::PredictionResult ObjectTracker::update(const cv::Mat& obstacle_mask, double dt) {
    const cv::Mat clean_mask = preprocess_mask(obstacle_mask);
    const std::vector<Detection> detections = detect(clean_mask);

    // KF predict for all existing tracks
    for (auto& track: tracks_) {
        kf_predict(track, dt);
    }

    // Data association (Hungarian)
    const std::vector<int> assignment = associate(detections);

    // Track management: update matched, create new, delete lost
    const std::vector<bool> detections_with_motion_prediction = update_tracks(detections, assignment);

    PredictionResult result;
    result.static_fallback_mask = build_static_fallback_mask(
        obstacle_mask,
        detections,
        detections_with_motion_prediction
    );
    result.motion_predictions = build_motion_predictions();
    return result;
}

// ═══════════════ Preprocessing ═══════════════

cv::Mat ObjectTracker::preprocess_mask(const cv::Mat& mask) const {
    if (params_.morph_close_kernel_size <= 1) return mask;
    cv::Mat cleaned;
    const int ks = params_.morph_close_kernel_size | 1; // ensure odd
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {ks, ks});
    cv::morphologyEx(mask, cleaned, cv::MORPH_CLOSE, kernel);
    return cleaned;
}

// ═══════════════ Detection (Connected Component Analysis) ═══════════════

std::vector<ObjectTracker::Detection> ObjectTracker::detect(const cv::Mat& mask) const {
    std::vector<Detection> detections;

    cv::Mat labels, stats, centroids;
    const int n_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    const int half = params_.local_grid_size / 2;
    const int grid_sz = params_.local_grid_size;

    // Label 0 is background
    for (int label = 1; label < n_labels; label++) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < params_.min_blob_area) continue;

        // Centroid (pixel, sub-pixel precision)
        const double cx_px = centroids.at<double>(label, 0);
        const double cy_px = centroids.at<double>(label, 1);

        Detection det;
        det.centroid_m = {cx_px * resolution_, cy_px * resolution_};

        // Extract local grid relative to rounded centroid
        const int cx_int = static_cast<int>(std::round(cx_px));
        const int cy_int = static_cast<int>(std::round(cy_px));
        det.centroid_px = {cx_int, cy_int};
        det.local_grid = cv::Mat::zeros(grid_sz, grid_sz, CV_32FC1);

        const int bb_left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int bb_top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int bb_w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int bb_h = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        for (int gy = bb_top; gy < bb_top + bb_h; gy++) {
            const int* label_row = labels.ptr<int>(gy);
            for (int gx = bb_left; gx < bb_left + bb_w; gx++) {
                if (label_row[gx] != label) continue;
                const int lx = gx - cx_int + half;
                const int ly = gy - cy_int + half;
                if (lx >= 0 && lx < grid_sz && ly >= 0 && ly < grid_sz) {
                    det.local_grid.at<float>(ly, lx) = 1.0f;
                }
            }
        }

        detections.push_back(std::move(det));
    }
    return detections;
}

// ═══════════════ Kalman Filter: Constant Velocity Model ═══════════════

void ObjectTracker::kf_predict(Track& track, double dt) const {
    // State transition
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;
    F(1, 3) = dt;

    // Process noise: discrete white-noise acceleration model
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;
    const double q = params_.process_noise_std * params_.process_noise_std;

    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    Q(0, 0) = dt4 / 4.0;
    Q(0, 2) = dt3 / 2.0;
    Q(1, 1) = dt4 / 4.0;
    Q(1, 3) = dt3 / 2.0;
    Q(2, 0) = dt3 / 2.0;
    Q(2, 2) = dt2;
    Q(3, 1) = dt3 / 2.0;
    Q(3, 3) = dt2;
    Q *= q;

    track.x = F * track.x;
    track.P = F * track.P * F.transpose() + Q;
}

void ObjectTracker::kf_update(Track& track, const Eigen::Vector2d& z) const {
    // Observation model: H = [I₂ 0₂]
    Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;

    const double rm = params_.measurement_noise_std * params_.measurement_noise_std;
    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * rm;

    // Innovation
    const Eigen::Vector2d y = z - H * track.x;
    const Eigen::Matrix2d S = H * track.P * H.transpose() + R;

    // Kalman gain
    const Eigen::Matrix<double, 4, 2> K = track.P * H.transpose() * S.inverse();

    // State update
    track.x += K * y;

    // Covariance update (Joseph form for numerical stability)
    const Eigen::Matrix4d I_KH = Eigen::Matrix4d::Identity() - K * H;
    track.P = I_KH * track.P * I_KH.transpose() + K * R * K.transpose();
}

// ═══════════════ Data Association ═══════════════

std::vector<int> ObjectTracker::associate(const std::vector<Detection>& detections) const {
    const int n_tracks = static_cast<int>(tracks_.size());
    const int n_dets = static_cast<int>(detections.size());

    if (n_tracks == 0 || n_dets == 0) {
        return std::vector<int>(static_cast<size_t>(n_tracks), -1);
    }

    // Cost matrix: Euclidean distance between predicted centroid and detection centroid
    std::vector<std::vector<double>> cost(
        static_cast<size_t>(n_tracks),
        std::vector<double>(static_cast<size_t>(n_dets))
    );

    for (int i = 0; i < n_tracks; i++) {
        const Eigen::Vector2d pred = tracks_[static_cast<size_t>(i)].x.head<2>();
        for (int j = 0; j < n_dets; j++) {
            cost[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                (pred - detections[static_cast<size_t>(j)].centroid_m).norm();
        }
    }

    return hungarian_solve(cost, n_tracks, n_dets, params_.max_association_dist);
}

// ═══════════════ Track Management ═══════════════

std::vector<bool> ObjectTracker::update_tracks(const std::vector<Detection>& detections, const std::vector<int>& assignment) {
    std::vector<bool> det_used(detections.size(), false);
    std::vector<bool> detections_with_motion_prediction(detections.size(), false);

    // Update matched tracks / age unmatched tracks
    for (size_t i = 0; i < tracks_.size(); i++) {
        if (assignment[i] >= 0) {
            const auto j = static_cast<size_t>(assignment[i]);
            det_used[j] = true;

            // Save predicted centroid before KF update (for local grid alignment)
            const Eigen::Vector2d centroid_before = tracks_[i].x.head<2>();

            // KF measurement update
            kf_update(tracks_[i], detections[j].centroid_m);

            // Compute centroid shift in pixels
            const Eigen::Vector2d centroid_after = tracks_[i].x.head<2>();
            const double dx_px = (centroid_after[0] - centroid_before[0]) / resolution_;
            const double dy_px = (centroid_after[1] - centroid_before[1]) / resolution_;

            // 将旧形状和当前量测都对齐到后验航迹坐标系，避免快速运动时出现形状/状态失配。
            const cv::Mat shifted_old = shift_local_grid(tracks_[i].local_grid, dx_px, dy_px);
            const double meas_to_track_dx_px = (centroid_after[0] - detections[j].centroid_m[0]) / resolution_;
            const double meas_to_track_dy_px = (centroid_after[1] - detections[j].centroid_m[1]) / resolution_;
            const cv::Mat measurement_in_track_frame = shift_local_grid(
                detections[j].local_grid,
                meas_to_track_dx_px,
                meas_to_track_dy_px
            );

            // Blend: decayed shifted old + new observation (take element-wise max)
            cv::Mat decayed_old = shifted_old;
            decayed_old *= params_.local_grid_decay;
            cv::max(decayed_old, measurement_in_track_frame, tracks_[i].local_grid);

            tracks_[i].hit_streak++;
            tracks_[i].confirmed = tracks_[i].confirmed || tracks_[i].hit_streak >= params_.min_hits_to_confirm;
            tracks_[i].lost_frames = 0;
            if (tracks_[i].confirmed) {
                detections_with_motion_prediction[j] = true;
            }
        } else {
            tracks_[i].hit_streak = 0;
            tracks_[i].lost_frames++;
            tracks_[i].local_grid *= params_.local_grid_decay;
        }
        tracks_[i].age++;
    }

    // Remove dead tracks
    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [this](const Track& t) { return t.lost_frames > params_.max_lost_frames; }
        ),
        tracks_.end()
    );

    // Create new tracks for unmatched detections
    for (size_t j = 0; j < detections.size(); j++) {
        if (det_used[j]) continue;

        Track t {};
        t.id = next_id_++;
        t.x << detections[j].centroid_m[0], detections[j].centroid_m[1], 0.0, 0.0;
        t.P = Eigen::Matrix4d::Zero();
        const double rm = params_.measurement_noise_std * params_.measurement_noise_std;
        t.P(0, 0) = rm;
        t.P(1, 1) = rm;
        t.P(2, 2) = 4.0; // initial velocity uncertainty: 2 m/s std
        t.P(3, 3) = 4.0;
        t.local_grid = detections[j].local_grid.clone();
        t.hit_streak = 1;
        t.confirmed = t.hit_streak >= params_.min_hits_to_confirm;
        t.lost_frames = 0;
        t.age = 1;
        tracks_.push_back(std::move(t));
    }
    return detections_with_motion_prediction;
}

// ═══════════════ Prediction Rendering ═══════════════

cv::Mat ObjectTracker::shift_local_grid(const cv::Mat& local_grid, double dx_px, double dy_px) const {
    cv::Mat shifted;
    const cv::Mat shift_mat = (cv::Mat_<double>(2, 3) << 1.0, 0.0, -dx_px, 0.0, 1.0, -dy_px);
    cv::warpAffine(
        local_grid,
        shifted,
        shift_mat,
        local_grid.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    return shifted;
}

void ObjectTracker::rasterize_local_grid(
    cv::Mat& mask,
    const cv::Mat& local_grid,
    const Eigen::Vector2i& centroid_px,
    uint8_t value
) const {
    const int half = params_.local_grid_size / 2;
    const int grid_sz = params_.local_grid_size;
    const float render_thresh = static_cast<float>(params_.local_grid_render_threshold);

    for (int ly = 0; ly < grid_sz; ly++) {
        const int gy = centroid_px.y() + ly - half;
        if (gy < 0 || gy >= height_) continue;
        const float* lg_row = local_grid.ptr<float>(ly);
        uint8_t* out_row = mask.ptr<uint8_t>(gy);
        for (int lx = 0; lx < grid_sz; lx++) {
            if (lg_row[lx] <= render_thresh) continue;
            const int gx = centroid_px.x() + lx - half;
            if (gx >= 0 && gx < width_) {
                out_row[gx] = value;
            }
        }
    }
}

std::vector<ObjectTracker::MotionPrediction> ObjectTracker::build_motion_predictions() const {
    std::vector<MotionPrediction> predictions;
    predictions.reserve(tracks_.size());
    const double inv_res = 1.0 / resolution_;
    for (const auto& track: tracks_) {
        if (!track.confirmed) continue;

        MotionPrediction prediction;
        cv::compare(
            track.local_grid,
            params_.local_grid_render_threshold,
            prediction.footprint_mask,
            cv::CMP_GT
        );
        prediction.future_centroids_px.reserve(static_cast<size_t>(params_.prediction_steps));

        Eigen::Vector2d velocity = track.x.tail<2>();
        const double speed = velocity.norm();
        if (params_.prediction_max_speed > 0.0 && speed > params_.prediction_max_speed) {
            velocity *= params_.prediction_max_speed / speed;
        }

        for (int step = 0; step < params_.prediction_steps; ++step) {
            const double t_future = static_cast<double>(step + 1) * params_.prediction_dt;
            const double prediction_time = params_.prediction_velocity_decay_tau > 0.0
                ? params_.prediction_velocity_decay_tau * (
                    1.0 - std::exp(-t_future / params_.prediction_velocity_decay_tau)
                )
                : t_future;
            const Eigen::Vector2d pred_m = track.x.head<2>() + velocity * prediction_time;
            prediction.future_centroids_px.emplace_back(
                static_cast<int>(std::round(pred_m.x() * inv_res)),
                static_cast<int>(std::round(pred_m.y() * inv_res))
            );
        }
        predictions.push_back(std::move(prediction));
    }
    return predictions;
}

cv::Mat ObjectTracker::build_static_fallback_mask(
    const cv::Mat& obstacle_mask,
    const std::vector<Detection>& detections,
    const std::vector<bool>& detections_with_motion_prediction
) const {
    cv::Mat fallback_mask = obstacle_mask.clone();
    for (size_t i = 0; i < detections.size(); i++) {
        if (!detections_with_motion_prediction[i]) continue;
        rasterize_local_grid(fallback_mask, detections[i].local_grid, detections[i].centroid_px, 0);
    }
    return fallback_mask;
}

} // namespace map_server
