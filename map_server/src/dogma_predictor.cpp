#include <map_server/dogma_predictor.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <algorithm>
#include <cmath>

namespace {
struct ComponentInfo {
    int label;
    cv::Point2d centroid_px;
    int area;
};

std::vector<ComponentInfo> extract_components(const cv::Mat& stats, const cv::Mat& centroids) {
    std::vector<ComponentInfo> components;
    for (int label = 1; label < stats.rows; label++) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area <= 0) {
            continue;
        }
        components.push_back(ComponentInfo {
            .label = label,
            .centroid_px = cv::Point2d(centroids.at<double>(label, 0), centroids.at<double>(label, 1)),
            .area = area,
        });
    }
    return components;
}
}

namespace map_server {

DOGMaPredictor::DOGMaPredictor(int width, int height, double resolution, const DOGMaParams& params)
    : width_(width), height_(height), resolution_(resolution), params_(params), rng_(42) {
    particles_.reserve(static_cast<size_t>(params_.max_particles));
    prev_obstacle_mask_ = cv::Mat::zeros(height_, width_, CV_8UC1);
    observed_velocity_map_ = cv::Mat::zeros(height_, width_, CV_32FC2);
}

std::vector<cv::Mat> DOGMaPredictor::update(const cv::Mat& obstacle_mask, double dt) {
    build_motion_observation(obstacle_mask, dt);
    predict(dt);
    update_weights(obstacle_mask);
    prune_particles();
    resample();
    birth(obstacle_mask);
    prev_obstacle_mask_ = obstacle_mask.clone();

    // 并行生成各时间步的预测占据栅格
    std::vector<cv::Mat> predictions(static_cast<size_t>(params_.prediction_steps));
    #pragma omp parallel for num_threads(params_.num_threads) schedule(static)
    for (int i = 0; i < params_.prediction_steps; i++) {
        const double t_future = static_cast<double>(i + 1) * params_.prediction_dt;
        predictions[static_cast<size_t>(i)] = predict_future(t_future);
    }
    return predictions;
}

void DOGMaPredictor::build_motion_observation(const cv::Mat& obstacle_mask, double dt) {
    observed_velocity_map_ = cv::Mat::zeros(height_, width_, CV_32FC2);
    if (dt <= 1e-6 || cv::countNonZero(obstacle_mask) == 0) {
        return;
    }

    cv::Mat current_labels, current_stats, current_centroids;
    cv::connectedComponentsWithStats(obstacle_mask, current_labels, current_stats, current_centroids, 8, CV_32S);
    const auto current_components = extract_components(current_stats, current_centroids);
    if (current_components.empty()) {
        return;
    }

    std::vector<cv::Point2d> label_velocity(static_cast<size_t>(current_stats.rows), cv::Point2d(0.0, 0.0));
    if (cv::countNonZero(prev_obstacle_mask_) > 0) {
        cv::Mat previous_labels, previous_stats, previous_centroids;
        cv::connectedComponentsWithStats(prev_obstacle_mask_, previous_labels, previous_stats, previous_centroids, 8, CV_32S);
        const auto previous_components = extract_components(previous_stats, previous_centroids);
        std::vector<bool> previous_used(previous_components.size(), false);
        const double max_assoc_distance_px = std::max(1.5, 1.5 * params_.birth_velocity_range * dt / resolution_);
        const double max_assoc_distance_sq = max_assoc_distance_px * max_assoc_distance_px;

        for (const auto& current : current_components) {
            int best_prev = -1;
            double best_distance_sq = max_assoc_distance_sq;
            for (size_t i = 0; i < previous_components.size(); i++) {
                if (previous_used[i]) {
                    continue;
                }
                const auto& previous = previous_components[i];
                const double area_ratio = static_cast<double>(current.area) / static_cast<double>(previous.area);
                if (area_ratio < 0.25 || area_ratio > 4.0) {
                    continue;
                }
                const double dx = current.centroid_px.x - previous.centroid_px.x;
                const double dy = current.centroid_px.y - previous.centroid_px.y;
                const double distance_sq = dx * dx + dy * dy;
                if (distance_sq < best_distance_sq) {
                    best_distance_sq = distance_sq;
                    best_prev = static_cast<int>(i);
                }
            }

            if (best_prev >= 0) {
                previous_used[static_cast<size_t>(best_prev)] = true;
                const auto& previous = previous_components[static_cast<size_t>(best_prev)];
                label_velocity[static_cast<size_t>(current.label)] = cv::Point2d(
                    (current.centroid_px.x - previous.centroid_px.x) * resolution_ / dt,
                    (current.centroid_px.y - previous.centroid_px.y) * resolution_ / dt
                );
            }
        }
    }

    for (int y = 0; y < height_; y++) {
        const int* labels_row = current_labels.ptr<int>(y);
        cv::Vec2f* velocity_row = observed_velocity_map_.ptr<cv::Vec2f>(y);
        for (int x = 0; x < width_; x++) {
            const int label = labels_row[x];
            if (label <= 0) {
                continue;
            }
            const auto& velocity = label_velocity[static_cast<size_t>(label)];
            velocity_row[x] = cv::Vec2f(static_cast<float>(velocity.x), static_cast<float>(velocity.y));
        }
    }
}

void DOGMaPredictor::clamp_velocity(double max_speed, double& vx, double& vy) {
    const double speed = std::hypot(vx, vy);
    if (speed <= max_speed || speed <= 1e-9) {
        return;
    }
    const double scale = max_speed / speed;
    vx *= scale;
    vy *= scale;
}

// ═══════════════ 预测步：推进粒子位置与速度 ═══════════════

void DOGMaPredictor::predict(double dt) {
    if (particles_.empty() || dt <= 0.0) return;
    const double noise_std = params_.velocity_noise_std;
    const double max_speed = params_.birth_velocity_range;

    // 为每个线程准备独立的 RNG
    const int nt = params_.num_threads;
    std::vector<std::mt19937> thread_rngs(static_cast<size_t>(nt));
    for (int t = 0; t < nt; t++) {
        thread_rngs[static_cast<size_t>(t)] = std::mt19937(rng_() + static_cast<unsigned>(t));
    }

    #pragma omp parallel for num_threads(nt) schedule(static)
    for (size_t i = 0; i < particles_.size(); i++) {
        auto& p = particles_[i];
        auto& local_rng = thread_rngs[static_cast<size_t>(omp_get_thread_num())];
        const double effective_noise_std = noise_std * std::clamp(p.confidence, 0.15, 1.0);
        std::normal_distribution<double> noise(0.0, effective_noise_std);
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vx += noise(local_rng);
        p.vy += noise(local_rng);
        clamp_velocity(max_speed, p.vx, p.vy);
    }
}

// ═══════════════ 观测更新：调整粒子权重 ═══════════════

void DOGMaPredictor::update_weights(const cv::Mat& obstacle_mask) {
    if (particles_.empty()) return;
    const double inv_res = 1.0 / resolution_;
    const double occ_boost = params_.occupied_boost;
    const double free_decay = params_.free_decay;
    const double velocity_sigma = std::max(0.2, 2.0 * params_.velocity_noise_std);
    const double velocity_sigma_sq = velocity_sigma * velocity_sigma;
    const double velocity_correction_gain = params_.velocity_correction_gain;
    const double max_speed = params_.birth_velocity_range;
    const double unobserved_velocity_damping = params_.unobserved_velocity_damping;

    #pragma omp parallel for num_threads(params_.num_threads) schedule(static)
    for (size_t i = 0; i < particles_.size(); i++) {
        auto& p = particles_[i];
        const int gx = static_cast<int>(p.x * inv_res);
        const int gy = static_cast<int>(p.y * inv_res);

        if (gx < 0 || gx >= width_ || gy < 0 || gy >= height_) {
            p.weight *= 0.01;
            p.confidence *= free_decay;
            p.unseen_updates++;
            continue;
        }

        if (obstacle_mask.at<uint8_t>(gy, gx) > 0) {
            const cv::Vec2f observed_velocity = observed_velocity_map_.at<cv::Vec2f>(gy, gx);
            const double dvx = p.vx - static_cast<double>(observed_velocity[0]);
            const double dvy = p.vy - static_cast<double>(observed_velocity[1]);
            const double velocity_likelihood = std::exp(-(dvx * dvx + dvy * dvy) / (2.0 * velocity_sigma_sq));
            p.weight *= occ_boost * std::max(0.05, velocity_likelihood);
            p.confidence = std::min(1.0, p.confidence * occ_boost);
            p.unseen_updates = 0;
            p.vx = (1.0 - velocity_correction_gain) * p.vx + velocity_correction_gain * static_cast<double>(observed_velocity[0]);
            p.vy = (1.0 - velocity_correction_gain) * p.vy + velocity_correction_gain * static_cast<double>(observed_velocity[1]);
            clamp_velocity(max_speed, p.vx, p.vy);
        } else {
            p.weight *= free_decay;
            p.confidence *= free_decay;
            p.unseen_updates++;
            p.vx *= unobserved_velocity_damping;
            p.vy *= unobserved_velocity_damping;
        }
    }

    // 使用有效权重 = 后验权重 * 存在置信度 做归一化
    double sum_w = 0.0;
    #pragma omp parallel for num_threads(params_.num_threads) reduction(+:sum_w)
    for (size_t i = 0; i < particles_.size(); i++) {
        particles_[i].weight *= std::max(particles_[i].confidence, 1e-6);
        sum_w += particles_[i].weight;
    }
    if (sum_w > 1e-15) {
        const double inv_sum = 1.0 / sum_w;
        #pragma omp parallel for num_threads(params_.num_threads)
        for (size_t i = 0; i < particles_.size(); i++) {
            particles_[i].weight *= inv_sum;
        }
    }
}

void DOGMaPredictor::prune_particles() {
    particles_.erase(
        std::remove_if(
            particles_.begin(),
            particles_.end(),
            [this](const Particle& particle) {
                const bool out_of_map =
                    particle.x < 0.0 || particle.x >= static_cast<double>(width_) * resolution_ ||
                    particle.y < 0.0 || particle.y >= static_cast<double>(height_) * resolution_;
                return out_of_map ||
                    particle.confidence < params_.min_particle_confidence ||
                    particle.unseen_updates > params_.max_unseen_updates ||
                    !std::isfinite(particle.x) || !std::isfinite(particle.y) ||
                    !std::isfinite(particle.vx) || !std::isfinite(particle.vy) || !std::isfinite(particle.weight);
            }
        ),
        particles_.end()
    );

    if (particles_.empty()) {
        return;
    }

    double sum_w = 0.0;
    for (const auto& particle : particles_) {
        sum_w += particle.weight;
    }
    if (sum_w <= 1e-15) {
        const double uniform_weight = 1.0 / static_cast<double>(particles_.size());
        for (auto& particle : particles_) {
            particle.weight = uniform_weight;
        }
        return;
    }
    const double inv_sum_w = 1.0 / sum_w;
    for (auto& particle : particles_) {
        particle.weight *= inv_sum_w;
    }
}

// ═══════════════ 系统性重采样 ═══════════════

void DOGMaPredictor::resample() {
    if (particles_.empty()) return;
    const auto N = particles_.size();
    const double n = static_cast<double>(N);

    // 计算有效样本数 ESS = 1 / Σ(wi²)
    double sum_sq = 0.0;
    for (const auto& p : particles_) sum_sq += p.weight * p.weight;
    const double ess = (sum_sq > 0.0) ? (1.0 / sum_sq) : 0.0;
    if (ess >= params_.resample_ess_ratio * n) return;

    // 累积分布
    std::vector<double> cumulative(N);
    cumulative[0] = particles_[0].weight;
    for (size_t i = 1; i < N; i++) {
        cumulative[i] = cumulative[i - 1] + particles_[i].weight;
    }

    // 系统性重采样
    std::uniform_real_distribution<double> dist(0.0, 1.0 / n);
    const double r = dist(rng_);
    const double uniform_w = 1.0 / n;

    std::vector<Particle> new_particles;
    new_particles.reserve(N);
    size_t j = 0;
    for (size_t i = 0; i < N; i++) {
        const double u = r + static_cast<double>(i) / n;
        while (j < N - 1 && u > cumulative[j]) j++;
        Particle p = particles_[j];
        p.weight = uniform_w;
        new_particles.push_back(p);
    }
    particles_ = std::move(new_particles);
}

// ═══════════════ 粒子出生 ═══════════════

void DOGMaPredictor::birth(const cv::Mat& obstacle_mask) {
    std::vector<std::pair<int, int>> new_cells, persistent_cells;
    for (int y = 0; y < height_; y++) {
        const uint8_t* row_cur = obstacle_mask.ptr<uint8_t>(y);
        const uint8_t* row_prev = prev_obstacle_mask_.ptr<uint8_t>(y);
        for (int x = 0; x < width_; x++) {
            if (row_cur[x] > 0) {
                if (row_prev[x] == 0) {
                    new_cells.emplace_back(x, y);
                } else {
                    persistent_cells.emplace_back(x, y);
                }
            }
        }
    }

    const auto max_total = static_cast<size_t>(params_.max_particles);
    std::uniform_real_distribution<double> pos_noise(-0.5, 0.5);
    std::normal_distribution<double> vel_noise(0.0, params_.velocity_noise_std);
    const double birth_weight = particles_.empty() ? 1.0 : std::max(1e-3, 1.0 / static_cast<double>(particles_.size()));

    auto spawn = [&](int cx, int cy, int count) {
        for (int s = 0; s < count && particles_.size() < max_total; s++) {
            const cv::Vec2f observed_velocity = observed_velocity_map_.at<cv::Vec2f>(cy, cx);
            Particle p;
            p.x = (static_cast<double>(cx) + 0.5 + pos_noise(rng_)) * resolution_;
            p.y = (static_cast<double>(cy) + 0.5 + pos_noise(rng_)) * resolution_;
            p.vx = static_cast<double>(observed_velocity[0]) + vel_noise(rng_);
            p.vy = static_cast<double>(observed_velocity[1]) + vel_noise(rng_);
            clamp_velocity(params_.birth_velocity_range, p.vx, p.vy);
            p.weight = birth_weight;
            p.confidence = 1.0;
            p.unseen_updates = 0;
            particles_.push_back(p);
        }
    };

    for (const auto& [cx, cy] : new_cells) spawn(cx, cy, params_.num_particles_per_cell);
    for (const auto& [cx, cy] : persistent_cells) spawn(cx, cy, params_.persistent_birth_count);

    // 超限裁剪（按权重从大到小保留）
    if (particles_.size() > max_total) {
        std::partial_sort(
            particles_.begin(),
            particles_.begin() + static_cast<ptrdiff_t>(max_total),
            particles_.end(),
            [](const Particle& a, const Particle& b) { return a.weight > b.weight; }
        );
        particles_.resize(max_total);
    }

    // 重新归一化
    if (particles_.empty()) {
        return;
    }
    double sum_w = 0.0;
    for (const auto& p : particles_) sum_w += p.weight;
    if (sum_w > 1e-15) {
        for (auto& p : particles_) p.weight /= sum_w;
    } else {
        const double uniform = 1.0 / static_cast<double>(particles_.size());
        for (auto& p : particles_) p.weight = uniform;
    }
}

// ═══════════════ 生成未来预测占据栅格 ═══════════════

cv::Mat DOGMaPredictor::predict_future(double t_future) const {
    cv::Mat predicted = cv::Mat::zeros(height_, width_, CV_8UC1);
    const double inv_res = 1.0 / resolution_;
    const double n = static_cast<double>(particles_.size());
    const double threshold = params_.occupancy_threshold;

    // 使用浮点累加图避免 uint8 溢出
    cv::Mat accum = cv::Mat::zeros(height_, width_, CV_32FC1);

    for (const auto& p : particles_) {
        const double fx = p.x + p.vx * t_future;
        const double fy = p.y + p.vy * t_future;
        const int gx = static_cast<int>(fx * inv_res);
        const int gy = static_cast<int>(fy * inv_res);
        if (gx < 0 || gx >= width_ || gy < 0 || gy >= height_) continue;
        accum.at<float>(gy, gx) += static_cast<float>(p.weight * p.confidence * n);
    }

    // 阈值化
    cv::threshold(accum, accum, threshold, 255.0, cv::THRESH_BINARY);
    accum.convertTo(predicted, CV_8UC1);
    return predicted;
}

} // namespace map_server