#include <map_server/dogma_predictor.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <algorithm>
#include <cmath>

namespace map_server {

DOGMaPredictor::DOGMaPredictor(int width, int height, double resolution, const DOGMaParams& params)
    : width_(width), height_(height), resolution_(resolution), params_(params), rng_(42) {
    particles_.reserve(static_cast<size_t>(params_.max_particles));
    prev_obstacle_mask_ = cv::Mat::zeros(height_, width_, CV_8UC1);
    prev_interior_dist_ = cv::Mat::zeros(height_, width_, CV_32FC1);
}

std::vector<cv::Mat> DOGMaPredictor::update(const cv::Mat& obstacle_mask, double dt) {
    // 预处理：形态学开运算去噪
    const cv::Mat clean_mask = preprocess_mask(obstacle_mask);

    // 计算障碍物内部距离场（到自由空间边界的距离，单位：像素）
    const cv::Mat interior_dist = compute_interior_distance(clean_mask);

    // 计算新生粒子的法向速度先验
    cv::Mat dir_x, dir_y, speed_mag;
    compute_birth_velocity_field(clean_mask, dt, dir_x, dir_y, speed_mag);

    // 粒子滤波主循环
    predict(dt);
    update_weights(clean_mask, interior_dist, dt);
    prune_particles();
    resample();
    birth(clean_mask, dir_x, dir_y, speed_mag, interior_dist);

    // 保存当前帧状态供下帧使用
    prev_obstacle_mask_ = clean_mask.clone();
    prev_interior_dist_ = interior_dist.clone();

    // 并行生成各时间步的预测占据栅格
    std::vector<cv::Mat> predictions(static_cast<size_t>(params_.prediction_steps));
    #pragma omp parallel for num_threads(params_.num_threads) schedule(static)
    for (int i = 0; i < params_.prediction_steps; i++) {
        const double t_future = static_cast<double>(i + 1) * params_.prediction_dt;
        predictions[static_cast<size_t>(i)] = predict_future(t_future);
    }
    return predictions;
}

void DOGMaPredictor::clamp_velocity(double max_speed, double& vx, double& vy) {
    const double speed = std::hypot(vx, vy);
    if (speed <= max_speed || speed <= 1e-9) return;
    const double scale = max_speed / speed;
    vx *= scale;
    vy *= scale;
}

// ═══════════════ 预处理 ═══════════════

cv::Mat DOGMaPredictor::preprocess_mask(const cv::Mat& mask) const {
    if (params_.morph_open_kernel_size <= 1) return mask;
    cv::Mat cleaned;
    const int ks = params_.morph_open_kernel_size | 1; // 确保奇数
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {ks, ks});
    cv::morphologyEx(mask, cleaned, cv::MORPH_OPEN, kernel);
    return cleaned;
}

cv::Mat DOGMaPredictor::compute_interior_distance(const cv::Mat& mask) const {
    // distanceTransform 计算前景像素到最近背景像素的距离
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, 3);
    return dist; // CV_32FC1, 单位：像素
}

void DOGMaPredictor::compute_birth_velocity_field(const cv::Mat& /*current_mask*/, double dt,
                                                   cv::Mat& direction_x, cv::Mat& direction_y,
                                                   cv::Mat& speed_magnitude) const {
    // 对上一帧掩码的**外部**做距离变换：每个像素到上一帧最近障碍物的距离
    // 新生区域（本帧有、上帧无）的距离值即为从旧边界到新位置的位移量
    cv::Mat prev_exterior_dist;
    if (cv::countNonZero(prev_obstacle_mask_) > 0) {
        // distanceTransform 要求输入为"前景=非零"，计算前景到背景的距离
        // 我们要的是"背景像素到最近前景的距离"，所以反转掩码
        cv::Mat inverted;
        cv::bitwise_not(prev_obstacle_mask_, inverted);
        cv::distanceTransform(inverted, prev_exterior_dist, cv::DIST_L2, 3);
    } else {
        prev_exterior_dist = cv::Mat::zeros(height_, width_, CV_32FC1);
    }

    // 用 Sobel 求梯度（梯度方向 = 远离旧障碍物的方向 = 膨胀方向）
    cv::Mat grad_x, grad_y;
    cv::Sobel(prev_exterior_dist, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(prev_exterior_dist, grad_y, CV_32F, 0, 1, 3);

    // 归一化梯度方向
    direction_x = cv::Mat::zeros(height_, width_, CV_32FC1);
    direction_y = cv::Mat::zeros(height_, width_, CV_32FC1);
    speed_magnitude = cv::Mat::zeros(height_, width_, CV_32FC1);

    const double inv_dt = (dt > 1e-6) ? (1.0 / dt) : 0.0;

    for (int y = 0; y < height_; y++) {
        const float* gx_row = grad_x.ptr<float>(y);
        const float* gy_row = grad_y.ptr<float>(y);
        const float* dist_row = prev_exterior_dist.ptr<float>(y);
        float* dx_row = direction_x.ptr<float>(y);
        float* dy_row = direction_y.ptr<float>(y);
        float* sm_row = speed_magnitude.ptr<float>(y);
        for (int x = 0; x < width_; x++) {
            const double gx = gx_row[x];
            const double gy = gy_row[x];
            const double mag = std::hypot(gx, gy);
            if (mag > 1e-6) {
                dx_row[x] = static_cast<float>(gx / mag);
                dy_row[x] = static_cast<float>(gy / mag);
                // 位移量(像素) * resolution / dt = 速度(m/s)
                sm_row[x] = static_cast<float>(dist_row[x] * resolution_ * inv_dt);
            }
        }
    }
}

// ═══════════════ 预测步：推进粒子位置与速度 ═══════════════

void DOGMaPredictor::predict(double dt) {
    if (particles_.empty() || dt <= 0.0) return;
    const double noise_std = params_.velocity_noise_std;
    const double max_speed = params_.birth_velocity_range;

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

// ═══════════════ 观测更新：距离变换 + 速度一致性 双重似然 ═══════════════

void DOGMaPredictor::update_weights(const cv::Mat& obstacle_mask, const cv::Mat& interior_dist, double dt) {
    if (particles_.empty()) return;
    const double inv_res = 1.0 / resolution_;
    const double free_decay = params_.free_decay;
    const double unobserved_velocity_damping = params_.unobserved_velocity_damping;
    const double pos_sigma = std::max(0.5, params_.position_sigma_cells); // 单位：栅格
    const double vel_sigma = std::max(0.1, params_.velocity_sigma);
    const double inv_2_vel_sigma_sq = -0.5 / (vel_sigma * vel_sigma);

    // 计算当前帧与上一帧距离场的时间差分，估计每个像素处的表观径向速度
    // apparent_speed = (prev_dist - cur_dist) * resolution / dt，正值表示障碍物在"长大"
    cv::Mat apparent_velocity_x, apparent_velocity_y;
    const bool has_prev = cv::countNonZero(prev_obstacle_mask_) > 0 && dt > 1e-6;
    if (has_prev) {
        // 用当前帧外部距离场的梯度方向作为速度方向参考
        cv::Mat cur_inverted;
        cv::bitwise_not(obstacle_mask, cur_inverted);
        cv::Mat cur_exterior_dist;
        cv::distanceTransform(cur_inverted, cur_exterior_dist, cv::DIST_L2, 3);

        cv::Mat grad_x, grad_y;
        cv::Sobel(cur_exterior_dist, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(cur_exterior_dist, grad_y, CV_32F, 0, 1, 3);

        // 表观速度 = 距离场变化 / dt，方向沿梯度
        cv::Mat speed_field = (prev_interior_dist_ - interior_dist) * static_cast<float>(resolution_ / dt);

        apparent_velocity_x = cv::Mat::zeros(height_, width_, CV_32FC1);
        apparent_velocity_y = cv::Mat::zeros(height_, width_, CV_32FC1);
        for (int y = 0; y < height_; y++) {
            const float* gx_row = grad_x.ptr<float>(y);
            const float* gy_row = grad_y.ptr<float>(y);
            const float* spd_row = speed_field.ptr<float>(y);
            float* avx_row = apparent_velocity_x.ptr<float>(y);
            float* avy_row = apparent_velocity_y.ptr<float>(y);
            for (int x = 0; x < width_; x++) {
                const double mag = std::hypot(gx_row[x], gy_row[x]);
                if (mag > 1e-6) {
                    const double s = spd_row[x];
                    avx_row[x] = static_cast<float>(s * gx_row[x] / mag);
                    avy_row[x] = static_cast<float>(s * gy_row[x] / mag);
                }
            }
        }
    }

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
            // ---- 位置似然：距离变换值越大（越靠近中心），权重越高 ----
            const double dist_val = interior_dist.at<float>(gy, gx); // 像素单位
            // 单调递增映射：靠近中心(dist大)的粒子得到更高权重
            const double pos_factor = 1.0 - std::exp(-dist_val / pos_sigma);

            // ---- 速度一致性似然（如果有历史帧）----
            double vel_factor = 1.0;
            if (has_prev) {
                const double avx = apparent_velocity_x.at<float>(gy, gx);
                const double avy = apparent_velocity_y.at<float>(gy, gx);
                const double dvx = p.vx - avx;
                const double dvy = p.vy - avy;
                const double vel_err_sq = dvx * dvx + dvy * dvy;
                vel_factor = std::exp(inv_2_vel_sigma_sq * vel_err_sq) + 0.1; // 底部偏移保持探索性
            }

            const double combined = pos_factor * vel_factor;
            p.weight *= combined;
            p.confidence = std::min(1.0, p.confidence + 0.05); // 温和提升
            p.unseen_updates = 0;
        } else {
            p.weight *= free_decay;
            p.confidence *= free_decay;
            p.unseen_updates++;
            p.vx *= unobserved_velocity_damping;
            p.vy *= unobserved_velocity_damping;
        }
    }

    // 归一化权重
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

    if (particles_.empty()) return;

    double sum_w = 0.0;
    for (const auto& particle : particles_) sum_w += particle.weight;
    if (sum_w <= 1e-15) {
        const double uniform_weight = 1.0 / static_cast<double>(particles_.size());
        for (auto& particle : particles_) particle.weight = uniform_weight;
        return;
    }
    const double inv_sum_w = 1.0 / sum_w;
    for (auto& particle : particles_) particle.weight *= inv_sum_w;
}

// ═══════════════ 系统性重采样 ═══════════════

void DOGMaPredictor::resample() {
    if (particles_.empty()) return;
    const auto N = particles_.size();
    const double n = static_cast<double>(N);

    double sum_sq = 0.0;
    for (const auto& p : particles_) sum_sq += p.weight * p.weight;
    const double ess = (sum_sq > 0.0) ? (1.0 / sum_sq) : 0.0;
    if (ess >= params_.resample_ess_ratio * n) return;

    std::vector<double> cumulative(N);
    cumulative[0] = particles_[0].weight;
    for (size_t i = 1; i < N; i++) {
        cumulative[i] = cumulative[i - 1] + particles_[i].weight;
    }

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

// ═══════════════ 粒子出生：法向速度先验 ═══════════════

void DOGMaPredictor::birth(const cv::Mat& obstacle_mask, const cv::Mat& dir_x, const cv::Mat& dir_y,
                            const cv::Mat& speed_mag, const cv::Mat& /*interior_dist*/) {
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
    const double dir_noise_std = params_.birth_direction_noise_std;
    std::normal_distribution<double> angle_noise(0.0, dir_noise_std);
    std::normal_distribution<double> speed_noise(0.0, params_.velocity_noise_std * 2.0);
    const double fallback_vel_std = std::max(0.05, params_.birth_velocity_range / 4.0);
    std::normal_distribution<double> fallback_vel(0.0, fallback_vel_std);
    const double birth_weight = particles_.empty() ? 1.0 : std::max(1e-3, 1.0 / static_cast<double>(particles_.size()));

    // 新生粒子：使用法向速度先验
    auto spawn_new = [&](int cx, int cy, int count) {
        const float dx = dir_x.at<float>(cy, cx);
        const float dy = dir_y.at<float>(cy, cx);
        const float spd = speed_mag.at<float>(cy, cx);
        const double base_angle = std::atan2(dy, dx);
        const bool has_direction = (std::abs(dx) > 1e-6 || std::abs(dy) > 1e-6);

        for (int s = 0; s < count && particles_.size() < max_total; s++) {
            Particle p;
            p.x = (static_cast<double>(cx) + 0.5 + pos_noise(rng_)) * resolution_;
            p.y = (static_cast<double>(cy) + 0.5 + pos_noise(rng_)) * resolution_;

            if (has_direction) {
                // 使用法向速度先验 + 方向噪声
                const double angle = base_angle + angle_noise(rng_);
                const double speed = std::max(0.0, static_cast<double>(spd) + speed_noise(rng_));
                p.vx = speed * std::cos(angle);
                p.vy = speed * std::sin(angle);
            } else {
                // 无方向信息时使用随机速度
                p.vx = fallback_vel(rng_);
                p.vy = fallback_vel(rng_);
            }
            clamp_velocity(params_.birth_velocity_range, p.vx, p.vy);
            p.weight = birth_weight;
            p.confidence = 1.0;
            p.unseen_updates = 0;
            particles_.push_back(p);
        }
    };

    // 持续占据栅格补充粒子：使用靠近中心优先+继承附近粒子速度方向
    auto spawn_persistent = [&](int cx, int cy, int count) {
        for (int s = 0; s < count && particles_.size() < max_total; s++) {
            Particle p;
            p.x = (static_cast<double>(cx) + 0.5 + pos_noise(rng_)) * resolution_;
            p.y = (static_cast<double>(cy) + 0.5 + pos_noise(rng_)) * resolution_;
            p.vx = fallback_vel(rng_);
            p.vy = fallback_vel(rng_);
            clamp_velocity(params_.birth_velocity_range, p.vx, p.vy);
            p.weight = birth_weight * 0.5; // 持续占据的补充权重更低
            p.confidence = 0.8;
            p.unseen_updates = 0;
            particles_.push_back(p);
        }
    };

    for (const auto& [cx, cy] : new_cells) spawn_new(cx, cy, params_.num_particles_per_cell);
    for (const auto& [cx, cy] : persistent_cells) spawn_persistent(cx, cy, params_.persistent_birth_count);

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
    if (particles_.empty()) return;
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

    cv::Mat accum = cv::Mat::zeros(height_, width_, CV_32FC1);

    for (const auto& p : particles_) {
        const double fx = p.x + p.vx * t_future;
        const double fy = p.y + p.vy * t_future;
        const int gx = static_cast<int>(fx * inv_res);
        const int gy = static_cast<int>(fy * inv_res);
        if (gx < 0 || gx >= width_ || gy < 0 || gy >= height_) continue;
        accum.at<float>(gy, gx) += static_cast<float>(p.weight * p.confidence * n);
    }

    cv::threshold(accum, accum, threshold, 255.0, cv::THRESH_BINARY);
    accum.convertTo(predicted, CV_8UC1);
    return predicted;
}

} // namespace map_server