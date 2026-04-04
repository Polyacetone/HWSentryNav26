#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <random>

namespace map_server {

struct DOGMaParams {
    int num_particles_per_cell;    // 新占据栅格出生粒子数
    int persistent_birth_count;    // 持续占据栅格每帧补充粒子数
    double birth_velocity_range;   // 出生速度范围 |vx|,|vy| (m/s)
    double velocity_noise_std;     // 速度随机游走噪声 σ (m/s)
    double free_decay;             // 未观测时粒子存在置信度衰减因子
    double occupied_boost;         // 观测命中时粒子存在置信度提升因子
    double velocity_correction_gain; // 观测速度对粒子速度的校正增益，越小越稳但收敛越慢
    double unobserved_velocity_damping; // 未观测时速度阻尼，防止离开ROI后继续发散
    double min_particle_confidence; // 粒子最小存在置信度，低于该值直接删除
    int max_unseen_updates;        // 连续未观测超过该次数后删除粒子
    double resample_ess_ratio;     // ESS < ratio * N 时触发重采样
    int max_particles;             // 粒子数上限
    int prediction_steps;          // 未来预测步数
    double prediction_dt;          // 预测步长 (s)
    double occupancy_threshold;    // 预测栅格占据阈值（加权粒子计数）
    int num_threads;
};

class DOGMaPredictor {
public:
    DOGMaPredictor(int width, int height, double resolution, const DOGMaParams& params);

    /// 用当前帧动态障碍物掩码更新粒子滤波，返回未来 prediction_steps 步的预测占据栅格
    /// obstacle_mask: CV_8UC1, 0=free, 255=occupied
    /// dt: 距上次调用的时间间隔 (s)
    /// 返回: predictions[i] 对应 t_0 + (i+1) * prediction_dt
    std::vector<cv::Mat> update(const cv::Mat& obstacle_mask, double dt);

    [[nodiscard]] size_t particle_count() const {
        return particles_.size();
    }

private:
    struct Particle {
        double x, y;     // 位置 (m, map frame)
        double vx, vy;   // 速度 (m/s)
        double weight;
        double confidence;
        int unseen_updates;
    };

    void build_motion_observation(const cv::Mat& obstacle_mask, double dt);
    void predict(double dt);
    void update_weights(const cv::Mat& obstacle_mask);
    void prune_particles();
    void resample();
    void birth(const cv::Mat& obstacle_mask);
    cv::Mat predict_future(double t_future) const;
    static void clamp_velocity(double max_speed, double& vx, double& vy);

    int width_, height_;
    double resolution_;
    DOGMaParams params_;
    std::vector<Particle> particles_;
    cv::Mat prev_obstacle_mask_;
    cv::Mat observed_velocity_map_;
    std::mt19937 rng_;
};

} // namespace map_server