#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <random>

namespace map_server {

struct DOGMaParams {
    int num_particles_per_cell;    // 新占据栅格出生粒子数
    int persistent_birth_count;    // 持续占据栅格每帧补充粒子数
    double birth_velocity_range;   // 预测允许的最大速度 (m/s)
    double velocity_noise_std;     // 速度随机游走噪声 σ (m/s)
    double free_decay;             // 未观测时粒子权重衰减因子
    double unobserved_velocity_damping; // 未观测时速度阻尼
    double min_particle_confidence; // 粒子最小存在置信度
    int max_unseen_updates;        // 连续未观测超过该次数后删除粒子
    double resample_ess_ratio;     // ESS < ratio * N 时触发重采样
    int max_particles;             // 粒子数上限
    int prediction_steps;          // 未来预测步数
    double prediction_dt;          // 预测步长 (s)
    double occupancy_threshold;    // 预测栅格占据阈值
    double position_sigma_cells;   // 距离变换位置似然的 σ（单位：栅格数）
    double velocity_sigma;         // 速度一致性似然的 σ (m/s)
    double birth_direction_noise_std; // 新生粒子方向噪声 σ (rad)
    int morph_open_kernel_size;    // 形态学开运算核大小（0=关闭）
    int num_threads;
};

class DOGMaPredictor {
public:
    DOGMaPredictor(int width, int height, double resolution, const DOGMaParams& params);

    /// 用当前帧动态障碍物掩码更新粒子滤波，返回未来 prediction_steps 步的预测占据栅格
    /// obstacle_mask: CV_8UC1, 0=free, 255=occupied
    /// dt: 距上次调用的时间间隔 (s)
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

    /// 预处理掩码（形态学开运算去噪）
    cv::Mat preprocess_mask(const cv::Mat& mask) const;
    /// 计算障碍物内部距离场（到自由空间边界的距离）
    cv::Mat compute_interior_distance(const cv::Mat& mask) const;
    /// 计算当前帧新生区域的法向速度先验（方向 + 速度大小）
    void compute_birth_velocity_field(const cv::Mat& current_mask, double dt,
                                      cv::Mat& direction_x, cv::Mat& direction_y,
                                      cv::Mat& speed_magnitude) const;

    void predict(double dt);
    void update_weights(const cv::Mat& obstacle_mask, const cv::Mat& interior_dist, double dt);
    void prune_particles();
    void resample();
    void birth(const cv::Mat& obstacle_mask, const cv::Mat& dir_x, const cv::Mat& dir_y,
               const cv::Mat& speed_mag, const cv::Mat& interior_dist);
    cv::Mat predict_future(double t_future) const;
    static void clamp_velocity(double max_speed, double& vx, double& vy);

    int width_, height_;
    double resolution_;
    DOGMaParams params_;
    std::vector<Particle> particles_;
    cv::Mat prev_obstacle_mask_;
    cv::Mat prev_interior_dist_;  // 上一帧的内部距离场，用于速度估计
    std::mt19937 rng_;
};

} // namespace map_server