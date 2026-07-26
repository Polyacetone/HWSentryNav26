#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>

namespace map_server {

struct ObjectTrackerParams {
    int morph_close_kernel_size; // 形态学闭运算核大小（0或1=关闭）
    int min_blob_area; // 最小连通域面积（像素），滤除噪声小斑块
    int local_grid_size; // 局部栅格地图边长（像素）
    double max_association_dist; // 匈牙利匹配最大关联距离（m）
    double process_noise_std; // KF 过程噪声（加速度标准差，m/s²）
    double measurement_noise_std; // KF 量测噪声标准差（m）
    int max_lost_frames; // 连续未匹配帧数上限，超过则删除航迹
    int min_hits_to_confirm; // 航迹连续匹配此次数后才参与预测输出
    double local_grid_decay; // 未匹配时局部栅格衰减因子
    double local_grid_render_threshold; // 局部栅格渲染阈值（0~1）
    int prediction_steps; // 未来预测步数
    double prediction_dt; // 预测时间步长（s）
    double prediction_velocity_decay_tau; // 预测渲染速度指数衰减时间常数（s，<=0 表示不衰减）
    double prediction_max_speed; // 预测渲染速度上限（m/s，<=0 表示不限制）
};

class ObjectTracker {
public:
    struct MotionPrediction {
        cv::Mat footprint_mask; // CV_8UC1，局部坐标系中的二值目标形状
        std::vector<Eigen::Vector2i> future_centroids_px;
    };

    struct PredictionResult {
        cv::Mat static_fallback_mask; // 当前帧未被运动预测覆盖的静态保底障碍物
        std::vector<MotionPrediction> motion_predictions;
    };

    ObjectTracker(int width, int height, double resolution, const ObjectTrackerParams& params);

    /// 用当前帧动态障碍物掩码更新跟踪器。
    /// 输出会将“可运动预测航迹”的未来占据与“当前帧静态保底残差”合并，保证
    /// 没有进入运动预测链路的障碍物也不会从未来代价地图中消失。
    /// obstacle_mask: CV_8UC1, 0=free, 255=occupied
    /// dt: 距上次调用的时间间隔 (s)
    PredictionResult update(const cv::Mat& obstacle_mask, double dt);

    [[nodiscard]] size_t track_count() const {
        return tracks_.size();
    }

private:
    struct Detection {
        Eigen::Vector2d centroid_m; // 质心位置（m）
        Eigen::Vector2i centroid_px; // 质心像素坐标（四舍五入）
        cv::Mat local_grid; // CV_32FC1 局部栅格 [0,1]
    };

    struct Track {
        int id;
        Eigen::Vector4d x; // 状态 [px, py, vx, vy]
        Eigen::Matrix4d P; // 协方差
        cv::Mat local_grid; // CV_32FC1 局部栅格 [0,1]
        int hit_streak; // 连续匹配次数
        bool confirmed; // 一旦确认过，就允许在短时丢失时继续运动预测
        int lost_frames; // 连续未匹配帧数
        int age; // 总存活帧数
    };

    cv::Mat preprocess_mask(const cv::Mat& mask) const;
    std::vector<Detection> detect(const cv::Mat& mask) const;
    void kf_predict(Track& track, double dt) const;
    void kf_update(Track& track, const Eigen::Vector2d& z) const;
    std::vector<int> associate(const std::vector<Detection>& detections) const;
    std::vector<bool> update_tracks(const std::vector<Detection>& detections, const std::vector<int>& assignment);
    cv::Mat shift_local_grid(const cv::Mat& local_grid, double dx_px, double dy_px) const;
    void rasterize_local_grid(cv::Mat& mask, const cv::Mat& local_grid, const Eigen::Vector2i& centroid_px, uint8_t value) const;
    std::vector<MotionPrediction> build_motion_predictions() const;
    cv::Mat build_static_fallback_mask(
        const cv::Mat& obstacle_mask,
        const std::vector<Detection>& detections,
        const std::vector<bool>& detections_with_motion_prediction
    ) const;

    int width_, height_;
    double resolution_;
    ObjectTrackerParams params_;
    std::vector<Track> tracks_;
    int next_id_ = 0;
};

} // namespace map_server
