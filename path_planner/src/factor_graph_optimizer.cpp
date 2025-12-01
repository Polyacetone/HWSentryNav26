#include <path_planner/factor_graph_optimizer.hpp>
#include <rclcpp/logging.hpp>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

namespace path_planner {

// 障碍物代价因子：约束路径点远离高代价（障碍物）区域
class ObstacleFactor : public gtsam::NoiseModelFactor1<gtsam::Point2> {
public:
    ObstacleFactor(gtsam::Key key, const gtsam::SharedNoiseModel& model, const CostMap& cost_map)
        : gtsam::NoiseModelFactor1<gtsam::Point2>(model, key), cost_map_(cost_map) {}

    gtsam::Vector evaluateError(
        const gtsam::Point2& p,
        gtsam::OptionalMatrixType H = nullptr
    ) const override {
        // 边界检查：若超出地图范围，给一个很大的代价，同时雅可比设为 0
        if (!cost_map_.is_valid_coord(p)) {
            if (H) *H = gtsam::Matrix::Zero(1, 2);
            return gtsam::Vector1(255); // 越界位置给予高代价
        }

        // 读取代价以及梯度
        double cost = cost_map_.interpolate(p);
        
        if (H) {
            Eigen::Vector2d grad = cost_map_.gradient(p);
            *H = (gtsam::Matrix(1, 2) << grad.x(), grad.y()).finished();
        }

        return gtsam::Vector1(cost);
    }

private:
    const CostMap& cost_map_;
};

// 方向场因子：约束路径局部方向尽量与给定方向场一致
class DirectionFactor : public gtsam::NoiseModelFactor2<gtsam::Point2, gtsam::Point2> {
public:
    DirectionFactor(gtsam::Key key1, gtsam::Key key2, const gtsam::SharedNoiseModel& model, const DirectionMap& dir_map)
        : gtsam::NoiseModelFactor2<gtsam::Point2, gtsam::Point2>(model, key1, key2), dir_map_(dir_map) {}

    gtsam::Vector evaluateError(
        const gtsam::Point2& p1, const gtsam::Point2& p2,
        gtsam::OptionalMatrixType H1 = nullptr,
        gtsam::OptionalMatrixType H2 = nullptr
    ) const override {
        // 使用中点在方向场里查询
        gtsam::Point2 mid = (p1 + p2) * 0.5;

        // 若中点超出方向场范围，则不施加约束（误差为 0，雅可比为 0）
        if (!dir_map_.is_valid_coord(mid)) {
            if (H1) *H1 = gtsam::Matrix::Zero(1, 2);
            if (H2) *H2 = gtsam::Matrix::Zero(1, 2);
            return gtsam::Vector1(0.0);
        }

        Eigen::Vector2d dir_vec = dir_map_.interpolate(mid);
        
        // 若方向向量很小，认为方向不可靠，不施加约束
        if (dir_vec.norm() < 1e-3) {
            if (H1) *H1 = gtsam::Matrix::Zero(1, 2);
            if (H2) *H2 = gtsam::Matrix::Zero(1, 2);
            return gtsam::Vector1(0.0);
        }
        
        dir_vec.normalize();

        // 路径方向向量
        gtsam::Point2 v = p2 - p1;
        
        // 期望 v 与 dir_vec 平行
        // 误差定义为 2D 叉积：error = v.x * d.y - v.y * d.x
        // 理想情况 error = 0
        double error = v.x() * dir_vec.y() - v.y() * dir_vec.x();

        if (H1) {
            // 对 p1 的导数
            // v = p2 - p1 => dv/dp1 = -I
            // d(error)/dv_x = dir_vec.y
            // d(error)/dv_y = -dir_vec.x
            // d(error)/dp1_x = -dir_vec.y
            // d(error)/dp1_y =  dir_vec.x
            *H1 = (gtsam::Matrix(1, 2) << -dir_vec.y(), dir_vec.x()).finished();
        }

        if (H2) {
            // 对 p2 的导数
            // v = p2 - p1 => dv/dp2 = I
            // d(error)/dp2_x = dir_vec.y
            // d(error)/dp2_y = -dir_vec.x
            *H2 = (gtsam::Matrix(1, 2) << dir_vec.y(), -dir_vec.x()).finished();
        }

        return gtsam::Vector1(error);
    }

private:
    const DirectionMap& dir_map_;
};

// 平滑因子：二阶差分（离散加速度）最小化，使路径更平滑
class SmoothnessFactor : public gtsam::NoiseModelFactor3<gtsam::Point2, gtsam::Point2, gtsam::Point2> {
public:
    SmoothnessFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor3<gtsam::Point2, gtsam::Point2, gtsam::Point2>(model, key1, key2, key3) {}

    gtsam::Vector evaluateError(
        const gtsam::Point2& p1,
        const gtsam::Point2& p2,
        const gtsam::Point2& p3,
        gtsam::OptionalMatrixType H1 = nullptr,
        gtsam::OptionalMatrixType H2 = nullptr,
        gtsam::OptionalMatrixType H3 = nullptr
    ) const override {
        gtsam::Point2 err = p1 - 2.0 * p2 + p3;

        if (H1) *H1 = gtsam::Matrix::Identity(2, 2);
        if (H2) *H2 = -2.0 * gtsam::Matrix::Identity(2, 2);
        if (H3) *H3 = gtsam::Matrix::Identity(2, 2);

        return err;
    }
};

// 等距因子：约束相邻两段路径的长度尽量相等，解决点分布不均的问题
class EquidistanceFactor : public gtsam::NoiseModelFactor3<gtsam::Point2, gtsam::Point2, gtsam::Point2> {
public:
    EquidistanceFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor3<gtsam::Point2, gtsam::Point2, gtsam::Point2>(model, key1, key2, key3) {}

    gtsam::Vector evaluateError(
        const gtsam::Point2& p1,
        const gtsam::Point2& p2,
        const gtsam::Point2& p3,
        gtsam::OptionalMatrixType H1 = nullptr,
        gtsam::OptionalMatrixType H2 = nullptr,
        gtsam::OptionalMatrixType H3 = nullptr
    ) const override {
        double d1 = (p2 - p1).norm();
        double d2 = (p3 - p2).norm();
        
        // 防止除零
        if (d1 < 1e-5) d1 = 1e-5;
        if (d2 < 1e-5) d2 = 1e-5;

        double error = d1 - d2;

        if (H1) {
            // d(error)/dp1 = d(d1)/dp1 = - (p2-p1)^T / d1
            *H1 = -(p2 - p1).transpose() / d1;
        }
        if (H2) {
            // d(error)/dp2 = d(d1)/dp2 - d(d2)/dp2
            // d(d1)/dp2 = (p2-p1)^T / d1
            // d(d2)/dp2 = - (p3-p2)^T / d2
            *H2 = (p2 - p1).transpose() / d1 + (p3 - p2).transpose() / d2;
        }
        if (H3) {
            // d(error)/dp3 = - d(d2)/dp3 = - (p3-p2)^T / d2
            *H3 = -(p3 - p2).transpose() / d2;
        }

        return gtsam::Vector1(error);
    }
};

FactorGraphOptimizer::FactorGraphOptimizer(
    const double smoothness_weight,
    const double length_weight,
    const double obstacle_weight,
    const double direction_weight,
    const int max_iterations
) : smoothness_weight_(smoothness_weight),
    length_weight_(length_weight),
    obstacle_weight_(obstacle_weight),
    direction_weight_(direction_weight),
    max_iterations_(max_iterations) {}

std::vector<Eigen::Vector2d> FactorGraphOptimizer::optimize(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const std::vector<Eigen::Vector2d>& init_path
) const {
    if (init_path.size() <= 2) {
        RCLCPP_WARN(rclcpp::get_logger("bspline_optimizer"), "Path too short to optimize!");
        return init_path;
    }

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_estimate;

    // 平滑项：最小化二阶差分（加速度）。权重越大，路径越平滑。
    auto smoothness_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0 / smoothness_weight_);
    
    // 长度项：最小化相邻点距离。权重越大，路径越短。
    auto length_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0 / length_weight_);
    
    // 障碍物项：最小化代价值。权重越大，越强烈避开障碍。
    auto obstacle_noise = gtsam::noiseModel::Isotropic::Sigma(1, 1.0 / obstacle_weight_);
    
    // 方向项：路径方向跟随方向场。权重越大，越强制贴合方向场。
    auto direction_noise = gtsam::noiseModel::Isotropic::Sigma(1, 1.0 / direction_weight_);

    // 等距项：使用与长度权重相关的噪声模型，确保点分布均匀
    // 这里复用 length_weight，因为均匀性也是关于长度的约束
    auto spacing_noise = gtsam::noiseModel::Isotropic::Sigma(1, 1.0 / length_weight_);

    // 先验项：固定起点和终点
    auto prior_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1e-6); // 很强的先验

    for (size_t i = 0; i < init_path.size(); i++) {
        gtsam::Key key = gtsam::Symbol('x', i);
        initial_estimate.insert(key, gtsam::Point2(init_path[i]));

        // 障碍物因子
        graph.add(gtsam::NonlinearFactor::shared_ptr(
            new ObstacleFactor(key, obstacle_noise, cost_map)
        ));

        // 固定起点和终点
        if (i == 0 || i == init_path.size() - 1) {
            graph.add(gtsam::PriorFactor<gtsam::Point2>(key, gtsam::Point2(init_path[i]), prior_noise));
        }

        if (i > 0) {
            gtsam::Key prev_key = gtsam::Symbol('x', i - 1);
            
            // 长度因子：BetweenFactor，测量为零向量，鼓励相邻点之间距离缩小
            graph.add(gtsam::BetweenFactor<gtsam::Point2>(prev_key, key, gtsam::Point2(0, 0), length_noise));

            // 方向因子：路径段对齐方向场
            graph.add(gtsam::NonlinearFactor::shared_ptr(
                new DirectionFactor(prev_key, key, direction_noise, direction_map)
            ));
        }

        if (i > 0 && i < init_path.size() - 1) {
            gtsam::Key prev_key = gtsam::Symbol('x', i - 1);
            gtsam::Key next_key = gtsam::Symbol('x', i + 1);
            
            // 平滑因子：二阶差分
            graph.add(gtsam::NonlinearFactor::shared_ptr(
                new SmoothnessFactor(prev_key, key, next_key, smoothness_noise)
            ));

            // 等距因子：约束前后两段距离相等
            graph.add(gtsam::NonlinearFactor::shared_ptr(
                new EquidistanceFactor(prev_key, key, next_key, spacing_noise)
            ));
        }
    }

    // 参数设置
    gtsam::DoglegParams params;
    params.setVerbosity("SILENT");
    params.setMaxIterations(max_iterations_);
    
    gtsam::DoglegOptimizer optimizer(graph, initial_estimate, params);
    gtsam::Values result = optimizer.optimize();

    std::vector<Eigen::Vector2d> optimized_path;
    optimized_path.reserve(init_path.size());

    for (size_t i = 0; i < init_path.size(); i++) {
        gtsam::Key key = gtsam::Symbol('x', i);
        optimized_path.push_back(result.at<gtsam::Point2>(key));
    }

    return optimized_path;
}
} // namespace path_planner