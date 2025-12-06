#include <small_glim/odometry/imu_integration.hpp>
#include <small_glim/common/config.hpp>
#include <small_glim/common/logger.hpp>

namespace small_glim {

IMUIntegrationParams::IMUIntegrationParams(const Config::Ptr config) {
    upright = config->param<bool>("sensors.imu_upright");
    acc_noise = config->param<double>("sensors.imu_acc_noise");
    gyro_noise = config->param<double>("sensors.imu_gyro_noise");
    int_noise = config->param<double>("sensors.imu_int_noise");
    acc_saturation_thresh = config->param<double>("sensors.imu_acc_saturation_thresh");
    gyro_saturation_thresh = config->param<double>("sensors.imu_gyro_saturation_thresh");
    saturation_mult = config->param<double>("sensors.imu_saturation_mult");
}

IMUIntegration::IMUIntegration(const Config::Ptr config) {
    params = std::make_unique<IMUIntegrationParams>(config);
    auto imu_params = gtsam::PreintegrationParams::MakeSharedU();
    if (!params->upright) {
        imu_params = gtsam::PreintegrationParams::MakeSharedD();
    }

    imu_params->accelerometerCovariance = gtsam::Matrix3::Identity() * std::pow(params->acc_noise, 2);
    imu_params->gyroscopeCovariance = gtsam::Matrix3::Identity() * std::pow(params->gyro_noise, 2);
    imu_params->integrationCovariance = gtsam::Matrix3::Identity() * pow(params->int_noise, 2);
    imu_measurements = std::make_shared<gtsam::PreintegratedImuMeasurements>(imu_params);
}

void IMUIntegration::insert_imu(
    double stamp,
    const Eigen::Vector3d& linear_acc,
    const Eigen::Vector3d& angular_vel
) {
    Eigen::Matrix<double, 7, 1> imu;
    imu << stamp, linear_acc, angular_vel;
    imu_queue.push_back(imu);
}

int IMUIntegration::integrate_imu(
    double start_time,
    double end_time,
    const gtsam::imuBias::ConstantBias& bias,
    int* num_integrated
) {
    *num_integrated = 0;
    imu_measurements->resetIntegrationAndSetBias(bias);

    int cursor = 0;
    auto imu_itr = imu_queue.begin();
    double last_stamp = start_time;

    if (imu_itr == imu_queue.end()) {
        return cursor;
    }

    for (; imu_itr != imu_queue.end(); imu_itr++, cursor++) {
        const auto& imu_frame = *imu_itr;
        const double imu_stamp = imu_frame[0];

        if (imu_stamp > end_time) {
            break;
        }

        const double dt = imu_stamp - last_stamp;
        if (dt <= 0.0) {
            continue;
        }

        const auto& a = imu_frame.block<3, 1>(1, 0);
        const auto& w = imu_frame.block<3, 1>(4, 0);

        bool saturated = a.cwiseAbs().maxCoeff() > params->acc_saturation_thresh ||
            w.cwiseAbs().maxCoeff() > params->gyro_saturation_thresh;

        if (saturated) {
            logger::info("imu_integration", "IMU saturation detected: |acc| = {:.2f}, |gyro| = {:.2f}", a.norm(), w.norm());
            auto cov_acc = imu_measurements->p().accelerometerCovariance;
            auto cov_gyro = imu_measurements->p().gyroscopeCovariance;
            imu_measurements->p().accelerometerCovariance *= params->saturation_mult;
            imu_measurements->p().gyroscopeCovariance *= params->saturation_mult;
            imu_measurements->integrateMeasurement(a, w, dt);
            imu_measurements->p().accelerometerCovariance = cov_acc;
            imu_measurements->p().gyroscopeCovariance = cov_gyro;
        } else {
            imu_measurements->integrateMeasurement(a, w, dt);
        }

        last_stamp = imu_stamp;
        (*num_integrated)++;
    }

    const double dt = end_time - last_stamp;
    if (dt > 0.0) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = imu_itr == imu_queue.end() ? *(imu_itr - 1) : *imu_itr;
        const auto& a = last_imu_frame.block<3, 1>(1, 0);
        const auto& w = last_imu_frame.block<3, 1>(4, 0);

        bool saturated = a.cwiseAbs().maxCoeff() > params->acc_saturation_thresh ||
            w.cwiseAbs().maxCoeff() > params->gyro_saturation_thresh;

        if (saturated) {
            logger::info("imu_integration", "IMU saturation detected: |acc| = {:.2f}, |gyro| = {:.2f}", a.norm(), w.norm());
            auto cov_acc = imu_measurements->p().accelerometerCovariance;
            auto cov_gyro = imu_measurements->p().gyroscopeCovariance;
            imu_measurements->p().accelerometerCovariance *= params->saturation_mult;
            imu_measurements->p().gyroscopeCovariance *= params->saturation_mult;
            imu_measurements->integrateMeasurement(a, w, dt);
            imu_measurements->p().accelerometerCovariance = cov_acc;
            imu_measurements->p().gyroscopeCovariance = cov_gyro;
        } else {
            imu_measurements->integrateMeasurement(a, w, dt);
        }
    }

    return cursor;
}

int IMUIntegration::integrate_imu(
    double start_time,
    double end_time,
    const gtsam::NavState& state,
    const gtsam::imuBias::ConstantBias& bias,
    std::vector<double>& pred_times,
    std::vector<Eigen::Isometry3d>& pred_poses
) {
    imu_measurements->resetIntegrationAndSetBias(bias);

    pred_times.emplace_back(start_time);
    pred_poses.emplace_back(state.pose().matrix());

    int cursor = 0;
    auto imu_itr = imu_queue.begin();
    double last_stamp = start_time;

    if (imu_itr == imu_queue.end()) {
        pred_times.emplace_back(end_time);
        pred_poses.emplace_back(state.pose().matrix());
        return cursor;
    }

    for (; imu_itr != imu_queue.end(); imu_itr++, cursor++) {
        const auto& imu_frame = *imu_itr;
        const double imu_stamp = imu_frame[0];
        if (imu_stamp > end_time) {
            break;
        }

        const double dt = imu_stamp - last_stamp;
        if (dt <= 0.0) {
            continue;
        }

        const auto& a = imu_frame.block<3, 1>(1, 0);
        const auto& w = imu_frame.block<3, 1>(4, 0);

        bool saturated = a.cwiseAbs().maxCoeff() > params->acc_saturation_thresh ||
            w.cwiseAbs().maxCoeff() > params->gyro_saturation_thresh;

        if (saturated) {
            logger::info("imu_integration", "IMU saturation detected: |acc| = {:.2f}, |gyro| = {:.2f}", a.norm(), w.norm());
            auto cov_acc = imu_measurements->p().accelerometerCovariance;
            auto cov_gyro = imu_measurements->p().gyroscopeCovariance;
            imu_measurements->p().accelerometerCovariance *= params->saturation_mult;
            imu_measurements->p().gyroscopeCovariance *= params->saturation_mult;
            imu_measurements->integrateMeasurement(a, w, dt);
            imu_measurements->p().accelerometerCovariance = cov_acc;
            imu_measurements->p().gyroscopeCovariance = cov_gyro;
        } else {
            imu_measurements->integrateMeasurement(a, w, dt);
        }

        auto predicted = imu_measurements->predict(state, bias);
        pred_times.emplace_back(imu_stamp);
        pred_poses.emplace_back(predicted.pose().matrix());
        last_stamp = imu_stamp;
    }

    const double dt = end_time - last_stamp;
    if (dt > 0.0) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = imu_itr == imu_queue.end() ? *(imu_itr - 1) : *imu_itr;
        const auto& a = last_imu_frame.block<3, 1>(1, 0);
        const auto& w = last_imu_frame.block<3, 1>(4, 0);

        bool saturated = a.cwiseAbs().maxCoeff() > params->acc_saturation_thresh ||
            w.cwiseAbs().maxCoeff() > params->gyro_saturation_thresh;

        if (saturated) {
            logger::info("imu_integration", "IMU saturation detected: |acc| = {:.2f}, |gyro| = {:.2f}", a.norm(), w.norm());
            auto cov_acc = imu_measurements->p().accelerometerCovariance;
            auto cov_gyro = imu_measurements->p().gyroscopeCovariance;
            imu_measurements->p().accelerometerCovariance *= params->saturation_mult;
            imu_measurements->p().gyroscopeCovariance *= params->saturation_mult;
            imu_measurements->integrateMeasurement(a, w, dt);
            imu_measurements->p().accelerometerCovariance = cov_acc;
            imu_measurements->p().gyroscopeCovariance = cov_gyro;
        } else {
            imu_measurements->integrateMeasurement(a, w, dt);
        }

        auto predicted = imu_measurements->predict(state, bias);
        pred_times.emplace_back(end_time);
        pred_poses.emplace_back(predicted.pose().matrix());
    }

    return cursor;
}

int IMUIntegration::find_imu_data(
    double start_time,
    double end_time,
    std::vector<double>& delta_times,
    std::vector<Eigen::Matrix<double, 7, 1>>& imu_data
) {
    int cursor = 0;
    auto imu_itr = imu_queue.begin();
    double last_stamp = start_time;

    if (imu_itr == imu_queue.end()) {
        return cursor;
    }

    for (; imu_itr != imu_queue.end(); imu_itr++, cursor++) {
        const auto& imu_frame = *imu_itr;
        const double imu_stamp = imu_frame[0];
        if (imu_stamp > end_time) {
            break;
        }

        const double dt = imu_stamp - last_stamp;
        if (dt <= 0.0) {
            continue;
        }

        delta_times.emplace_back(dt);
        imu_data.emplace_back(imu_frame);
        last_stamp = imu_stamp;
    }

    const double dt = end_time - last_stamp;
    if (dt > 0.0) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = imu_itr == imu_queue.end() ? *(imu_itr - 1) : *imu_itr;
        delta_times.emplace_back(dt);
        imu_data.emplace_back(last_imu_frame);
    }

    return cursor;
}

void IMUIntegration::erase_imu_data(int last) {
    imu_queue.erase(imu_queue.begin(), imu_queue.begin() + last);
}

const gtsam::PreintegratedImuMeasurements& IMUIntegration::integrated_measurements() const {
    return *imu_measurements;
}

const std::deque<Eigen::Matrix<double, 7, 1>>& IMUIntegration::imu_data_in_queue() const {
    return imu_queue;
}

} // namespace small_glim