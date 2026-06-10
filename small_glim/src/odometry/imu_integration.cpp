#include <small_glim/odometry/imu_integration.hpp>
#include <small_glim/common/config.hpp>
#include <small_glim/common/logger.hpp>

namespace small_glim {

inline void update_saturation_axes(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& w,
    const double acc_thresh,
    const double gyro_thresh,
    IMUSaturationStatus* status
) {
    if (!status) return;

    const Eigen::Array3d abs_a = a.cwiseAbs().array();
    const Eigen::Array3d abs_w = w.cwiseAbs().array();

    for (int i = 0; i < 3; i++) {
        status->acc_axes[static_cast<size_t>(i)] = status->acc_axes[static_cast<size_t>(i)] || (abs_a[i] > acc_thresh);
        status->gyro_axes[static_cast<size_t>(i)] = status->gyro_axes[static_cast<size_t>(i)] || (abs_w[i] > gyro_thresh);
    }
}

inline void update_saturation_stats(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& w,
    const bool saturated,
    const bool count_as_imu_frame,
    IMUSaturationStatus* status
) {
    if (!status) return;

    if (count_as_imu_frame) {
        status->imu_count++;
        if (saturated) {
            status->saturated_imu_count++;
        }
    }

    if (!saturated) {
        return;
    }

    if (!count_as_imu_frame) {
        return;
    }

    const Eigen::Array3d abs_a = a.cwiseAbs().array();
    const Eigen::Array3d abs_w = w.cwiseAbs().array();

    for (int i = 0; i < 3; i++) {
        const auto idx = static_cast<size_t>(i);
        status->acc_abs_max[idx] = std::max(status->acc_abs_max[idx], abs_a[i]);
        status->gyro_abs_max[idx] = std::max(status->gyro_abs_max[idx], abs_w[i]);
        status->acc_abs_sum[idx] += abs_a[i];
        status->gyro_abs_sum[idx] += abs_w[i];
    }

    const double acc_norm = a.norm();
    const double gyro_norm = w.norm();
    status->acc_norm_max = std::max(status->acc_norm_max, acc_norm);
    status->gyro_norm_max = std::max(status->gyro_norm_max, gyro_norm);
    status->acc_norm_sum += acc_norm;
    status->gyro_norm_sum += gyro_norm;
}

inline Eigen::Matrix3d axis_scaled_cov(
    const Eigen::Matrix3d& cov,
    const std::array<bool, 3>& sat_axes,
    const double mult
) {
    if (!(sat_axes[0] || sat_axes[1] || sat_axes[2])) {
        return cov;
    }

    const double s0 = sat_axes[0] ? mult : 1.0;
    const double s1 = sat_axes[1] ? mult : 1.0;
    const double s2 = sat_axes[2] ? mult : 1.0;
    const Eigen::Matrix3d S = (Eigen::Vector3d(s0, s1, s2)).asDiagonal();
    return S * cov * S;
}

inline void integrate_with_optional_saturation(
    gtsam::PreintegratedImuMeasurements& pim,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& w,
    const double dt,
    const IMUIntegrationParams& params,
    const bool count_as_imu_frame,
    IMUSaturationStatus* status
) {
    IMUSaturationStatus local;
    update_saturation_axes(a, w, params.acc_saturation_thresh, params.gyro_saturation_thresh, &local);
    update_saturation_axes(a, w, params.acc_saturation_thresh, params.gyro_saturation_thresh, status);

    const bool saturated = local.any();
    update_saturation_stats(a, w, saturated, count_as_imu_frame, status);
    if (!saturated) {
        pim.integrateMeasurement(a, w, dt);
        return;
    }

    const auto cov_acc_orig = pim.p().accelerometerCovariance;
    const auto cov_gyro_orig = pim.p().gyroscopeCovariance;

    pim.p().accelerometerCovariance = axis_scaled_cov(cov_acc_orig, local.acc_axes, params.saturation_mult);
    pim.p().gyroscopeCovariance = axis_scaled_cov(cov_gyro_orig, local.gyro_axes, params.saturation_mult);

    pim.integrateMeasurement(a, w, dt);

    pim.p().accelerometerCovariance = cov_acc_orig;
    pim.p().gyroscopeCovariance = cov_gyro_orig;
}

inline void log_saturation_summary_once(
    const double start_time,
    const double end_time,
    const IMUIntegrationParams& params,
    const IMUSaturationStatus& st
) {
    if (!st.any()) {
        return;
    }
    if (st.saturated_imu_count == 0 || st.imu_count == 0) {
        logger::info(
            "imu_integration",
            "IMU saturation summary: t=[{:.6f},{:.6f}] imu={}/{} acc=[{}{}{}] gyro=[{}{}{}] (thresh acc>{:.3f} gyro>{:.3f})",
            start_time,
            end_time,
            st.saturated_imu_count,
            st.imu_count,
            st.acc_axes[0] ? "x" : "-",
            st.acc_axes[1] ? "y" : "-",
            st.acc_axes[2] ? "z" : "-",
            st.gyro_axes[0] ? "x" : "-",
            st.gyro_axes[1] ? "y" : "-",
            st.gyro_axes[2] ? "z" : "-",
            params.acc_saturation_thresh,
            params.gyro_saturation_thresh
        );
        return;
    }

    const double n = static_cast<double>(st.saturated_imu_count);
    const std::array<double, 3> acc_abs_mean {
        st.acc_abs_sum[0] / n,
        st.acc_abs_sum[1] / n,
        st.acc_abs_sum[2] / n,
    };
    const std::array<double, 3> gyro_abs_mean {
        st.gyro_abs_sum[0] / n,
        st.gyro_abs_sum[1] / n,
        st.gyro_abs_sum[2] / n,
    };

    logger::info(
        "imu_integration",
        "IMU saturation summary: t=[{:.6f},{:.6f}] imu={}/{} acc=[{}{}{}] gyro=[{}{}{}] "
        "|a|max=[{:.3f},{:.3f},{:.3f}] mean=[{:.3f},{:.3f},{:.3f}] "
        "|w|max=[{:.3f},{:.3f},{:.3f}] mean=[{:.3f},{:.3f},{:.3f}] "
        "|a|max={:.3f} |w|max={:.3f} (mult={:.1f})",
        start_time,
        end_time,
        st.saturated_imu_count,
        st.imu_count,
        st.acc_axes[0] ? "x" : "-",
        st.acc_axes[1] ? "y" : "-",
        st.acc_axes[2] ? "z" : "-",
        st.gyro_axes[0] ? "x" : "-",
        st.gyro_axes[1] ? "y" : "-",
        st.gyro_axes[2] ? "z" : "-",
        st.acc_abs_max[0],
        st.acc_abs_max[1],
        st.acc_abs_max[2],
        acc_abs_mean[0],
        acc_abs_mean[1],
        acc_abs_mean[2],
        st.gyro_abs_max[0],
        st.gyro_abs_max[1],
        st.gyro_abs_max[2],
        gyro_abs_mean[0],
        gyro_abs_mean[1],
        gyro_abs_mean[2],
        st.acc_norm_max,
        st.gyro_norm_max,
        params.saturation_mult
    );
}

IMUIntegrationParams::IMUIntegrationParams(const Config::Ptr config) {
    upright = config->param<bool>("sensors.imu_upright");
    acc_noise = config->param<double>("sensors.imu_acc_noise");
    gyro_noise = config->param<double>("sensors.imu_gyro_noise");
    int_noise = config->param<double>("sensors.imu_int_noise");
    acc_saturation_thresh = config->param<double>("sensors.imu_acc_saturation_thresh");
    gyro_saturation_thresh = config->param<double>("sensors.imu_gyro_saturation_thresh");
    saturation_mult = config->param<double>("sensors.imu_saturation_mult");
    max_integration_dt = config->param<double>("sensors.imu_max_integration_dt");
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

size_t IMUIntegration::integrate_imu(
    double start_time,
    double end_time,
    const gtsam::imuBias::ConstantBias& bias,
    size_t* num_integrated,
    IMUSaturationStatus* saturation_status
) {
    *num_integrated = 0;
    imu_measurements->resetIntegrationAndSetBias(bias);

    if (saturation_status) {
        *saturation_status = IMUSaturationStatus{};
    }

    size_t cursor = 0;
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
        if (!std::isfinite(dt) || dt > params->max_integration_dt) {
            logger::warn("imu_integration", "skip IMU integration with invalid dt={:.6f}", dt);
            continue;
        }

        const auto& a = imu_frame.block<3, 1>(1, 0);
        const auto& w = imu_frame.block<3, 1>(4, 0);

        integrate_with_optional_saturation(
            *imu_measurements,
            a,
            w,
            dt,
            *params,
            true,
            saturation_status
        );

        last_stamp = imu_stamp;
        (*num_integrated)++;
    }

    const double dt = end_time - last_stamp;
    if (dt > 0.0 && dt <= params->max_integration_dt) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = imu_itr == imu_queue.end() ? *(imu_itr - 1) : *imu_itr;
        const auto& a = last_imu_frame.block<3, 1>(1, 0);
        const auto& w = last_imu_frame.block<3, 1>(4, 0);

        integrate_with_optional_saturation(
            *imu_measurements,
            a,
            w,
            dt,
            *params,
            false,
            saturation_status
        );
    }

    if (saturation_status) {
        log_saturation_summary_once(start_time, end_time, *params, *saturation_status);
    }

    return cursor;
}

size_t IMUIntegration::integrate_imu(
    double start_time,
    double end_time,
    const gtsam::NavState& state,
    const gtsam::imuBias::ConstantBias& bias,
    std::vector<double>& pred_times,
    std::vector<Eigen::Isometry3d>& pred_poses,
    IMUSaturationStatus* saturation_status
) {
    imu_measurements->resetIntegrationAndSetBias(bias);

    if (saturation_status) {
        *saturation_status = IMUSaturationStatus{};
    }

    pred_times.emplace_back(start_time);
    pred_poses.emplace_back(state.pose().matrix());

    size_t cursor = 0;
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
        if (!std::isfinite(dt) || dt > params->max_integration_dt) {
            logger::warn("imu_integration", "skip IMU pose prediction with invalid dt={:.6f}", dt);
            continue;
        }

        const auto& a = imu_frame.block<3, 1>(1, 0);
        const auto& w = imu_frame.block<3, 1>(4, 0);

        integrate_with_optional_saturation(
            *imu_measurements,
            a,
            w,
            dt,
            *params,
            true,
            saturation_status
        );

        auto predicted = imu_measurements->predict(state, bias);
        pred_times.emplace_back(imu_stamp);
        pred_poses.emplace_back(predicted.pose().matrix());
        last_stamp = imu_stamp;
    }

    const double dt = end_time - last_stamp;
    if (dt > 0.0 && dt <= params->max_integration_dt) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = imu_itr == imu_queue.end() ? *(imu_itr - 1) : *imu_itr;
        const auto& a = last_imu_frame.block<3, 1>(1, 0);
        const auto& w = last_imu_frame.block<3, 1>(4, 0);

        integrate_with_optional_saturation(
            *imu_measurements,
            a,
            w,
            dt,
            *params,
            false,
            saturation_status
        );

        auto predicted = imu_measurements->predict(state, bias);
        pred_times.emplace_back(end_time);
        pred_poses.emplace_back(predicted.pose().matrix());
    }

    if (saturation_status) {
        log_saturation_summary_once(start_time, end_time, *params, *saturation_status);
    }

    return cursor;
}

size_t IMUIntegration::find_imu_data(
    double start_time,
    double end_time,
    std::vector<double>& delta_times,
    std::vector<Eigen::Matrix<double, 7, 1>>& imu_data
) {
    size_t cursor = 0;
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
        if (!std::isfinite(dt) || dt > params->max_integration_dt) {
            logger::warn("imu_integration", "skip IMU lookup with invalid dt={:.6f}", dt);
            continue;
        }

        delta_times.emplace_back(dt);
        imu_data.emplace_back(imu_frame);
        last_stamp = imu_stamp;
    }

    const double dt = end_time - last_stamp;
    if (dt > 0.0 && dt <= params->max_integration_dt) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = imu_itr == imu_queue.end() ? *(imu_itr - 1) : *imu_itr;
        delta_times.emplace_back(dt);
        imu_data.emplace_back(last_imu_frame);
    }

    return cursor;
}

void IMUIntegration::erase_imu_data(size_t last) {
    imu_queue.erase(imu_queue.begin(), imu_queue.begin() + static_cast<int64_t>(last));
}

const gtsam::PreintegratedImuMeasurements& IMUIntegration::integrated_measurements() const {
    return *imu_measurements;
}

const std::deque<Eigen::Matrix<double, 7, 1>>& IMUIntegration::imu_data_in_queue() const {
    return imu_queue;
}

} // namespace small_glim
