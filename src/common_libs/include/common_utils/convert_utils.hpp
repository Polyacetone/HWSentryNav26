#pragma once

#if __has_include(<opencv2/core/types.hpp>)
#include <opencv2/core/types.hpp>
#define HAVE_OPENCV
#endif

#if __has_include(<eigen3/Eigen/Dense>)
#include <eigen3/Eigen/Dense>
#define HAVE_EIGEN
#endif

#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#define HAVE_TF2_MSGS
#endif

#if __has_include(<geometry_msgs/msg/point.hpp>)
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#define HAVE_GEOMETRY_MSGS
#endif

#if __has_include(<pcl/point_types.h>)
#include <pcl/point_types.h>
#define HAVE_PCL
#endif

namespace utils {
template<typename Point> struct PointTraits;
#ifdef HAVE_OPENCV
template<> struct PointTraits<cv::Point3f> {
    static auto x(const cv::Point3f& p) { return p.x; }
    static auto y(const cv::Point3f& p) { return p.y; }
    static auto z(const cv::Point3f& p) { return p.z; }
    template<std::convertible_to<float> T> static cv::Point3f create(T x, T y, T z) {
        return cv::Point3f(x, y, z);
    }
};
template<> struct PointTraits<cv::Point3d> {
    static auto x(const cv::Point3d& p) { return p.x; }
    static auto y(const cv::Point3d& p) { return p.y; }
    static auto z(const cv::Point3d& p) { return p.z; }
    template<std::convertible_to<double> T> static cv::Point3d create(T x, T y, T z) {
        return cv::Point3d(x, y, z);
    }
};
#endif
#ifdef HAVE_GEOMETRY_MSGS
template<> struct PointTraits<geometry_msgs::msg::Point> {
    static auto x(const geometry_msgs::msg::Point& p) { return p.x; }
    static auto y(const geometry_msgs::msg::Point& p) { return p.y; }
    static auto z(const geometry_msgs::msg::Point& p) { return p.z; }
    template<std::convertible_to<double> T> static geometry_msgs::msg::Point create(T x, T y, T z) {
        geometry_msgs::msg::Point p;
        p.x = static_cast<double>(x);
        p.y = static_cast<double>(y);
        p.z = static_cast<double>(z);
        return p;
    }
};
template<> struct PointTraits<geometry_msgs::msg::Point32> {
    static auto x(const geometry_msgs::msg::Point32& p) { return p.x; }
    static auto y(const geometry_msgs::msg::Point32& p) { return p.y; }
    static auto z(const geometry_msgs::msg::Point32& p) { return p.z; }
    template<std::convertible_to<float> T> static geometry_msgs::msg::Point32 create(T x, T y, T z) {
        geometry_msgs::msg::Point32 p;
        p.x = static_cast<float>(x);
        p.y = static_cast<float>(y);
        p.z = static_cast<float>(z);
        return p;
    }
};
template<> struct PointTraits<geometry_msgs::msg::Vector3> {
    static auto x(const geometry_msgs::msg::Vector3& p) { return p.x; }
    static auto y(const geometry_msgs::msg::Vector3& p) { return p.y; }
    static auto z(const geometry_msgs::msg::Vector3& p) { return p.z; }
    template<std::convertible_to<double> T> static geometry_msgs::msg::Vector3 create(T x, T y, T z) {
        geometry_msgs::msg::Vector3 p;
        p.x = static_cast<double>(x);
        p.y = static_cast<double>(y);
        p.z = static_cast<double>(z);
        return p;
    }
};
#endif
#ifdef HAVE_TF2_MSGS
template<> struct PointTraits<tf2::Vector3> {
    static auto x(const tf2::Vector3& p) { return p.x(); }
    static auto y(const tf2::Vector3& p) { return p.y(); }
    static auto z(const tf2::Vector3& p) { return p.z(); }
    template<std::convertible_to<double> T> static tf2::Vector3 create(T x, T y, T z) {
        return tf2::Vector3(x, y, z);
    }
};
#endif
#ifdef HAVE_EIGEN
template<> struct PointTraits<Eigen::Vector3d> {
    static auto x(const Eigen::Vector3d& p) { return p.x(); }
    static auto y(const Eigen::Vector3d& p) { return p.y(); }
    static auto z(const Eigen::Vector3d& p) { return p.z(); }
    template<std::convertible_to<double> T> static Eigen::Vector3d create(T x, T y, T z) {
        return Eigen::Vector3d(x, y, z);
    }
};
template<> struct PointTraits<Eigen::Vector3f> {
    static auto x(const Eigen::Vector3f& p) { return p.x(); }
    static auto y(const Eigen::Vector3f& p) { return p.y(); }
    static auto z(const Eigen::Vector3f& p) { return p.z(); }
    template<std::convertible_to<float> T> static Eigen::Vector3f create(T x, T y, T z) {
        return Eigen::Vector3f(x, y, z);
    }
};
#endif
#ifdef HAVE_PCL
template<> struct PointTraits<pcl::PointXYZ> {
    static auto x(const pcl::PointXYZ& p) { return p.x; }
    static auto y(const pcl::PointXYZ& p) { return p.y; }
    static auto z(const pcl::PointXYZ& p) { return p.z; }
    template<std::convertible_to<float> T> static pcl::PointXYZ create(T x, T y, T z) {
        return pcl::PointXYZ(x, y, z);
    }
};
#endif
template<typename T>
concept PointLike = requires(const T& p) {
    { PointTraits<T>::x(p) } -> std::floating_point;
    { PointTraits<T>::y(p) } -> std::floating_point;
    { PointTraits<T>::z(p) } -> std::floating_point;
};
template<PointLike To, PointLike From>
static constexpr To convert_to(const From& src) {
    return PointTraits<To>::create(
        PointTraits<From>::x(src),
        PointTraits<From>::y(src),
        PointTraits<From>::z(src)
    );
}
template<PointLike To, PointLike From>
static constexpr void convert_to(const From& src, To& dst) {
    dst = PointTraits<To>::create(
        PointTraits<From>::x(src),
        PointTraits<From>::y(src),
        PointTraits<From>::z(src)
    );
}
}

namespace utils {
template<typename Quaternion> struct QuaternionTraits;
#ifdef HAVE_GEOMETRY_MSGS
template<> struct QuaternionTraits<geometry_msgs::msg::Quaternion> {
    static auto x(const geometry_msgs::msg::Quaternion& p) { return p.x; }
    static auto y(const geometry_msgs::msg::Quaternion& p) { return p.y; }
    static auto z(const geometry_msgs::msg::Quaternion& p) { return p.z; }
    static auto w(const geometry_msgs::msg::Quaternion& p) { return p.w; }
    template<std::convertible_to<double> T> static geometry_msgs::msg::Quaternion create(T x, T y, T z, T w) {
        geometry_msgs::msg::Quaternion p;
        p.x = static_cast<double>(x);
        p.y = static_cast<double>(y);
        p.z = static_cast<double>(z);
        p.w = static_cast<double>(w);
        return p;
    }
};
#endif
#ifdef HAVE_TF2_MSGS
template<> struct QuaternionTraits<tf2::Quaternion> {
    static auto x(const tf2::Quaternion& p) { return p.x(); }
    static auto y(const tf2::Quaternion& p) { return p.y(); }
    static auto z(const tf2::Quaternion& p) { return p.z(); }
    static auto w(const tf2::Quaternion& p) { return p.w(); }
    template<std::convertible_to<double> T> static tf2::Quaternion create(T x, T y, T z, T w) {
        return tf2::Quaternion(x, y, z, w);
    }
};
#endif
#ifdef HAVE_EIGEN
template<> struct QuaternionTraits<Eigen::Quaterniond> {
    static auto x(const Eigen::Quaterniond& p) { return p.x(); }
    static auto y(const Eigen::Quaterniond& p) { return p.y(); }
    static auto z(const Eigen::Quaterniond& p) { return p.z(); }
    static auto w(const Eigen::Quaterniond& p) { return p.w(); }
    template<std::convertible_to<double> T> static Eigen::Quaterniond create(T x, T y, T z, T w) {
        return Eigen::Quaterniond(w, x, y, z);
    }
};
template<> struct QuaternionTraits<Eigen::Quaternionf> {
    static auto x(const Eigen::Quaternionf& p) { return p.x(); }
    static auto y(const Eigen::Quaternionf& p) { return p.y(); }
    static auto z(const Eigen::Quaternionf& p) { return p.z(); }
    static auto w(const Eigen::Quaternionf& p) { return p.w(); }
    template<std::convertible_to<float> T> static Eigen::Quaternionf create(T x, T y, T z, T w) {
        return Eigen::Quaternionf(w, x, y, z);
    }
};
#endif
template<typename T>
concept QuaternionLike = requires(const T& p) {
    { QuaternionTraits<T>::x(p) } -> std::floating_point;
    { QuaternionTraits<T>::y(p) } -> std::floating_point;
    { QuaternionTraits<T>::z(p) } -> std::floating_point;
    { QuaternionTraits<T>::w(p) } -> std::floating_point;
};
template<QuaternionLike To, QuaternionLike From>
static constexpr To convert_to(const From& src) {
    return QuaternionTraits<To>::create(
        QuaternionTraits<From>::x(src),
        QuaternionTraits<From>::y(src),
        QuaternionTraits<From>::z(src),
        QuaternionTraits<From>::w(src)
    );
}
template<QuaternionLike To, QuaternionLike From>
static constexpr void convert_to(const From& src, To& dst) {
    dst = QuaternionTraits<To>::create(
        QuaternionTraits<From>::x(src),
        QuaternionTraits<From>::y(src),
        QuaternionTraits<From>::z(src),
        QuaternionTraits<From>::w(src)
    );
}
}

namespace utils {
template<typename Pose> struct PoseTraits;
#ifdef HAVE_TF2_MSGS
template<> struct PoseTraits<tf2::Transform> {
    static auto translation(const tf2::Transform& t) { return t.getOrigin(); }
    static auto rotation(const tf2::Transform& t) { return t.getRotation(); }
    template<PointLike T, QuaternionLike R>
    static tf2::Transform create(T translation, R rotation) {
        return tf2::Transform(
            convert_to<tf2::Quaternion>(rotation),
            convert_to<tf2::Vector3>(translation)
        );
    }
};
#endif
#ifdef HAVE_GEOMETRY_MSGS
template<> struct PoseTraits<geometry_msgs::msg::Transform> {
    static auto translation(const geometry_msgs::msg::Transform& t) { return t.translation; }
    static auto rotation(const geometry_msgs::msg::Transform& t) { return t.rotation; }
    template<PointLike T, QuaternionLike R>
    static geometry_msgs::msg::Transform create(T translation, R rotation) {
        geometry_msgs::msg::Transform t;
        t.translation = convert_to<geometry_msgs::msg::Vector3>(translation);
        t.rotation = convert_to<geometry_msgs::msg::Quaternion>(rotation);
        return t;
    }
};
template<> struct PoseTraits<geometry_msgs::msg::Pose> {
    static auto translation(const geometry_msgs::msg::Pose& t) { return t.position; }
    static auto rotation(const geometry_msgs::msg::Pose& t) { return t.orientation; }
    template<PointLike T, QuaternionLike R>
    static geometry_msgs::msg::Pose create(T translation, R rotation) {
        geometry_msgs::msg::Pose t;
        t.position = convert_to<geometry_msgs::msg::Point>(translation);
        t.orientation = convert_to<geometry_msgs::msg::Quaternion>(rotation);
        return t;
    }
};
#endif
template<typename T>
concept PoseLike = requires(const T& p) {
    { PoseTraits<T>::translation(p) } -> PointLike;
    { PoseTraits<T>::rotation(p) } -> QuaternionLike;
};
template<PoseLike To, PoseLike From>
static constexpr To convert_to(const From& src) {
    return PoseTraits<To>::create(
        PoseTraits<From>::translation(src),
        PoseTraits<From>::rotation(src)
    );
}
template<PoseLike To, PoseLike From>
static constexpr void convert_to(const From& src, To& dst) {
    dst = PoseTraits<To>::create(
        PoseTraits<From>::translation(src),
        PoseTraits<From>::rotation(src)
    );
}
}