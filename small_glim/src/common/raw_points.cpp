#include <small_glim/common/raw_points.hpp>
#include <format>

namespace small_glim {

template <typename T>
Eigen::Vector4d get_vec4(const void* x, const void* y, const void* z) {
    return Eigen::Vector4d(*reinterpret_cast<const T*>(x), *reinterpret_cast<const T*>(y), *reinterpret_cast<const T*>(z), 1.0);
}

RawPoints::RawPoints(
    const sensor_msgs::msg::PointCloud2& points_msg,
    const std::string& intensity_channel,
    const std::string& ring_channel
) {
    using sensor_msgs::msg::PointField;
    size_t num_points = points_msg.width * points_msg.height;

    uint8_t x_type = 0;
    uint8_t y_type = 0;
    uint8_t z_type = 0;
    uint8_t time_type = 0; // ouster and livox
    uint8_t intensity_type = 0;
    uint8_t color_type = 0;
    uint8_t ring_type = 0;

    size_t x_offset = static_cast<size_t>(-1);
    size_t y_offset = static_cast<size_t>(-1);
    size_t z_offset = static_cast<size_t>(-1);
    size_t time_offset = static_cast<size_t>(-1);
    size_t intensity_offset = static_cast<size_t>(-1);
    size_t color_offset = static_cast<size_t>(-1);
    size_t ring_offset = static_cast<size_t>(-1);

    std::unordered_map<std::string, std::pair<uint8_t*, size_t*>> fields;
    fields["x"] = std::make_pair(&x_type, &x_offset);
    fields["y"] = std::make_pair(&y_type, &y_offset);
    fields["z"] = std::make_pair(&z_type, &z_offset);
    fields["t"] = std::make_pair(&time_type, &time_offset);
    fields["time"] = std::make_pair(&time_type, &time_offset);
    fields["time_stamp"] = std::make_pair(&time_type, &time_offset);
    fields["timestamp"] = std::make_pair(&time_type, &time_offset);
    fields[intensity_channel] = std::make_pair(&intensity_type, &intensity_offset);
    fields["rgba"] = std::make_pair(&color_type, &color_offset);
    fields[ring_channel] = std::make_pair(&ring_type, &ring_offset);

    for (const auto& field: points_msg.fields) {
        auto found = fields.find(field.name);
        if (found == fields.end()) {
            continue;
        }
        *found->second.first = field.datatype;
        *found->second.second = field.offset;
    }

    if (x_offset == static_cast<size_t>(-1) || y_offset == static_cast<size_t>(-1) || z_offset == static_cast<size_t>(-1)) {
        throw std::runtime_error("missing point coordinate fields");
    }

    if ((x_type != PointField::FLOAT32 && x_type != PointField::FLOAT64) || x_type != y_type || x_type != y_type) {
        throw std::runtime_error("unsupported points type");
    }

    points.resize(num_points);

    if (x_type == PointField::FLOAT32 && y_offset == x_offset + sizeof(float) && z_offset == y_offset + sizeof(float)) {
        // Special case: contiguous 3 floats
        for (size_t i = 0; i < num_points; i++) {
            const auto* x_ptr = &points_msg.data[points_msg.point_step * i + x_offset];
            points[i] << Eigen::Map<const Eigen::Vector3f>(reinterpret_cast<const float*>(x_ptr)).cast<double>(), 1.0;
        }
    } else if (x_type == PointField::FLOAT64 && y_offset == x_offset + sizeof(double) && z_offset == y_offset + sizeof(double)) {
        // Special case: contiguous 3 doubles
        for (size_t i = 0; i < num_points; i++) {
            const auto* x_ptr = &points_msg.data[points_msg.point_step * i + x_offset];
            points[i] << Eigen::Map<const Eigen::Vector3d>(reinterpret_cast<const double*>(x_ptr)), 1.0;
        }
    } else {
        for (size_t i = 0; i < num_points; i++) {
            const auto* x_ptr = &points_msg.data[points_msg.point_step * i + x_offset];
            const auto* y_ptr = &points_msg.data[points_msg.point_step * i + y_offset];
            const auto* z_ptr = &points_msg.data[points_msg.point_step * i + z_offset];

            if (x_type == PointField::FLOAT32) {
                points[i] = get_vec4<float>(x_ptr, y_ptr, z_ptr);
            } else {
                points[i] = get_vec4<double>(x_ptr, y_ptr, z_ptr);
            }
        }
    }

    if (time_offset != static_cast<size_t>(-1)) {
        times.resize(num_points);
        for (size_t i = 0; i < num_points; i++) {
            const auto* time_ptr = &points_msg.data[points_msg.point_step * i + time_offset];
            switch (time_type) {
                case PointField::UINT32: {
                    times[i] = *reinterpret_cast<const uint32_t*>(time_ptr) / 1e9;
                    break;
                }
                case PointField::FLOAT32: {
                    times[i] = *reinterpret_cast<const float*>(time_ptr);
                    break;
                }
                case PointField::FLOAT64: {
                    times[i] = *reinterpret_cast<const double*>(time_ptr);
                    break;
                }
                default: {
                    throw std::runtime_error(std::format("unsupported time type {}", time_type));
                }
            }
        }
    }

    if (intensity_offset != static_cast<size_t>(-1)) {
        intensities.resize(num_points);
        for (size_t i = 0; i < num_points; i++) {
            const auto* intensity_ptr = &points_msg.data[points_msg.point_step * i + intensity_offset];
            switch (intensity_type) {
                case PointField::UINT8: {
                    intensities[i] = *reinterpret_cast<const std::uint8_t*>(intensity_ptr);
                    break;
                }
                case PointField::UINT16: {
                    intensities[i] = *reinterpret_cast<const std::uint16_t*>(intensity_ptr);
                    break;
                }
                case PointField::UINT32: {
                    intensities[i] = *reinterpret_cast<const std::uint32_t*>(intensity_ptr);
                    break;
                }
                case PointField::FLOAT32: {
                    intensities[i] = *reinterpret_cast<const float*>(intensity_ptr);
                    break;
                }
                case PointField::FLOAT64: {
                    intensities[i] = *reinterpret_cast<const double*>(intensity_ptr);
                    break;
                }
                default: {
                    throw std::runtime_error(std::format("unsupported intensity type {}", intensity_type));
                }
            }
        }
    }

    if (color_offset != static_cast<size_t>(-1)) {
        if (color_type != PointField::UINT32) {
            throw std::runtime_error(std::format("unsupported color type {}", color_type));
        } else {
            colors.resize(num_points);
            for (size_t i = 0; i < num_points; i++) {
                const auto* color_ptr = &points_msg.data[points_msg.point_step * i + color_offset];
                colors[i] = Eigen::Matrix<unsigned char, 4, 1>(reinterpret_cast<const std::uint8_t*>(color_ptr)).cast<double>() / 255.0;
            }
        }
    }

    if (ring_offset != static_cast<size_t>(-1)) {
        rings.resize(num_points);
        for (size_t i = 0; i < num_points; i++) {
            const auto* ring_ptr = &points_msg.data[points_msg.point_step * i + ring_offset];
            switch (ring_type) {
                case PointField::UINT8: {
                    rings[i] = *reinterpret_cast<const std::uint8_t*>(ring_ptr);
                    break;
                }
                case PointField::UINT16: {
                    rings[i] = *reinterpret_cast<const std::uint16_t*>(ring_ptr);
                    break;
                }
                case PointField::UINT32: {
                    rings[i] = *reinterpret_cast<const std::uint32_t*>(ring_ptr);
                    break;
                }
                default: {
                    throw std::runtime_error(std::format("unsupported ring type {}", ring_type));
                }
            }
        }
    }

    stamp = points_msg.header.stamp.sec + points_msg.header.stamp.nanosec / 1e9;
}

}