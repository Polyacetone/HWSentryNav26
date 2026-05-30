#include "map_server/map_utils.hpp"
#include <fstream>
#include <cmath>
#include <msgpack.hpp>

namespace map_server::map_utils {

TerrainMapData load_terrain_msgpack(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open " + path);
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (buffer.empty()) {
        throw std::runtime_error("Empty msgpack file: " + path);
    }

    auto handle = msgpack::unpack(buffer.data(), buffer.size());
    auto obj = handle.get();
    if (obj.type != msgpack::type::MAP) {
        throw std::runtime_error("Msgpack root must be a MAP");
    }

    TerrainMapData data{};
    auto m = obj.via.map;
    for (uint32_t i = 0; i < m.size; i++) {
        auto key = m.ptr[i].key.as<std::string>();
        if (key == "width") {
            data.width = m.ptr[i].val.as<int>();
        } else if (key == "height") {
            data.height = m.ptr[i].val.as<int>();
        } else if (key == "resolution") {
            data.resolution = m.ptr[i].val.as<double>();
        } else if (key == "terrain") {
            data.terrain = m.ptr[i].val.as<std::vector<uint8_t>>();
        } else if (key == "direction") {
            data.direction = m.ptr[i].val.as<std::vector<uint8_t>>();
        }
    }

    const size_t expected = static_cast<size_t>(data.width) * static_cast<size_t>(data.height);
    if (data.terrain.size() != expected) {
        throw std::runtime_error(
            "terrain data size mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(data.terrain.size())
        );
    }
    if (data.direction.size() != expected) {
        throw std::runtime_error(
            "direction data size mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(data.direction.size())
        );
    }

    return data;
}

cv::Mat inflate_cost_map(
    const cv::Mat& source,
    const MapInflationParams& params
) {
    CV_Assert(source.type() == CV_8UC1);
    const int h = source.rows;
    const int w = source.cols;

    // 二值化: 非零值视为障碍物源
    cv::Mat bin_mask;
    cv::threshold(source, bin_mask, 0, 1, cv::THRESH_BINARY);

    // 距离变换（像素单位）
    cv::Mat dist_px;
    cv::distanceTransform(1 - bin_mask, dist_px, cv::DIST_L2, 3);

    cv::Mat out = source.clone();
    const float robot_r = static_cast<float>(params.robot_radius_px);
    const float cutoff_r = static_cast<float>(params.cutoff_radius_px);

    for (int y = 0; y < h; y++) {
        const float* dist_row = dist_px.ptr<float>(y);
        uint8_t* out_row = out.ptr<uint8_t>(y);
        for (int x = 0; x < w; x++) {
            if (bin_mask.at<uint8_t>(y, x)) continue;

            const float d = dist_row[x];
            if (d <= robot_r) {
                out_row[x] = 255;
            } else if (d <= cutoff_r) {
                const float v = 255.0f * static_cast<float>(
                    std::exp(-params.decay_alpha * static_cast<double>(d - robot_r)));
                out_row[x] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
            }
        }
    }
    return out;
}

void inflate_direction_field(
    const TerrainMapData& data,
    const MapInflationParams& params,
    cv::Mat& out_angle,
    cv::Mat& out_magnitude
) {
    const int h = data.height;
    const int w = data.width;
    const size_t N = static_cast<size_t>(h) * static_cast<size_t>(w);

    out_angle = cv::Mat::zeros(h, w, CV_8UC1);
    out_magnitude = cv::Mat::zeros(h, w, CV_8UC1);

    std::vector<float> sum_vx(N, 0.0f);
    std::vector<float> sum_vy(N, 0.0f);
    std::vector<float> max_mag(N, 0.0f);

    const int radius = params.cutoff_radius_px;
    const int robot_r = params.robot_radius_px;

    for (int sy = 0; sy < h; sy++) {
        for (int sx = 0; sx < w; sx++) {
            const size_t src_idx = static_cast<size_t>(sy) * static_cast<size_t>(w) + static_cast<size_t>(sx);
            if (!is_directional_label(data.terrain[src_idx])) continue;

            const double raw_angle = data.direction[src_idx] / 255.0 * 2.0 * M_PI;
            const float src_vx = static_cast<float>(std::cos(raw_angle));
            const float src_vy = static_cast<float>(std::sin(raw_angle));

            const int y0 = std::max(0, sy - radius);
            const int y1 = std::min(h, sy + radius + 1);
            const int x0 = std::max(0, sx - radius);
            const int x1 = std::min(w, sx + radius + 1);

            for (int ny = y0; ny < y1; ny++) {
                const int dy = ny - sy;
                const float dy2 = static_cast<float>(dy * dy);
                for (int nx = x0; nx < x1; nx++) {
                    const int dx = nx - sx;
                    const float dist = std::sqrt(dy2 + static_cast<float>(dx * dx));
                    if (dist > static_cast<float>(radius)) continue;

                    float mag = 1.0f;
                    if (dist > static_cast<float>(robot_r)) {
                        mag *= static_cast<float>(std::exp(
                            -params.decay_alpha * static_cast<double>(dist - static_cast<float>(robot_r))));
                    }

                    const size_t idx = static_cast<size_t>(ny) * static_cast<size_t>(w) + static_cast<size_t>(nx);
                    sum_vx[idx] += src_vx * mag;
                    sum_vy[idx] += src_vy * mag;
                    if (mag > max_mag[idx]) {
                        max_mag[idx] = mag;
                    }
                }
            }
        }
    }

    uint8_t* angle_row = out_angle.ptr<uint8_t>(0);
    uint8_t* mag_row = out_magnitude.ptr<uint8_t>(0);
    for (size_t i = 0; i < N; i++) {
        const float total = std::sqrt(sum_vx[i] * sum_vx[i] + sum_vy[i] * sum_vy[i]);
        if (total > 1e-12f) {
            const float final_mag = max_mag[i];
            const float fx = sum_vx[i] / total * final_mag;
            const float fy = sum_vy[i] / total * final_mag;
            double angle_rad = std::atan2(fy, fx);
            if (angle_rad < 0) angle_rad += 2.0 * M_PI;
            angle_row[i] = static_cast<uint8_t>(angle_rad / (2.0 * M_PI) * 255.0);
            mag_row[i] = static_cast<uint8_t>(std::clamp(final_mag * 255.0f, 0.0f, 255.0f));
        }
    }
}

void build_terrain_3chan(
    const cv::Mat& angle,
    const cv::Mat& magnitude,
    const cv::Mat& terrain,
    cv::Mat& out
) {
    std::vector<cv::Mat> channels = {angle, magnitude, terrain};
    cv::merge(channels, out);
}

} // namespace map_server::map_utils
