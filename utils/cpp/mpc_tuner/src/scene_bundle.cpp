#include <mpc_tuner/scene_bundle.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include <msgpack.hpp>
#include <opencv2/core.hpp>

namespace mpc_tuner {
namespace {

using Packer = msgpack::packer<std::ofstream>;

const msgpack::object& field(const msgpack::object& object, const std::string_view key) {
    if (object.type != msgpack::type::MAP) {
        throw std::runtime_error("Expected msgpack map while reading '" + std::string(key) + "'");
    }
    for (uint32_t i = 0; i < object.via.map.size; ++i) {
        const auto& item = object.via.map.ptr[i];
        if (item.key.type == msgpack::type::STR
            && std::string_view(item.key.via.str.ptr, item.key.via.str.size) == key) {
            return item.val;
        }
    }
    throw std::runtime_error("Missing msgpack field: " + std::string(key));
}

template<typename T>
T value(const msgpack::object& object, const std::string_view key) {
    return field(object, key).as<T>();
}

std::vector<uint8_t> binary(const msgpack::object& object, const std::string_view key) {
    const auto& data = field(object, key);
    if (data.type != msgpack::type::BIN) {
        throw std::runtime_error("Field '" + std::string(key) + "' must be binary");
    }
    const auto* begin = reinterpret_cast<const uint8_t*>(data.via.bin.ptr);
    return std::vector<uint8_t>(begin, begin + data.via.bin.size);
}

void pack_key(Packer& packer, const std::string_view key) {
    packer.pack_str(static_cast<uint32_t>(key.size()));
    packer.pack_str_body(key.data(), static_cast<uint32_t>(key.size()));
}

void pack_binary(Packer& packer, const std::vector<uint8_t>& data) {
    packer.pack_bin(static_cast<uint32_t>(data.size()));
    packer.pack_bin_body(reinterpret_cast<const char*>(data.data()), static_cast<uint32_t>(data.size()));
}

void pack_vector2(Packer& packer, const Eigen::Vector2d& point) {
    packer.pack_array(2);
    packer.pack(point.x());
    packer.pack(point.y());
}

Eigen::Vector2d vector2(const msgpack::object& object) {
    if (object.type != msgpack::type::ARRAY || object.via.array.size != 2) {
        throw std::runtime_error("Expected a two-element vector");
    }
    return {object.via.array.ptr[0].as<double>(), object.via.array.ptr[1].as<double>()};
}

void pack_traversal(Packer& packer, const nav_executor::StepTraversalConstraint& constraint) {
    packer.pack_map(9);
    pack_key(packer, "speed_min"); packer.pack(constraint.speed_min);
    pack_key(packer, "speed_max"); packer.pack(constraint.speed_max);
    pack_key(packer, "approach_start_u"); packer.pack(constraint.approach_start_u);
    pack_key(packer, "commit_u"); packer.pack(constraint.commit_u);
    pack_key(packer, "step_enter_u"); packer.pack(constraint.step_enter_u);
    pack_key(packer, "exit_u"); packer.pack(constraint.exit_u);
    pack_key(packer, "gate_start_u"); packer.pack(constraint.gate_start_u);
    pack_key(packer, "gate_end_u"); packer.pack(constraint.gate_end_u);
    pack_key(packer, "dir_map"); pack_vector2(packer, constraint.dir_map);
}

nav_executor::StepTraversalConstraint traversal(const msgpack::object& object) {
    return {
        .speed_min = value<double>(object, "speed_min"),
        .speed_max = value<double>(object, "speed_max"),
        .approach_start_u = value<double>(object, "approach_start_u"),
        .commit_u = value<double>(object, "commit_u"),
        .step_enter_u = value<double>(object, "step_enter_u"),
        .exit_u = value<double>(object, "exit_u"),
        .gate_start_u = value<double>(object, "gate_start_u"),
        .gate_end_u = value<double>(object, "gate_end_u"),
        .dir_map = vector2(field(object, "dir_map")),
    };
}

void pack_step_segment(Packer& packer, const nav_executor::StepPlanSegment& segment) {
    packer.pack_map(15);
    pack_key(packer, "prepare_u"); packer.pack(segment.prepare_u);
    pack_key(packer, "active_u"); packer.pack(segment.active_u);
    pack_key(packer, "commit_u"); packer.pack(segment.commit_u);
    pack_key(packer, "step_enter_u"); packer.pack(segment.step_enter_u);
    pack_key(packer, "step_exit_u"); packer.pack(segment.step_exit_u);
    pack_key(packer, "release_u"); packer.pack(segment.release_u);
    pack_key(packer, "step_enter_pos"); pack_vector2(packer, segment.step_enter_pos_map);
    pack_key(packer, "step_exit_pos"); pack_vector2(packer, segment.step_exit_pos_map);
    pack_key(packer, "dir_map"); pack_vector2(packer, segment.dir_map);
    pack_key(packer, "step_direction"); packer.pack(static_cast<uint8_t>(segment.direction));
    pack_key(packer, "chassis_mode"); packer.pack(segment.chassis_command.mode);
    pack_key(packer, "capability"); packer.pack(static_cast<uint8_t>(segment.chassis_command.capability));
    pack_key(packer, "traversal"); pack_traversal(packer, segment.traversal_constraint);
    pack_key(packer, "terrain_label"); packer.pack(segment.terrain_label);
    pack_key(packer, "requires_high_performance"); packer.pack(segment.requires_high_performance);
}

nav_executor::StepPlanSegment step_segment(const msgpack::object& object) {
    nav_executor::StepPlanSegment segment;
    segment.prepare_u = value<double>(object, "prepare_u");
    segment.active_u = value<double>(object, "active_u");
    segment.commit_u = value<double>(object, "commit_u");
    segment.step_enter_u = value<double>(object, "step_enter_u");
    segment.step_exit_u = value<double>(object, "step_exit_u");
    segment.release_u = value<double>(object, "release_u");
    segment.step_enter_pos_map = vector2(field(object, "step_enter_pos"));
    segment.step_exit_pos_map = vector2(field(object, "step_exit_pos"));
    segment.dir_map = vector2(field(object, "dir_map"));
    segment.direction = static_cast<nav_executor::StepDirection>(value<uint8_t>(object, "step_direction"));
    segment.chassis_command.mode = value<uint8_t>(object, "chassis_mode");
    segment.chassis_command.capability = static_cast<nav_executor::CapabilityLevel>(value<uint8_t>(object, "capability"));
    segment.traversal_constraint = traversal(field(object, "traversal"));
    segment.terrain_label = value<uint8_t>(object, "terrain_label");
    segment.requires_high_performance = value<bool>(object, "requires_high_performance");
    return segment;
}

void pack_map_snapshot(Packer& packer, const MapSnapshot& map) {
    packer.pack_map(7);
    pack_key(packer, "width"); packer.pack(map.width);
    pack_key(packer, "height"); packer.pack(map.height);
    pack_key(packer, "resolution"); packer.pack(map.resolution);
    pack_key(packer, "origin_x"); packer.pack(map.origin_x);
    pack_key(packer, "origin_y"); packer.pack(map.origin_y);
    pack_key(packer, "global_cost"); pack_binary(packer, map.global_cost_data);
    pack_key(packer, "direction_image"); pack_binary(packer, map.direction_image_data);
}

MapSnapshot map_snapshot(const msgpack::object& object) {
    MapSnapshot map;
    map.width = value<int>(object, "width");
    map.height = value<int>(object, "height");
    map.resolution = value<double>(object, "resolution");
    map.origin_x = value<double>(object, "origin_x");
    map.origin_y = value<double>(object, "origin_y");
    map.global_cost_data = binary(object, "global_cost");
    map.direction_image_data = binary(object, "direction_image");
    return map;
}

void pack_route(Packer& packer, const StoredRoute& route) {
    packer.pack_map(8);
    pack_key(packer, "name"); packer.pack(route.spec.name);
    pack_key(packer, "start");
    packer.pack_array(3);
    packer.pack(route.spec.start_pose.x());
    packer.pack(route.spec.start_pose.y());
    packer.pack(route.spec.start_pose.z());
    pack_key(packer, "goal"); pack_vector2(packer, route.spec.goal);
    pack_key(packer, "timeout"); packer.pack(route.spec.timeout);
    pack_key(packer, "seeds"); packer.pack(route.spec.seeds);
    pack_key(packer, "spline_control_points");
    packer.pack_array(static_cast<uint32_t>(route.spline_control_points.size()));
    for (const auto& point : route.spline_control_points) pack_vector2(packer, point);
    pack_key(packer, "step_segments");
    packer.pack_array(static_cast<uint32_t>(route.step_segments.size()));
    for (const auto& segment : route.step_segments) pack_step_segment(packer, segment);
    pack_key(packer, "control_cost"); pack_binary(packer, route.control_cost_data);
}

StoredRoute stored_route(const msgpack::object& object, const std::string& split) {
    StoredRoute route;
    route.spec.name = value<std::string>(object, "name");
    route.spec.split = split;
    const auto& start = field(object, "start");
    if (start.type != msgpack::type::ARRAY || start.via.array.size != 3) {
        throw std::runtime_error("Route start must contain x, y, yaw");
    }
    route.spec.start_pose = {
        start.via.array.ptr[0].as<double>(),
        start.via.array.ptr[1].as<double>(),
        start.via.array.ptr[2].as<double>(),
    };
    route.spec.goal = vector2(field(object, "goal"));
    route.spec.timeout = value<double>(object, "timeout");
    route.spec.seeds = value<std::vector<uint64_t>>(object, "seeds");

    const auto& points = field(object, "spline_control_points");
    if (points.type != msgpack::type::ARRAY) throw std::runtime_error("spline_control_points must be an array");
    route.spline_control_points.reserve(points.via.array.size);
    for (uint32_t i = 0; i < points.via.array.size; ++i) {
        route.spline_control_points.push_back(vector2(points.via.array.ptr[i]));
    }

    const auto& segments = field(object, "step_segments");
    if (segments.type != msgpack::type::ARRAY) throw std::runtime_error("step_segments must be an array");
    route.step_segments.reserve(segments.via.array.size);
    for (uint32_t i = 0; i < segments.via.array.size; ++i) {
        route.step_segments.push_back(step_segment(segments.via.array.ptr[i]));
    }
    route.control_cost_data = binary(object, "control_cost");
    return route;
}

void validate_bundle(const SceneBundle& bundle) {
    if (bundle.format_version != SceneBundle::FORMAT_VERSION) {
        throw std::runtime_error("Unsupported scene bundle version: " + std::to_string(bundle.format_version));
    }
    if (bundle.split != "train" && bundle.split != "validation") {
        throw std::runtime_error("Scene bundle split must be 'train' or 'validation'");
    }
    if (bundle.map.width <= 0 || bundle.map.height <= 0 || bundle.map.resolution <= 0.0) {
        throw std::runtime_error("Invalid scene bundle map geometry");
    }
    const size_t cells = static_cast<size_t>(bundle.map.width) * static_cast<size_t>(bundle.map.height);
    if (bundle.map.global_cost_data.size() != cells) {
        throw std::runtime_error("Scene bundle global cost map size mismatch");
    }
    if (bundle.map.direction_image_data.size() != cells * 3) {
        throw std::runtime_error("Scene bundle direction image size mismatch");
    }
    if (bundle.routes.empty()) throw std::runtime_error("Scene bundle contains no routes");
    for (const auto& route : bundle.routes) {
        if (route.spec.name.empty()) throw std::runtime_error("Scene bundle contains an unnamed route");
        if (route.spline_control_points.size() < 3) {
            throw std::runtime_error("Route '" + route.spec.name + "' has fewer than three spline control points");
        }
        if (route.control_cost_data.size() != cells) {
            throw std::runtime_error("Route '" + route.spec.name + "' control cost map size mismatch");
        }
        if (route.spec.seeds.empty()) throw std::runtime_error("Route '" + route.spec.name + "' has no seeds");
    }
}

} // namespace

void write_scene_bundle(const SceneBundle& bundle, const std::filesystem::path& path) {
    validate_bundle(bundle);
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Failed to open scene bundle for writing: " + path.string());
    Packer packer(stream);
    packer.pack_map(5);
    pack_key(packer, "format_version"); packer.pack(bundle.format_version);
    pack_key(packer, "split"); packer.pack(bundle.split);
    pack_key(packer, "created_at"); packer.pack(bundle.created_at);
    pack_key(packer, "map"); pack_map_snapshot(packer, bundle.map);
    pack_key(packer, "routes");
    packer.pack_array(static_cast<uint32_t>(bundle.routes.size()));
    for (const auto& route : bundle.routes) pack_route(packer, route);
    stream.close();
    if (!stream) throw std::runtime_error("Failed while writing scene bundle: " + path.string());
}

SceneBundle load_scene_bundle(const std::filesystem::path& path) {
    try {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("Failed to open file");
        const std::vector<char> bytes {
            std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()
        };
        if (bytes.empty()) throw std::runtime_error("File is empty");
        const auto handle = msgpack::unpack(bytes.data(), bytes.size());
        const auto& root = handle.get();

        SceneBundle bundle;
        bundle.format_version = value<uint32_t>(root, "format_version");
        bundle.split = value<std::string>(root, "split");
        bundle.created_at = value<std::string>(root, "created_at");
        bundle.map = map_snapshot(field(root, "map"));
        const auto& routes = field(root, "routes");
        if (routes.type != msgpack::type::ARRAY) throw std::runtime_error("routes must be an array");
        bundle.routes.reserve(routes.via.array.size);
        for (uint32_t i = 0; i < routes.via.array.size; ++i) {
            bundle.routes.push_back(stored_route(routes.via.array.ptr[i], bundle.split));
        }
        validate_bundle(bundle);
        return bundle;
    } catch (const std::exception& error) {
        throw std::runtime_error("Failed to load scene bundle '" + path.string() + "': " + error.what());
    }
}

std::vector<CompiledScenario> materialize_scene_bundle(const SceneBundle& bundle) {
    validate_bundle(bundle);
    auto global_cost = std::make_shared<const nav_executor::CostMap>(
        bundle.map.width, bundle.map.height, bundle.map.resolution,
        bundle.map.origin_x, bundle.map.origin_y, bundle.map.global_cost_data
    );
    cv::Mat direction_image(
        bundle.map.height, bundle.map.width, CV_8UC3,
        const_cast<uint8_t*>(bundle.map.direction_image_data.data())
    );
    auto direction = std::make_shared<const nav_executor::DirectionMap>(
        direction_image, bundle.map.resolution, bundle.map.origin_x, bundle.map.origin_y
    );

    std::vector<CompiledScenario> scenarios;
    scenarios.reserve(bundle.routes.size());
    for (const auto& stored : bundle.routes) {
        auto path = std::make_shared<nav_executor::AnnotatedPath>(
            nav_executor::SplinePath(stored.spline_control_points)
        );
        path->goal_pos = stored.spec.goal;
        path->step_segments = stored.step_segments;
        std::vector<nav_executor::StepTraversalConstraint> constraints;
        constraints.reserve(path->step_segments.size());
        for (const auto& segment : path->step_segments) {
            constraints.push_back(segment.traversal_constraint);
        }
        path->step_constraint_schedule = std::make_shared<const nav_executor::StepConstraintSchedule>(
            std::move(constraints)
        );
        auto control_cost = std::make_shared<const nav_executor::CostMap>(
            bundle.map.width, bundle.map.height, bundle.map.resolution,
            bundle.map.origin_x, bundle.map.origin_y, stored.control_cost_data
        );
        scenarios.push_back({
            .spec = stored.spec,
            .path = std::move(path),
            .global_cost_map = global_cost,
            .control_cost_map = std::move(control_cost),
            .direction_map = direction,
        });
    }
    return scenarios;
}

std::vector<CompiledScenario> load_scene_splits(const std::filesystem::path& scenes_directory) {
    const auto load_split = [](const std::filesystem::path& directory, const std::string_view split) {
        if (!std::filesystem::is_directory(directory)) {
            throw std::runtime_error(
                "Required " + std::string(split) + " scene directory does not exist: " + directory.string()
            );
        }

        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".msgpack") {
                paths.push_back(entry.path());
            }
        }
        std::ranges::sort(paths);
        if (paths.empty()) {
            throw std::runtime_error(
                "No .msgpack scene bundles found in required " + std::string(split)
                + " scene directory: " + directory.string()
            );
        }

        std::vector<CompiledScenario> scenarios;
        for (const auto& path : paths) {
            const SceneBundle bundle = load_scene_bundle(path);
            if (bundle.split != split) {
                throw std::runtime_error(
                    "Scene bundle '" + path.string() + "' has split '" + bundle.split
                    + "' but is stored in the " + std::string(split) + " directory"
                );
            }
            auto loaded = materialize_scene_bundle(bundle);
            scenarios.insert(
                scenarios.end(),
                std::make_move_iterator(loaded.begin()),
                std::make_move_iterator(loaded.end())
            );
        }
        return scenarios;
    };

    auto train = load_split(scenes_directory / "train", "train");
    auto validation = load_split(scenes_directory / "validation", "validation");
    train.insert(
        train.end(),
        std::make_move_iterator(validation.begin()),
        std::make_move_iterator(validation.end())
    );
    return train;
}

} // namespace mpc_tuner
