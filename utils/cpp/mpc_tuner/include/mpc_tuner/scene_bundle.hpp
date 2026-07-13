#pragma once

#include <filesystem>
#include <vector>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

void write_scene_bundle(const SceneBundle& bundle, const std::filesystem::path& path);
SceneBundle load_scene_bundle(const std::filesystem::path& path);

[[nodiscard]] std::vector<CompiledScenario> materialize_scene_bundle(const SceneBundle& bundle);
[[nodiscard]] std::vector<CompiledScenario> load_scene_directory(const std::filesystem::path& directory);

} // namespace mpc_tuner
