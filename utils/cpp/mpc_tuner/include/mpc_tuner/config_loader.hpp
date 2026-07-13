#pragma once

#include <filesystem>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

TunerConfig load_tuner_config(const std::filesystem::path& path);
RuntimeConfig load_runtime_config(const std::filesystem::path& nav_config_directory);

} // namespace mpc_tuner
