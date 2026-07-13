#pragma once

#include <deque>
#include <mutex>
#include <thread>

namespace mpc_tuner {

class ConsoleInput {
public:
    ConsoleInput();
    ~ConsoleInput();

    ConsoleInput(const ConsoleInput&) = delete;
    ConsoleInput& operator=(const ConsoleInput&) = delete;

    [[nodiscard]] bool available() const { return tty_fd_ >= 0; }
    [[nodiscard]] std::deque<char> take_commands();

private:
    void read_commands(std::stop_token stop_token);

    std::mutex mutex_;
    std::deque<char> commands_;
    int tty_fd_ = -1;
    std::jthread thread_;
};

} // namespace mpc_tuner
