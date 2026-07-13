#include <mpc_tuner/console_input.hpp>

#include <cctype>

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

namespace mpc_tuner {

ConsoleInput::ConsoleInput(): tty_fd_(::open("/dev/tty", O_RDONLY | O_NONBLOCK)) {
    if (tty_fd_ >= 0) {
        thread_ = std::jthread([this](const std::stop_token stop_token) {
            read_commands(stop_token);
        });
    }
}

ConsoleInput::~ConsoleInput() {
    thread_.request_stop();
    if (thread_.joinable()) thread_.join();
    if (tty_fd_ >= 0) ::close(tty_fd_);
}

std::deque<char> ConsoleInput::take_commands() {
    std::scoped_lock lock(mutex_);
    std::deque<char> result;
    result.swap(commands_);
    return result;
}

void ConsoleInput::read_commands(const std::stop_token stop_token) {
    pollfd descriptor {.fd = tty_fd_, .events = POLLIN, .revents = 0};
    while (!stop_token.stop_requested()) {
        descriptor.revents = 0;
        if (::poll(&descriptor, 1, 100) <= 0) continue;

        char buffer[64];
        const ssize_t count = ::read(tty_fd_, buffer, sizeof(buffer));
        if (count <= 0) continue;

        std::scoped_lock lock(mutex_);
        for (ssize_t i = 0; i < count; ++i) {
            const char command = static_cast<char>(
                std::tolower(static_cast<unsigned char>(buffer[i]))
            );
            if (!std::isspace(static_cast<unsigned char>(command))) {
                commands_.push_back(command);
            }
        }
    }
}

} // namespace mpc_tuner
