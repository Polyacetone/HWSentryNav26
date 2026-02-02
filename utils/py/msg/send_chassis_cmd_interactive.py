import math
import select
import sys
import termios
import tty
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from interfaces.msg import ChassisCmd

def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class _SineAction:
    kind: str  # 'v', 'w_pos', 'w_neg'
    start_time_s: float


@dataclass
class _AutoAction:
    kind: str  # 'step_static', 'step_moving', 'sweep'
    start_time_s: float


class _Key:
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'


class _NonBlockingKeyReader:
    def __init__(self) -> None:
        self._enabled = False
        self._fd: int | None = None
        self._old_attrs = None
        self._buffer = ''

    @property
    def enabled(self) -> bool:
        return self._enabled

    def open(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            self._enabled = False
            return

        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # immediate keypresses, no Enter needed
        self._enabled = True

    def close(self) -> None:
        if not self._enabled:
            return
        try:
            if self._fd is not None and self._old_attrs is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        finally:
            self._enabled = False
            self._fd = None
            self._old_attrs = None
            self._buffer = ''

    def read_available(self) -> list[str]:
        if not self._enabled:
            return []

        # Read all currently available characters.
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            ch = sys.stdin.read(1)
            if ch == '':
                break
            self._buffer += ch

        # Parse buffer into key events (including arrow keys).
        events: list[str] = []
        i = 0
        while i < len(self._buffer):
            ch = self._buffer[i]
            if ch != '\x1b':
                events.append(ch)
                i += 1
                continue

            # Escape sequence. Expect "\x1b[<code>" for arrows.
            if i + 1 >= len(self._buffer):
                break
            if self._buffer[i + 1] != '[':
                # Unknown escape; emit ESC and continue.
                events.append(ch)
                i += 1
                continue
            if i + 2 >= len(self._buffer):
                break

            code = self._buffer[i + 2]
            if code == 'A':
                events.append(_Key.UP)
                i += 3
            elif code == 'B':
                events.append(_Key.DOWN)
                i += 3
            elif code == 'C':
                events.append(_Key.RIGHT)
                i += 3
            elif code == 'D':
                events.append(_Key.LEFT)
                i += 3
            else:
                # Unrecognized CSI; drop ESC and continue.
                events.append(ch)
                i += 1

        # Keep any unconsumed tail (partial escape sequence).
        self._buffer = self._buffer[i:]
        return events


class ChassisCmdPublisher(Node):
    def __init__(self):
        super().__init__('chassis_cmd_sine_profile_publisher')

        self.declare_parameter('topic', '/path_follower/chassis_cmd')
        self.declare_parameter('rate_hz', 10.0)

        self.declare_parameter('max_velocity', 1.0)  # m/s
        self.declare_parameter('max_palstance', 6.0)  # rad/s
        self.declare_parameter('max_accel', 2.5)  # m/s^2
        self.declare_parameter('max_palstance_accel', 12.0)  # rad/s^2

        # Keyboard behavior
        self.declare_parameter('keyboard_enable', True)

        # Manual (arrow keys) increments
        self.declare_parameter('manual_velocity_step', 0.5)  # m/s per keypress
        self.declare_parameter('manual_palstance_step', 2.0)  # rad/s per keypress

        # Auto: step tests
        # Static step: hold baseline for pre_s, then apply step for hold_s, then return baseline for post_s.
        self.declare_parameter('step_static_pre_s', 1.0)
        self.declare_parameter('step_static_hold_s', 2.0)
        self.declare_parameter('step_static_post_s', 1.0)
        self.declare_parameter('step_static_velocity', 1.0)  # m/s (>=0)
        self.declare_parameter('step_static_palstance', 0.0)  # rad/s

        # Moving step: bias -> bias+delta -> bias
        self.declare_parameter('step_moving_pre_s', 1.0)
        self.declare_parameter('step_moving_hold_s', 2.0)
        self.declare_parameter('step_moving_post_s', 1.0)
        self.declare_parameter('step_moving_bias_velocity', 0.5)  # m/s (>=0)
        self.declare_parameter('step_moving_bias_palstance', 0.0)  # rad/s
        self.declare_parameter('step_moving_delta_velocity', 0.5)  # m/s
        self.declare_parameter('step_moving_delta_palstance', 0.0)  # rad/s

        # Auto: sweep (chirp) test under a fixed bias velocity
        self.declare_parameter('sweep_duration_s', 5.0)
        self.declare_parameter('sweep_bias_velocity', 0.5)  # m/s (>=0)
        self.declare_parameter('sweep_bias_palstance', 0.0)  # rad/s
        self.declare_parameter('sweep_amplitude_palstance', 2.0)  # rad/s
        self.declare_parameter('sweep_f_start_hz', 0.1)
        self.declare_parameter('sweep_f_end_hz', 1.0)

        self.declare_parameter('publish_zero_on_shutdown', True)

        topic = str(self.get_parameter('topic').value)
        rate_hz = float(self.get_parameter('rate_hz').value)
        if rate_hz <= 0.0:
            raise ValueError('rate_hz must be > 0')

        self._dt = 1.0 / rate_hz
        self._publisher = self.create_publisher(ChassisCmd, topic, 10)

        self._last_time_s: float | None = None
        self._v_out = 0.0
        self._w_out = 0.0
        self._theta_out = 0.0

        self._mode: str = 'auto'  # 'auto' or 'manual'
        self._active_action: _AutoAction | None = None

        # Manual setpoints (arrow keys adjust these; output is rate-limited).
        self._manual_v_set = 0.0
        self._manual_w_set = 0.0

        self._keyboard = _NonBlockingKeyReader()
        if bool(self.get_parameter('keyboard_enable').value):
            self._keyboard.open()
            if not self._keyboard.enabled:
                self.get_logger().warning('keyboard_enable=true but stdin is not a TTY; keyboard control disabled.')

        self.get_logger().info(f'Publishing ChassisCmd on {topic} at {rate_hz:.1f} Hz')
        self._print_help()

        self._timer = self.create_timer(self._dt, self._timer_callback)

    def destroy_node(self):
        try:
            self._keyboard.close()
        finally:
            return super().destroy_node()

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _half_sine(self, t: float, duration: float, amplitude: float) -> float:
        """从 0 -> amplitude -> 0 的半正弦（t∈[0,duration]）。"""
        if duration <= 0.0:
            return 0.0
        x = _clamp(t / duration, 0.0, 1.0)
        return float(amplitude) * math.sin(math.pi * x)

    def _linear_chirp(self, t: float, duration: float, f0_hz: float, f1_hz: float, amplitude: float) -> float:
        """线性扫频正弦 (chirp): sin(2π(f0 t + 0.5 k t^2))."""
        if duration <= 0.0:
            return 0.0
        tt = _clamp(t, 0.0, duration)
        k = (f1_hz - f0_hz) / duration
        phase = 2.0 * math.pi * (f0_hz * tt + 0.5 * k * tt * tt)
        return float(amplitude) * math.sin(phase)

    def _print_help(self) -> None:
        self.get_logger().info('Keyboard controls:')
        self.get_logger().info('  m: toggle mode (auto/manual)')
        self.get_logger().info("  h: show this help")
        self.get_logger().info('Auto mode:')
        self.get_logger().info('  1: static step test (baseline->step->baseline)')
        self.get_logger().info('  2: moving step test (bias->bias+delta->bias)')
        self.get_logger().info('  3: sweep(chirp) test (fixed bias velocity + omega chirp)')
        self.get_logger().info('  space: cancel auto action')
        self.get_logger().info('Manual mode:')
        self.get_logger().info('  ↑/↓: velocity +=/-= manual_velocity_step (velocity >= 0)')
        self.get_logger().info('  ←/→: omega +=/-= manual_palstance_step')
        self.get_logger().info('  r: reset manual setpoints to 0')
        self.get_logger().info('  space: stop manual commands (setpoints=0)')
        self.get_logger().info('All outputs are accel/alpha-limited and speed-limited.')

    def _slew_limit(self, current: float, target: float, max_rate: float, dt: float) -> float:
        if dt <= 0.0 or max_rate <= 0.0:
            return target
        delta = target - current
        max_delta = max_rate * dt
        delta_limited = _clamp(delta, -max_delta, max_delta)
        return current + delta_limited

    def _handle_key(self, ch: str, now_s: float) -> None:
        if ch in ('h', 'H'):
            self._print_help()
            return

        if ch in ('m', 'M'):
            self._mode = 'manual' if self._mode == 'auto' else 'auto'
            self._active_action = None
            self._manual_v_set = 0.0
            self._manual_w_set = 0.0
            self.get_logger().info(f'Mode switched to: {self._mode}')
            return

        if ch == ' ':
            self._active_action = None
            self._manual_v_set = 0.0
            self._manual_w_set = 0.0
            return

        if self._mode == 'auto':
            if ch == '1':
                self._active_action = _AutoAction(kind='step_static', start_time_s=now_s)
            elif ch == '2':
                self._active_action = _AutoAction(kind='step_moving', start_time_s=now_s)
            elif ch == '3':
                self._active_action = _AutoAction(kind='sweep', start_time_s=now_s)
            return

        # Manual mode: arrow keys change setpoints discretely.
        max_v = abs(float(self.get_parameter('max_velocity').value))
        max_w = abs(float(self.get_parameter('max_palstance').value))
        v_step = abs(float(self.get_parameter('manual_velocity_step').value))
        w_step = abs(float(self.get_parameter('manual_palstance_step').value))

        if ch == _Key.UP:
            self._manual_v_set = _clamp(self._manual_v_set + v_step, 0.0, max_v)
        elif ch == _Key.DOWN:
            self._manual_v_set = _clamp(self._manual_v_set - v_step, 0.0, max_v)
        elif ch == _Key.LEFT:
            self._manual_w_set = _clamp(self._manual_w_set + w_step, -max_w, max_w)
        elif ch == _Key.RIGHT:
            self._manual_w_set = _clamp(self._manual_w_set - w_step, -max_w, max_w)
        elif ch in ('r', 'R'):
            self._manual_v_set = 0.0
            self._manual_w_set = 0.0

    def _desired_from_action(self, now_s: float) -> tuple[float, float]:
        if self._active_action is None:
            return 0.0, 0.0

        elapsed = max(0.0, now_s - self._active_action.start_time_s)

        if self._active_action.kind == 'step_static':
            pre_s = max(0.0, float(self.get_parameter('step_static_pre_s').value))
            hold_s = max(0.0, float(self.get_parameter('step_static_hold_s').value))
            post_s = max(0.0, float(self.get_parameter('step_static_post_s').value))
            total = pre_s + hold_s + post_s
            if elapsed >= total:
                self._active_action = None
                return 0.0, 0.0

            v_step = max(0.0, float(self.get_parameter('step_static_velocity').value))
            w_step = float(self.get_parameter('step_static_palstance').value)
            if elapsed < pre_s:
                return 0.0, 0.0
            if elapsed < pre_s + hold_s:
                return v_step, w_step
            return 0.0, 0.0

        if self._active_action.kind == 'step_moving':
            pre_s = max(0.0, float(self.get_parameter('step_moving_pre_s').value))
            hold_s = max(0.0, float(self.get_parameter('step_moving_hold_s').value))
            post_s = max(0.0, float(self.get_parameter('step_moving_post_s').value))
            total = pre_s + hold_s + post_s
            if elapsed >= total:
                self._active_action = None
                return 0.0, 0.0

            v_bias = max(0.0, float(self.get_parameter('step_moving_bias_velocity').value))
            w_bias = float(self.get_parameter('step_moving_bias_palstance').value)
            dv = float(self.get_parameter('step_moving_delta_velocity').value)
            dw = float(self.get_parameter('step_moving_delta_palstance').value)

            if elapsed < pre_s:
                return v_bias, w_bias
            if elapsed < pre_s + hold_s:
                return max(0.0, v_bias + dv), (w_bias + dw)
            return v_bias, w_bias

        if self._active_action.kind == 'sweep':
            duration = max(0.0, float(self.get_parameter('sweep_duration_s').value))
            if elapsed >= duration:
                self._active_action = None
                return 0.0, 0.0

            v_bias = max(0.0, float(self.get_parameter('sweep_bias_velocity').value))
            w_bias = float(self.get_parameter('sweep_bias_palstance').value)
            amp_w = abs(float(self.get_parameter('sweep_amplitude_palstance').value))
            f0 = max(0.0, float(self.get_parameter('sweep_f_start_hz').value))
            f1 = max(0.0, float(self.get_parameter('sweep_f_end_hz').value))

            w = w_bias + self._linear_chirp(elapsed, duration, f0, f1, amp_w)
            return v_bias, w

        # Unknown kind
        self._active_action = None
        return 0.0, 0.0

    def _desired_from_manual(self) -> tuple[float, float]:
        return float(self._manual_v_set), float(self._manual_w_set)

    def _publish_cmd(self, velocity: float, omega: float) -> None:
        msg = ChassisCmd()
        msg.velocity = float(velocity)
        msg.omega = float(omega)
        msg.theta = float(self._theta_out)
        msg.step_up_ahead = False
        msg.step_down_ahead = False
        msg.slow_spin = False
        msg.fast_spin = False
        self._publisher.publish(msg)

    def _timer_callback(self):
        now_s = self._now_s()
        if self._last_time_s is None:
            self._last_time_s = now_s

        if self._keyboard.enabled:
            for ch in self._keyboard.read_available():
                self._handle_key(ch, now_s)

        # Use actual dt to make rate-limit stable even if timer jitters
        dt = max(0.0, now_s - (self._last_time_s or now_s))
        self._last_time_s = now_s

        if self._mode == 'auto':
            v_des, w_des = self._desired_from_action(now_s)
        else:
            v_des, w_des = self._desired_from_manual()

        max_v = abs(float(self.get_parameter('max_velocity').value))
        max_w = abs(float(self.get_parameter('max_palstance').value))
        a_max = abs(float(self.get_parameter('max_accel').value))
        alpha_max = abs(float(self.get_parameter('max_palstance_accel').value))

        # Velocity is constrained to be non-negative.
        v_des = _clamp(v_des, 0.0, max_v)
        w_des = _clamp(w_des, -max_w, max_w)

        self._v_out = self._slew_limit(self._v_out, v_des, a_max, dt)
        self._w_out = self._slew_limit(self._w_out, w_des, alpha_max, dt)

        # Open-loop angle: integrate omega to get theta (initially 0).
        self._theta_out += self._w_out * dt

        # Safety clamp after slew
        self._v_out = _clamp(self._v_out, -max_v, max_v)
        self._w_out = _clamp(self._w_out, -max_w, max_w)

        self._publish_cmd(self._v_out, self._w_out)

def main(args=None):
    rclpy.init(args=args)
    node = ChassisCmdPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if bool(node.get_parameter('publish_zero_on_shutdown').value):
            try:
                node._theta_out = 0.0
                node._publish_cmd(0.0, 0.0)
            except Exception:
                pass
        try:
            node._keyboard.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()