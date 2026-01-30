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


class _NonBlockingKeyReader:
    def __init__(self) -> None:
        self._enabled = False
        self._fd: int | None = None
        self._old_attrs = None

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

    def read_available(self) -> list[str]:
        if not self._enabled:
            return []

        chars: list[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            ch = sys.stdin.read(1)
            if ch == '':
                break
            chars.append(ch)
        return chars


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
        # Manual mode uses a latch window to avoid key-repeat jitter.
        # Pressing/holding a key refreshes its latch.
        self.declare_parameter('manual_latch_s', 0.6)

        # Triggered half-sine durations
        self.declare_parameter('velocity_duration_s', 2.0)
        self.declare_parameter('palstance_pos_duration_s', 1.5)
        self.declare_parameter('palstance_neg_duration_s', 1.5)

        # Manual (W/A/D) targets
        self.declare_parameter('manual_velocity', 0.5)  # m/s
        self.declare_parameter('manual_palstance', 1.0)  # rad/s

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

        self._mode: str = 'trigger'  # 'trigger' or 'manual'
        self._active_action: _SineAction | None = None

        self._manual_v_until_s: float | None = None
        self._manual_w_until_s: float | None = None
        self._manual_w_sign: int = 0  # +1 / -1 / 0

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

    def _print_help(self) -> None:
        self.get_logger().info('Keyboard controls:')
        self.get_logger().info('  m: toggle mode (trigger/manual)')
        self.get_logger().info("  h: show this help")
        self.get_logger().info('Trigger mode:')
        self.get_logger().info('  1: velocity half-sine (0->+v->0)')
        self.get_logger().info('  2: +palstance half-sine (0->+w->0)')
        self.get_logger().info('  3: -palstance half-sine (0->-w->0)')
        self.get_logger().info('  space: cancel current action')
        self.get_logger().info('Manual mode:')
        self.get_logger().info('  W: latch +velocity (manual_velocity) for manual_latch_s')
        self.get_logger().info('  A: latch +palstance (manual_palstance) for manual_latch_s')
        self.get_logger().info('  D: latch -palstance (manual_palstance) for manual_latch_s')
        self.get_logger().info('  space: stop manual commands')
        self.get_logger().info('No key => command ramps back to 0 (acc/alpha-limited).')

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
            self._mode = 'manual' if self._mode == 'trigger' else 'trigger'
            self._active_action = None
            self._manual_v_until_s = None
            self._manual_w_until_s = None
            self._manual_w_sign = 0
            self.get_logger().info(f'Mode switched to: {self._mode}')
            return

        if ch == ' ':
            self._active_action = None
            self._manual_v_until_s = None
            self._manual_w_until_s = None
            self._manual_w_sign = 0
            return

        if self._mode == 'trigger':
            if ch == '1':
                self._active_action = _SineAction(kind='v', start_time_s=now_s)
            elif ch == '2':
                self._active_action = _SineAction(kind='w_pos', start_time_s=now_s)
            elif ch == '3':
                self._active_action = _SineAction(kind='w_neg', start_time_s=now_s)
            return

        # Manual mode: latch commands for a short window to avoid key-repeat jitter.
        latch_s = max(0.0, float(self.get_parameter('manual_latch_s').value))
        until = now_s + latch_s
        if ch in ('w', 'W'):
            self._manual_v_until_s = until if self._manual_v_until_s is None else max(self._manual_v_until_s, until)
        elif ch in ('a', 'A'):
            self._manual_w_sign = +1
            self._manual_w_until_s = until if self._manual_w_until_s is None else max(self._manual_w_until_s, until)
        elif ch in ('d', 'D'):
            self._manual_w_sign = -1
            self._manual_w_until_s = until if self._manual_w_until_s is None else max(self._manual_w_until_s, until)

    def _desired_from_action(self, now_s: float) -> tuple[float, float]:
        if self._active_action is None:
            return 0.0, 0.0

        max_v = abs(float(self.get_parameter('max_velocity').value))
        max_w = abs(float(self.get_parameter('max_palstance').value))

        elapsed = max(0.0, now_s - self._active_action.start_time_s)
        if self._active_action.kind == 'v':
            duration = float(self.get_parameter('velocity_duration_s').value)
            if elapsed >= duration:
                self._active_action = None
                return 0.0, 0.0
            return self._half_sine(elapsed, duration, max_v), 0.0

        if self._active_action.kind == 'w_pos':
            duration = float(self.get_parameter('palstance_pos_duration_s').value)
            if elapsed >= duration:
                self._active_action = None
                return 0.0, 0.0
            return 0.0, self._half_sine(elapsed, duration, max_w)

        if self._active_action.kind == 'w_neg':
            duration = float(self.get_parameter('palstance_neg_duration_s').value)
            if elapsed >= duration:
                self._active_action = None
                return 0.0, 0.0
            return 0.0, -self._half_sine(elapsed, duration, max_w)

        # Unknown kind
        self._active_action = None
        return 0.0, 0.0

    def _desired_from_manual(self, now_s: float) -> tuple[float, float]:
        manual_v = float(self.get_parameter('manual_velocity').value)
        manual_w = float(self.get_parameter('manual_palstance').value)

        v_active = self._manual_v_until_s is not None and now_s <= self._manual_v_until_s
        w_active = self._manual_w_until_s is not None and now_s <= self._manual_w_until_s

        v_des = manual_v if v_active else 0.0
        w_des = (abs(manual_w) * float(self._manual_w_sign)) if (w_active and self._manual_w_sign != 0) else 0.0
        return v_des, w_des

    def _publish_cmd(self, velocity: float, palstance: float) -> None:
        msg = ChassisCmd()
        msg.velocity = float(velocity)
        msg.palstance = float(palstance)
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

        if self._mode == 'trigger':
            v_des, w_des = self._desired_from_action(now_s)
        else:
            v_des, w_des = self._desired_from_manual(now_s)

        max_v = abs(float(self.get_parameter('max_velocity').value))
        max_w = abs(float(self.get_parameter('max_palstance').value))
        a_max = abs(float(self.get_parameter('max_accel').value))
        alpha_max = abs(float(self.get_parameter('max_palstance_accel').value))

        v_des = _clamp(v_des, -max_v, max_v)
        w_des = _clamp(w_des, -max_w, max_w)

        self._v_out = self._slew_limit(self._v_out, v_des, a_max, dt)
        self._w_out = self._slew_limit(self._w_out, w_des, alpha_max, dt)

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