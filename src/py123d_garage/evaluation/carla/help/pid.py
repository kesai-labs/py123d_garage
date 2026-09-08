from __future__ import annotations

import math
from collections import deque

import numpy as np
import numpy.typing as npt


class PIDController:
    """PID controller with a sliding error window for the integral and derivative terms."""

    def __init__(
        self,
        k_p: float,
        k_i: float,
        k_d: float,
        window_size: int = 20,
    ) -> None:
        """
        Initializes the controller with zeroed error history.

        Args:
            k_p: proportional gain.
            k_i: integral gain.
            k_d: derivative gain.
            window_size: length of the sliding error window.
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self._window: deque[float] = deque(
            [0.0] * window_size,
            maxlen=window_size,
        )

    def step(self, error: float) -> float:
        """
        Computes the control output for the current error.

        Args:
            error: current error value.

        Returns:
            the control output.
        """
        self._window.append(error)
        integral = sum(self._window) / len(self._window)
        derivative = self._window[-1] - self._window[-2]
        return self.k_p * error + self.k_i * integral + self.k_d * derivative


class WaypointFollower:
    """
    Follows predicted spatio-temporal waypoints with PID controllers.

    The waypoint spacing encodes the desired speed; steering aims at the first
    waypoint beyond a speed regime dependent aim distance.
    """

    def __init__(
        self,
        interval_length: float,
        turn_kp: float = 1.25,
        turn_ki: float = 0.75,
        turn_kd: float = 0.3,
        speed_kp: float = 1.75,
        speed_ki: float = 1.0,
        speed_kd: float = 2.0,
        speed_delta_clip: float = 0.99,
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        aim_distance_fast: float = 3.0,
        aim_distance_slow: float = 2.25,
        aim_distance_threshold: float = 5.5,
    ) -> None:
        """
        Initializes both PID controllers and the waypoint timing.

        Args:
            interval_length: time between consecutive waypoints in seconds.
            turn_kp: lateral proportional gain.
            turn_ki: lateral integral gain.
            turn_kd: lateral derivative gain.
            speed_kp: longitudinal proportional gain.
            speed_ki: longitudinal integral gain.
            speed_kd: longitudinal derivative gain.
            speed_delta_clip: maximum speed error fed to the longitudinal controller.
            brake_speed: desired speed below which the ego brakes.
            brake_ratio: ratio of speed to desired speed above which the ego brakes.
            aim_distance_fast: steering aim distance outside intersections.
            aim_distance_slow: steering aim distance inside intersections.
            aim_distance_threshold: desired speed separating the two aim distances.
        """
        index_half_second = round(0.5 / interval_length) - 1
        index_full_second = round(1.0 / interval_length) - 1
        assert 0 <= index_half_second < index_full_second, (
            f"Waypoint interval {interval_length} s cannot resolve the 0.5 s and 1.0 s poses "
            f"the desired-speed estimate reads."
        )
        self._index_half_second = index_half_second
        self._index_full_second = index_full_second

        self.lateral_controller = PIDController(turn_kp, turn_ki, turn_kd)
        self.longitudinal_controller = PIDController(
            speed_kp,
            speed_ki,
            speed_kd,
        )
        self.speed_delta_clip = speed_delta_clip
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.aim_distance_fast = aim_distance_fast
        self.aim_distance_slow = aim_distance_slow
        self.aim_distance_threshold = aim_distance_threshold

    def step(
        self,
        waypoints: npt.NDArray[np.float64],
        speed: float,
    ) -> tuple[float, float, float]:
        """
        Computes vehicle controls from the predicted waypoints.

        Args:
            waypoints: predicted future waypoints in the ego frame, shape (n, 2).
            speed: current ego speed in m/s.

        Returns:
            (steer, throttle, brake) in their CARLA ranges.
        """
        desired_speed = float(
            np.linalg.norm(
                waypoints[self._index_half_second] - waypoints[self._index_full_second],
            )
            * 2.0,
        )
        brake = desired_speed < self.brake_speed or speed / desired_speed > self.brake_ratio

        delta_speed = float(
            np.clip(desired_speed - speed, 0.0, self.speed_delta_clip),
        )
        throttle = 0.0 if brake else self.longitudinal_controller.step(delta_speed)

        aim_distance = self.aim_distance_slow if desired_speed < self.aim_distance_threshold else self.aim_distance_fast
        beyond_aim_distance = np.linalg.norm(waypoints, axis=1) >= aim_distance
        aim = waypoints[np.argmax(beyond_aim_distance) if beyond_aim_distance.any() else -1]

        # Waypoints are ISO 8855 (y left); CARLA steers right for positive values.
        angle = -math.degrees(math.atan2(aim[1], aim[0])) / 90.0
        if brake or speed < 0.01:
            angle = 0.0  # Keeps the standstill angle error out of the integral window.
        steer = float(np.clip(self.lateral_controller.step(angle), -1.0, 1.0))
        return steer, float(throttle), float(brake)
