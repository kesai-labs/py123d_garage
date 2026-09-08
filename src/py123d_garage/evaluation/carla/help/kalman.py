from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from py123d.geometry.utils.rotation_utils import normalize_angle

_STATE_DIM = 4
_ALPHA = 1e-5
_BETA = 2.0
_KAPPA = 0.0

_INITIAL_COVARIANCE = np.diag([0.5, 0.5, 1e-6, 1e-6])
_MEASUREMENT_COVARIANCE = np.diag([0.5, 0.5, 1e-15, 1e-15])
_PROCESS_COVARIANCE = np.diag([1e-4, 1e-4, 1e-3, 1e-3])

_FRONT_WHEELBASE_M = -0.090769015
_REAR_WHEELBASE_M = 1.4178275
_STEER_GAIN = 0.36848336
_BRAKE_ACCELERATION_MPS2 = -4.952399
_THROTTLE_ACCELERATION_MPS2 = 0.5633837


def _bicycle_model_forward(
    state: npt.NDArray[np.float64],
    dt: float,
    steer: float,
    throttle: float,
    brake: float,
) -> npt.NDArray[np.float64]:
    """
    Propagates one state through leaderboard 1.0's kinematic bicycle model.

    The wheelbase, gain and acceleration constants are World on Rails' fit of
    CARLA's default ego vehicle.

    Args:
        state: (x, y, yaw, speed) in the CARLA world frame.
        dt: timestep in seconds.
        steer: the steering command that was applied.
        throttle: the throttle command that was applied.
        brake: the brake command that was applied.

    Returns:
        the propagated state.
    """
    x, y, yaw, speed = state
    acceleration = _BRAKE_ACCELERATION_MPS2 if brake else _THROTTLE_ACCELERATION_MPS2 * throttle
    slip = math.atan(
        _REAR_WHEELBASE_M / (_FRONT_WHEELBASE_M + _REAR_WHEELBASE_M) * math.tan(_STEER_GAIN * steer),
    )
    return np.array(
        [
            x + speed * math.cos(yaw + slip) * dt,
            y + speed * math.sin(yaw + slip) * dt,
            yaw + speed / _REAR_WHEELBASE_M * math.sin(slip) * dt,
            max(speed + acceleration * dt, 0.0),
        ],
    )


def _residual(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Subtracts two states, taking the shorter way around on the yaw component.

    Args:
        left: the state subtracted from.
        right: the state subtracted.

    Returns:
        the difference.
    """
    difference = left - right
    difference[..., 2] = normalize_angle(difference[..., 2])
    return difference


def _weighted_mean(
    sigmas: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Averages sigma points, averaging their yaws as unit vectors.

    Args:
        sigmas: the sigma points, one per row.
        weights: the weight of each sigma point.

    Returns:
        the mean state.
    """
    mean = sigmas.T @ weights
    mean[2] = math.atan2(np.sin(sigmas[:, 2]) @ weights, np.cos(sigmas[:, 2]) @ weights)
    return mean


class GnssKalmanFilter:
    """
    Fuses the noisy GNSS fix with the vehicle model driven by the applied controls.

    The leaderboard's GNSS carries about half a metre of white noise per axis, twice
    the width of a BEV pixel, which jitters the route progress and smears the lidar
    sweeps a policy accumulates across ticks. The state is (x, y, yaw, speed) in the
    CARLA world frame and the measurement is that whole state, so only the vehicle
    model needs the unscented treatment; the correction is the linear one.
    """

    def __init__(self, dt: float) -> None:
        """
        Prepares the sigma-point weights; the first measurement sets the state.

        Args:
            dt: seconds between simulator ticks.
        """
        self._dt = dt
        scaling = _ALPHA**2 * (_STATE_DIM + _KAPPA) - _STATE_DIM
        self._spread = _STATE_DIM + scaling
        self._weights_mean = np.full(2 * _STATE_DIM + 1, 0.5 / self._spread)
        self._weights_covariance = self._weights_mean.copy()
        self._weights_mean[0] = scaling / self._spread
        self._weights_covariance[0] = scaling / self._spread + 1.0 - _ALPHA**2 + _BETA
        self._covariance = _INITIAL_COVARIANCE.copy()
        self._state = np.zeros(_STATE_DIM)
        self._origin = np.zeros(2)
        self._initialized = False

    def step(
        self,
        position: npt.NDArray[np.float64],
        yaw: float,
        speed: float,
        steer: float,
        throttle: float,
        brake: float,
    ) -> npt.NDArray[np.float64]:
        """
        Predicts through the controls that were applied, then corrects with this tick's fix.

        Args:
            position: the noisy (x, y) GNSS fix in the CARLA world frame.
            yaw: the compass heading in radians in the CARLA world frame.
            speed: the speedometer reading in m/s.
            steer: the steering command applied since the last call.
            throttle: the throttle command applied since the last call.
            brake: the brake command applied since the last call.

        Returns:
            the filtered (x, y) position in the CARLA world frame.
        """
        if not self._initialized:
            self._origin = position.copy()
            self._state = np.array([0.0, 0.0, normalize_angle(yaw), speed])
            self._initialized = True

        model_covariance = self._predict(steer, throttle, brake)
        self._update(
            np.array(
                [
                    position[0] - self._origin[0],
                    position[1] - self._origin[1],
                    normalize_angle(yaw),
                    speed,
                ],
            ),
            model_covariance,
        )
        return self._state[:2] + self._origin

    def _predict(self, steer: float, throttle: float, brake: float) -> npt.NDArray[np.float64]:
        """
        Propagates the state through the vehicle model.

        Args:
            steer: the steering command applied since the last call.
            throttle: the throttle command applied since the last call.
            brake: the brake command applied since the last call.

        Returns:
            the propagated covariance before the process noise is added to it.
        """
        offsets = np.linalg.cholesky(self._spread * self._covariance).T.astype(np.float64)
        mean = self._state[None, :]
        sigmas = np.concatenate((mean, _residual(mean, -offsets), _residual(mean, offsets)))
        propagated = np.array(
            [_bicycle_model_forward(sigma, self._dt, steer, throttle, brake) for sigma in sigmas],
        )
        self._state = _weighted_mean(propagated, self._weights_mean)
        deviation = _residual(propagated, self._state)
        model_covariance = np.einsum("k,ki,kj->ij", self._weights_covariance, deviation, deviation)
        self._covariance = model_covariance + _PROCESS_COVARIANCE
        return model_covariance

    def _update(
        self,
        measurement: npt.NDArray[np.float64],
        model_covariance: npt.NDArray[np.float64],
    ) -> None:
        """
        Corrects the predicted state towards the measurement.

        Args:
            measurement: the observed (x, y, yaw, speed).
            model_covariance: the propagated covariance the sigma points spread over.
        """
        innovation_covariance = model_covariance + _MEASUREMENT_COVARIANCE
        gain = model_covariance @ np.linalg.inv(innovation_covariance)
        self._state = self._state + gain @ _residual(measurement, self._state)
        self._covariance = self._covariance - gain @ innovation_covariance @ gain.T
