from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch
from py123d.api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from torch import Tensor
from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.api.abstract_offline_data_source_config import AbstractOfflineDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)
from py123d_garage.api.contract_verifications import (
    verify_policy_against_benchmark,
    verify_policy_against_offline_data_source,
    verify_policy_against_scene_filter,
)
from py123d_garage.cache.codec import TensorCodec
from py123d_garage.datatypes.trajectory import TrajectorySE2

if TYPE_CHECKING:
    import lightning as L


@dataclass(frozen=True)
class CacheTensorSpec:
    """One cached tensor's declaration: its storage codec and the config values its builder reads."""

    codec: TensorCodec
    # Only fields that decide the tensor's shape or content, stringified.
    signature: dict[str, str]


FeaturesT = TypeVar("FeaturesT", bound=AbstractFeatures)
LabelsT = TypeVar("LabelsT", bound=AbstractLabels)
PredictionsT = TypeVar("PredictionsT", bound=AbstractPredictions)


class AbstractPolicy(
    torch.nn.Module,
    abc.ABC,
    Generic[FeaturesT, LabelsT, PredictionsT],
):
    """Interface a learned driving policy implements for training and evaluation."""

    def initialize(
        self,
        evaluation_checkpoint_file: str | None = None,
    ) -> None:
        """
        Builds the model, then strictly loads the evaluation checkpoint when given.

        Called once before training or inference; evaluation passes the
        checkpoint.
        """
        self.build_model(from_local_checkpoint=evaluation_checkpoint_file is not None)
        if evaluation_checkpoint_file is None:
            return
        state_dict: dict[str, Any] = torch.load(
            evaluation_checkpoint_file,
            map_location=None if torch.cuda.is_available() else torch.device("cpu"),
            weights_only=True,
        )
        # Accept both the raw policy state dict (training checkpoints) and a
        # Lightning checkpoint's "state_dict" entry with its wrapper prefixes.
        if "state_dict" in state_dict:
            state_dict = {k.removeprefix("policy."): v for k, v in state_dict["state_dict"].items()}
        self.load_state_dict(state_dict, strict=True)

    @property
    @abc.abstractmethod
    def policy_config(self) -> AbstractPolicyConfig:
        """
        The policy's config, fulfilling the AbstractPolicyConfig contract.

        Returns:
            the config object policy-agnostic code reads.
        """

    @abc.abstractmethod
    def build_model(self, from_local_checkpoint: bool) -> None:
        """
        Builds the model's modules; initialize loads any weights afterwards.

        Args:
            from_local_checkpoint: whether a local checkpoint overwrites the weights
                right after; skips fetching pretrained weights that would be discarded.
        """

    @abc.abstractmethod
    @override
    def forward(
        self,
        features: FeaturesT,
        navigation: NavigationConditioning,
    ) -> PredictionsT:
        """
        Computes predictions for one collated batch.

        Args:
            features: batched feature bundle from build_features.
            navigation: batched navigation conditioning.

        Returns:
            the prediction bundle.
        """

    @abc.abstractmethod
    def build_features(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> FeaturesT:
        """
        The model's input tensors of one scene, deterministic in the scene.

        Args:
            scene_api: scene interface anchored at the current frame.
            cached: tensors already read from the cache store — empty, or complete
                per cache_codecs. Take cached entries as-is, build the rest.

        Returns:
            the single-sample feature bundle, the inference inputs.
        """

    @abc.abstractmethod
    def build_labels(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> LabelsT:
        """
        The ground-truth tensors of one scene, deterministic in the scene. Training only.

        Args:
            scene_api: scene interface anchored at the current frame.
            cached: tensors already read from the cache store — empty, or complete
                per cache_codecs. Take cached entries as-is, build the rest.

        Returns:
            the single-sample label bundle, the supervision of one sample.
        """

    @abc.abstractmethod
    def _cache_specs(self) -> dict[str, CacheTensorSpec]:
        """
        Declare for each cached tensor its codec and signature.

        Returns:
            spec by tensor name; empty when the policy caches nothing.
        """

    def cache_codecs(self) -> dict[str, TensorCodec]:
        """
        Cache codec of each cached tensor, needed for cache compression and decompression.

        Returns:
            codec by tensor name, e.g. {"lidar_feature": PngCodec(quantization_scale=255)}.
        """
        return {tensor_name: spec.codec for tensor_name, spec in self._cache_specs().items()}

    def cache_signature(self, dataset: str) -> dict[str, dict[str, str]]:
        """
        Cache signature of each cached tensor, needed for cache stallness detection and invalidation.

        Args:
            dataset: the dataset name of the store's scenes.

        Returns:
            the store's cache signature; empty when the policy caches nothing.
        """
        del dataset
        return {tensor_name: spec.signature for tensor_name, spec in self._cache_specs().items()}

    @abc.abstractmethod
    def compute_loss(
        self,
        labels: LabelsT,
        predictions: PredictionsT,
    ) -> dict[str, torch.Tensor]:
        """
        Computes the per-task losses of one batch, unweighted.

        Args:
            labels: batched label bundle from build_labels.
            predictions: prediction bundle from forward.

        Returns:
            per-task scalar losses, keyed exactly as loss_weights.
        """

    @abc.abstractmethod
    def loss_weights(self, epoch: int) -> dict[str, float]:
        """
        Loss weights for one epoch, keyed like compute_loss, summing to one.

        Normalize with normalize_loss_weights; disabled heads carry weight zero.

        Args:
            epoch: current training epoch

        Returns:
            the weight of every per-task loss
        """

    def compute_trajectory(
        self,
        scene_api: SceneAPI,
        navigation: NavigationConditioning,
    ) -> TrajectorySE2:
        """
        Computes the ego vehicle trajectory of a single scene.

        Args:
            scene_api: scene interface anchored at the current frame (iteration 0).
            navigation: single-scene navigation conditioning, unbatched.

        Returns:
            predicted future ego trajectory, relative to the initial ego rear-axle
            pose (origin at current ego, x forward).
        """
        features = self.build_features(scene_api, {}).apply(
            lambda tensor: tensor.unsqueeze(0),
        )
        return self.compute_trajectories(
            features,
            navigation.apply(lambda tensor: tensor.unsqueeze(0)),
            [scene_api],
        )[0]

    def compute_trajectories(
        self,
        features: FeaturesT,
        navigation: NavigationConditioning,
        scene_apis: Sequence[SceneAPI],
    ) -> list[TrajectorySE2]:
        """
        Computes the ego vehicle trajectories of one batch of scenes.

        Args:
            features: collated feature bundle, one row per scene (see build_features).
            navigation: batched navigation conditioning, one row per scene.
            scene_apis: scene interfaces anchored at their current frame (iteration 0).

        Returns:
            predicted future ego trajectories, each relative to its scene's initial
            ego rear-axle pose (origin at current ego, x forward).
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            predictions = self.forward(
                features.to(device),
                navigation.to(device),
            )
        return self.trajectories_from_predictions(predictions, scene_apis)

    def trajectories_from_predictions(
        self,
        predictions: PredictionsT,
        scene_apis: Sequence[SceneAPI],
    ) -> list[TrajectorySE2]:
        """
        Stamps one batch of predictions onto each scene's trajectory grid.

        Args:
            predictions: the forward output, one row per scene.
            scene_apis: the scenes the rows belong to, anchored at their current frame.

        Returns:
            one trajectory per scene, relative to its anchor rear-axle pose.
        """
        poses_se2_batch = cast(
            npt.NDArray[np.float64],
            predictions.ego_trajectory_se2.detach()
            .cpu()
            .numpy()  # pyright: ignore[reportUnknownMemberType]
            .astype(np.float64),
        )

        num_steps = self.policy_config.trajectory_num_steps
        interval_us = self.policy_config.trajectory_interval_us

        # Basic sanity test
        if poses_se2_batch.shape[1] != num_steps:
            raise ValueError(
                f"{type(self).__name__} predicted {poses_se2_batch.shape[1]} poses, but its "
                f"config asks for {num_steps} "
                f"(horizon_us {self.policy_config.trajectory_horizon_us}, interval_us {interval_us}).",
            )

        # Create timestamps for trajectory points, relative to the scene's initial ego state timestamp.
        trajectories: list[TrajectorySE2] = []
        for batch_index, scene_api in enumerate(scene_apis):
            ego_state_se3: EgoStateSE3 | None = scene_api.get_ego_state_se3_at_iteration(0)
            assert ego_state_se3 is not None, "Ego state should be available for trajectory computation!"
            timestamps: np.ndarray[tuple[int], np.dtype[np.signedinteger]] = (
                ego_state_se3.timestamp.time_us
                + np.arange(
                    1,
                    num_steps + 1,
                )
                * interval_us
            )
            trajectories.append(
                TrajectorySE2(
                    pose_se2_array=poses_se2_batch[batch_index],
                    timestamps_us=timestamps,
                ),
            )
        return trajectories

    def get_training_callbacks(self) -> list[L.Callback]:
        """Returns the lightning callbacks used during training; empty by default."""
        return []

    def compute_metrics(
        self,
        labels: LabelsT,
        predictions: PredictionsT,
    ) -> dict[str, Tensor]:
        """
        Quality metrics of one batch for debug logging; empty by default.

        Args:
            labels: batched label bundle from build_labels.
            predictions: prediction bundle from forward.

        Returns:
            scalar metric tensors by name; the trainer prefixes "metric/".
        """
        del labels, predictions
        return {}

    def verify_contract(
        self,
        *,
        benchmark_config: AbstractBenchmarkConfig | None = None,
        offline_data_source_config: AbstractOfflineDataSourceConfig | None = None,
        scene_filter: SceneFilter | None = None,
    ) -> None:
        """
        Refuses any declaration this policy cannot faithfully consume.

        Args:
            benchmark_config: the benchmark about to score this policy; None = nothing to verify.
            offline_data_source_config: the source about to feed this policy; None = nothing to verify.
            scene_filter: the filter about to select scenes for this policy; None = nothing to verify.

        Raises:
            ValueError: on the first declaration this policy cannot meet.
        """
        if benchmark_config is not None:
            verify_policy_against_benchmark(self.policy_config, benchmark_config)
        if offline_data_source_config is not None:
            verify_policy_against_offline_data_source(self.policy_config, offline_data_source_config)
        if scene_filter is not None:
            verify_policy_against_scene_filter(self.policy_config, scene_filter)

    def augment_features(self, features: FeaturesT) -> FeaturesT:
        """
        Augments a collated feature batch on device; no-op by default.

        Runs after the cache store (so stored tensors stay un-augmented) and outside any
        torch.compile graph. Training batches only.

        Args:
            features: the batched feature bundle, already on the training device.

        Returns:
            the features to train on.
        """
        return features

    def visualize_batch(
        self,
        features: FeaturesT,
        labels: LabelsT | None,
        navigation: NavigationConditioning,
        predictions: PredictionsT,
        scene_api: SceneAPI,
    ) -> dict[str, Tensor]:
        """
        Renders named visualization images (uint8 HWC) for a batch; empty means no visuals.

        Called by VisualizationCallback
        on train/validation samples — the callback moves tensors and logs, the rendering
        knowledge lives with the policy. Default: no visualization.

        The scene is the first sample's, the one every renderer draws; overlays that
        no tensor carries (the ego box, the map route) are read from it.

        Labels are None wherever no ground truth exists, e.g. driving a simulator;
        views that need them are then left out.
        """
        return {}


# The type that the training and evaluation infrastructure handles.
AnyPolicy: TypeAlias = AbstractPolicy[Any, Any, Any]


def normalize_loss_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    Scales loss weights to sum to one, the contract loss_weights returns.

    Args:
        weights: the per-task weights, in any magnitude.

    Returns:
        the same weights, summing to one.

    Raises:
        ValueError: if every weight is zero, leaving nothing to train.
    """
    total: float = sum(weights.values())
    if total == 0:
        raise ValueError(
            "Every task loss weight is zero, so there is nothing to train.",
        )
    return {name: weight / total for name, weight in weights.items()}
