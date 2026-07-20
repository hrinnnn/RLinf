# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import ConcatDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.algorithms.awbc import compute_arm_awbc_weights
from rlinf.config import SupportedModel
from rlinf.data.awbc import attach_awbc_to_openpi_dataloader
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        self.awbc_cfg = cfg.get("awbc", {})
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            awbc_enabled = bool(self.awbc_cfg.get("enabled", False))
            path_entries = (
                list(data_paths)
                if awbc_enabled and isinstance(data_paths, (list, tuple, ListConfig))
                else [data_paths]
            )
            repo_ids = [resolve_lerobot_repo_id(entry) for entry in path_entries]
            if not repo_ids or any(repo_id is None for repo_id in repo_ids):
                raise ValueError(
                    "OpenPI SFT requires data.train_data_paths to be set to a local "
                    "dataset path or LeRobot repo id."
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            data_loaders = []
            for repo_id in repo_ids:
                config = get_openpi_config(
                    self.cfg.actor.model.openpi.config_name,
                    model_path=self.cfg.actor.model.model_path,
                    batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                    repo_id=repo_id,
                    data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
                )
                data_loaders.append(
                    openpi_data_loader.create_data_loader(
                        config, framework="pytorch", shuffle=True
                    )
                )
            data_loader = data_loaders[0]
            if not eval_dataset and awbc_enabled:
                manifest_path = self.awbc_cfg.get("progress_manifest")
                if not manifest_path:
                    raise ValueError(
                        "awbc.progress_manifest is required when AWBC is enabled"
                    )
                combined_dataset = (
                    ConcatDataset(
                        [
                            self._openpi_pytorch_dataloader(loader).dataset
                            for loader in data_loaders
                        ]
                    )
                    if len(data_loaders) > 1
                    else None
                )
                data_loader = attach_awbc_to_openpi_dataloader(
                    data_loader,
                    manifest_path=str(manifest_path),
                    expert_sampling_ratio=(
                        None
                        if self.awbc_cfg.get("expert_sampling_ratio", None) is None
                        else float(self.awbc_cfg.expert_sampling_ratio)
                    ),
                    seed=int(self.cfg.actor.get("seed", 0)) + self._rank,
                    dataset_override=combined_dataset,
                    valid_only=bool(self.awbc_cfg.get("sample_valid_only", True)),
                )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        awbc_metrics: dict[str, Any] = {}
        if bool(self.awbc_cfg.get("enabled", False)):
            if not isinstance(batch, dict):
                raise TypeError("AWBC requires an OpenPI dictionary batch")
            required = (
                "awbc_delta_phi",
                "awbc_episode_length",
                "awbc_valid",
                "awbc_confidence",
                "awbc_source",
            )
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError(f"AWBC batch is missing metadata: {missing}")

            mode = str(self.awbc_cfg.get("mode", "arm_paper_exact"))
            if mode not in {
                "uniform",
                "arm_paper_exact",
                "robodopamine_robust",
            }:
                raise ValueError(f"Unsupported AWBC mode: {mode}")
            robust = mode == "robodopamine_robust"
            gain_clip_value = self.awbc_cfg.get("gain_clip") if robust else None
            gain_clip = (
                tuple(float(value) for value in gain_clip_value)
                if gain_clip_value is not None
                else None
            )
            stats_device = torch.device("cuda", self.device)
            delta_phi = torch.as_tensor(
                batch["awbc_delta_phi"], device=stats_device
            )
            valid = torch.as_tensor(batch["awbc_valid"], device=stats_device)
            if mode == "uniform":
                delta_phi = torch.zeros_like(delta_phi)
                valid = torch.ones_like(valid, dtype=torch.bool)
            result = compute_arm_awbc_weights(
                delta_phi,
                torch.as_tensor(batch["awbc_episode_length"], device=stats_device),
                valid=valid,
                confidence=torch.as_tensor(
                    batch["awbc_confidence"], device=stats_device
                ),
                sigma_multiplier=float(self.awbc_cfg.get("sigma_multiplier", 2.0)),
                negative_delta_policy=(
                    str(self.awbc_cfg.get("negative_delta_policy", "continuous"))
                    if robust
                    else "continuous"
                ),
                confidence_power=(
                    float(self.awbc_cfg.get("confidence_power", 0.0))
                    if robust
                    else 0.0
                ),
                weight_floor=(
                    float(self.awbc_cfg.get("weight_floor", 0.0)) if robust else 0.0
                ),
                gain_clip=gain_clip,
                distributed=self._world_size > 1,
            )
            batch["awbc_weight"] = result.weights

            weights = result.weights.detach()
            sources = torch.as_tensor(batch["awbc_source"], device=weights.device).bool()
            expert_weights = weights[sources]
            policy_weights = weights[~sources]
            awbc_metrics = {
                "awbc_weight_mean": weights.mean().item(),
                "awbc_weight_min": weights.min().item(),
                "awbc_weight_max": weights.max().item(),
                "awbc_zero_weight_rate": (weights <= 0).float().mean().item(),
                "awbc_effective_sample_size": result.effective_sample_size.item(),
                "awbc_gain_mean": result.gain_mean.item(),
                "awbc_gain_std": result.gain_std.item(),
                "awbc_valid_count": result.valid_count,
                "awbc_fallback": int(result.used_fallback),
            }
            if expert_weights.numel() > 0:
                awbc_metrics["awbc_expert_weight_mean"] = expert_weights.mean().item()
            if policy_weights.numel() > 0:
                awbc_metrics["awbc_policy_weight_mean"] = policy_weights.mean().item()

        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item(), **awbc_metrics}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
