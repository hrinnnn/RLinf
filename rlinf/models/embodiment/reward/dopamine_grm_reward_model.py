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

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

logger = logging.getLogger(__name__)

DOPAMINE_GRM_SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START - Robot Front Image (task just starting): <image>
- REFERENCE END - Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = "just beginning"; REFERENCE END = "fully completed."
   - Visually estimate how far BEFORE and AFTER are along this START->END continuum.
2) Direction:
   - AFTER better than BEFORE -> positive score.
   - AFTER worse than BEFORE -> negative score.
   - Essentially the same -> 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the Front view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the Left/Right Wrist views to inspect fine-grained gripper state, contact, slippage, misalignment, unintended contact, or occluded collisions.
   - When views disagree, prioritize the view that provides decisive cues for the criterion at hand. Wrist views often override for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success, let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that do not affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged -> 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score> or <score>-NN%</score> or <score>0%</score>
"""


@dataclass(frozen=True)
class ParsedGRMScore:
    raw_score: float
    valid: bool


def parse_dopamine_grm_score(text: str) -> ParsedGRMScore:
    match = re.search(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*%\s*</score>", str(text))
    if match is None:
        return ParsedGRMScore(0.0, False)
    try:
        raw_score = float(match.group(1)) / 100.0
    except ValueError:
        return ParsedGRMScore(0.0, False)
    return ParsedGRMScore(float(np.clip(raw_score, -1.0, 1.0)), True)


def dopamine_grm_score_to_phi(
    mode: str,
    raw_score: float,
    prev_phi: float,
    phi_clip: tuple[float, float] = (0.0, 1.0),
) -> float:
    raw_score = float(np.clip(raw_score, -1.0, 1.0))
    prev_phi = float(np.clip(prev_phi, 0.0, 1.0))
    if mode == "incremental":
        if raw_score >= 0:
            phi = prev_phi + (1.0 - prev_phi) * raw_score
        else:
            phi = prev_phi + prev_phi * raw_score
    elif mode == "forward":
        phi = raw_score
    elif mode == "backward":
        phi = 1.0 + raw_score
    else:
        raise ValueError(f"Unsupported Dopamine GRM mode: {mode}")
    return float(np.clip(phi, phi_clip[0], phi_clip[1]))


def fuse_valid_phis(phis: list[float]) -> float | None:
    if not phis:
        return None
    return float(np.mean(np.asarray(phis, dtype=np.float32)))


def consistency_aware_phi(
    prev_phi: float,
    incremental_phi: float,
    forward_phi: float,
    backward_phi: float,
    alpha: float,
    epsilon: float,
    phi_clip: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Implement Robo-Dopamine Eq. 9--11 for online progress updates."""
    global_mean = (forward_phi + backward_phi) / 2.0
    discrepancy = abs(backward_phi - forward_phi) / (global_mean + epsilon)
    confidence = float(np.exp(-alpha * discrepancy**2))
    incremental_delta = incremental_phi - prev_phi
    phi = prev_phi + confidence / 2.0 * (
        global_mean - prev_phi + incremental_delta
    )
    return (
        float(np.clip(phi, phi_clip[0], phi_clip[1])),
        float(global_mean),
        float(discrepancy),
        confidence,
    )


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f"Unsupported GRM image type: {type(image)}")

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        if np.max(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Invalid GRM image shape: {arr.shape}")
    return Image.fromarray(arr[..., :3]).convert("RGB")


def _image_to_data_url(image: Any) -> str:
    pil_image = _to_pil_image(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _slice_value(value: Any, env_idx: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[env_idx]
    if isinstance(value, np.ndarray):
        return value[env_idx]
    if isinstance(value, (list, tuple)):
        return value[env_idx]
    return value


def _clone_obs(obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
    cloned = {}
    for key in (
        "main_images",
        "wrist_images",
        "reference_start_main_images",
        "reference_start_wrist_images",
        "task_descriptions",
        "task_ids",
    ):
        if key not in obs:
            continue
        value = _slice_value(obs[key], env_idx)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().clone()
        elif isinstance(value, np.ndarray):
            value = value.copy()
        cloned[key] = value
    return cloned


class DopamineGRMRewardModel(BaseRewardModel):
    """Robo-Dopamine GRM client reward model with potential-based shaping."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.endpoint = str(cfg.get("grm_endpoint", ""))
        self.model_name = str(cfg.get("model_name", cfg.get("model_path", "")))
        self.goal_bank_dir = Path(str(cfg.get("goal_bank_dir", "")))
        self.modes = list(cfg.get("modes", ["incremental", "forward", "backward"]))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.grm_interval_chunks = max(1, int(cfg.get("grm_interval_chunks", 1)))
        self.gamma_eff = self.gamma**self.grm_interval_chunks
        self.invalid_reward = float(cfg.get("invalid_reward", 0.0))
        self.enable_consistency_check = bool(cfg.get("consistency_check", True))
        self.consistency_alpha = float(cfg.get("consistency_alpha", 1.0))
        self.consistency_epsilon = float(cfg.get("consistency_epsilon", 1e-6))
        self.terminal_potential = float(cfg.get("terminal_potential", 0.0))
        phi_clip = cfg.get("phi_clip", [0.0, 1.0])
        self.phi_clip = (float(phi_clip[0]), float(phi_clip[1]))
        self.request_timeout = float(cfg.get("request_timeout", 120.0))
        self.max_tokens = int(cfg.get("max_tokens", 32))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.num_envs = int(cfg.get("num_envs", 1))
        metrics_log_path = cfg.get("metrics_log_path", None)
        self.metrics_log_path = Path(str(metrics_log_path)) if metrics_log_path else None
        if self.metrics_log_path is not None:
            self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.prev_phi = torch.zeros(self.num_envs, dtype=torch.float32)
        self.has_prev_phi = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reward_call_counts = torch.zeros(self.num_envs, dtype=torch.long)
        self.previous_grm_obs: list[dict[str, Any] | None] = [None] * self.num_envs
        self.goal_bank = self._load_goal_bank(self.goal_bank_dir)

    def forward(
        self, input_data: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "DopamineGRMRewardModel is an inference-time reward model."
        )

    def _load_goal_bank(self, goal_bank_dir: Path) -> dict[str, dict[str, Any]]:
        if not goal_bank_dir:
            return {}
        if not goal_bank_dir.exists():
            logger.warning("Dopamine GRM goal bank does not exist: %s", goal_bank_dir)
            return {}
        goal_bank: dict[str, dict[str, Any]] = {}
        for meta_path in goal_bank_dir.glob("task_*/meta.json"):
            with open(meta_path, encoding="utf-8") as file:
                meta = json.load(file)
            task_id = meta.get("task_id")
            task_description = str(meta.get("task_description", "")).strip()
            views = dict(meta.get("views", {}))
            entry = {
                "task_id": task_id,
                "task_description": task_description,
                "main": meta_path.parent / views.get("main", "goal_main.png"),
                "wrist": (
                    meta_path.parent / views["wrist"] if views.get("wrist") else None
                ),
            }
            if task_id is not None:
                goal_bank[f"id:{int(task_id)}"] = entry
            if task_description:
                goal_bank[f"desc:{task_description}"] = entry
        return goal_bank

    def _batch_size(self, observations: dict[str, Any]) -> int:
        for key in ("main_images", "task_descriptions", "dones"):
            value = observations.get(key)
            if value is None:
                continue
            return len(value)
        raise ValueError("Cannot infer Dopamine GRM batch size.")

    def _extract_success(
        self, observations: dict[str, Any], batch_size: int
    ) -> torch.Tensor:
        success = torch.zeros(batch_size, dtype=torch.bool)
        env_infos = observations.get("env_infos")
        if not isinstance(env_infos, dict):
            return success

        candidates: list[Any] = []
        for info_dict in (
            env_infos,
            env_infos.get("episode"),
            env_infos.get("final_info"),
            env_infos.get("final_info", {}).get("episode")
            if isinstance(env_infos.get("final_info"), dict)
            else None,
        ):
            if isinstance(info_dict, dict):
                candidates.extend(
                    info_dict.get(key)
                    for key in ("success", "success_once", "success_at_end")
                    if key in info_dict
                )

        for candidate in candidates:
            if candidate is None:
                continue
            tensor = torch.as_tensor(candidate).reshape(-1).bool().cpu()
            if tensor.numel() == batch_size:
                success |= tensor
        return success

    def _goal_entry(self, observations: dict[str, Any], env_idx: int) -> dict[str, Any]:
        task_id = observations.get("task_ids")
        if task_id is not None:
            task_id_value = int(torch.as_tensor(_slice_value(task_id, env_idx)).item())
            entry = self.goal_bank.get(f"id:{task_id_value}")
            if entry is not None:
                return entry

        task_description = str(_slice_value(observations["task_descriptions"], env_idx))
        entry = self.goal_bank.get(f"desc:{task_description.strip()}")
        if entry is None:
            task_id_value = (
                int(torch.as_tensor(_slice_value(task_id, env_idx)).item())
                if task_id is not None
                else None
            )
            raise KeyError(
                "No Dopamine GRM goal image for "
                f"task_id={task_id_value} task={task_description!r}"
            )
        return entry

    def _image_triplet(self, obs: dict[str, Any]) -> tuple[Any, Any, Any]:
        main = obs.get("main_images")
        wrist = obs.get("wrist_images")
        if wrist is None:
            wrist = main
        return main, wrist, wrist

    def _build_mode_payload(
        self,
        task: str,
        reference_start: dict[str, Any],
        goal_entry: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        start_main = reference_start.get("reference_start_main_images")
        if start_main is None:
            start_main = reference_start.get("main_images")
        goal_main = goal_entry["main"]
        before_main, before_left, before_right = self._image_triplet(before)
        after_main, after_left, after_right = self._image_triplet(after)
        return {
            "task": task,
            "images": [
                start_main,
                goal_main,
                before_main,
                before_left,
                before_right,
                after_main,
                after_left,
                after_right,
            ],
        }

    def _build_messages(self, task: str, images: list[Any]) -> list[dict[str, Any]]:
        prompt_parts = DOPAMINE_GRM_SYSTEM_PROMPT.format(task=task).split("<image>")
        content: list[dict[str, Any]] = []
        for idx, text in enumerate(prompt_parts):
            if text:
                content.append({"type": "text", "text": text})
            if idx < len(images):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(images[idx])},
                    }
                )
        return [{"role": "user", "content": content}]

    def _request_grm(self, payloads: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
        if not self.endpoint:
            raise ValueError("reward.model.grm_endpoint must be set for dopamine_grm")
        outputs = []
        latencies = []
        for payload in payloads:
            body = {
                "model": self.model_name,
                "messages": self._build_messages(payload["task"], payload["images"]),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_time = time.perf_counter()
            try:
                with urllib.request.urlopen(
                    request, timeout=self.request_timeout
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                logger.warning("Dopamine GRM request failed: %s", exc)
                outputs.append("")
                latencies.append(time.perf_counter() - start_time)
                continue
            outputs.append(
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", data.get("choices", [{}])[0].get("text", ""))
            )
            latencies.append(time.perf_counter() - start_time)
        return outputs, latencies

    def _write_metric_record(self, record: dict[str, Any]) -> None:
        if self.metrics_log_path is None:
            return
        record = {
            "time_unix": time.time(),
            "model_type": "dopamine_grm",
            **record,
        }
        with open(self.metrics_log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _reference_start_obs(self, current_obs: dict[str, Any]) -> dict[str, Any]:
        start_main = current_obs.get("reference_start_main_images")
        if start_main is None:
            start_main = current_obs.get("main_images")
        start_wrist = current_obs.get("reference_start_wrist_images")
        if start_wrist is None:
            start_wrist = current_obs.get("wrist_images")
        return {
            "main_images": start_main,
            "wrist_images": start_wrist,
            "reference_start_main_images": start_main,
        }

    def _compute_env_reward(
        self,
        observations: dict[str, Any],
        env_idx: int,
        success: bool,
        done: bool,
    ) -> float:
        episode_terminal = bool(done or success)
        self.reward_call_counts[env_idx] += 1
        call_count = int(self.reward_call_counts[env_idx].item())
        if (
            self.reward_call_counts[env_idx].item() % self.grm_interval_chunks != 0
            and not episode_terminal
        ):
            self._write_metric_record(
                {
                    "env_idx": env_idx,
                    "call_count": call_count,
                    "success": success,
                    "done": done,
                    "skipped": True,
                    "grm_interval_chunks": self.grm_interval_chunks,
                    "shaping_reward": 0.0,
                }
            )
            return 0.0

        env_start_time = time.perf_counter()
        current_obs = _clone_obs(observations, env_idx)
        if not self.has_prev_phi[env_idx]:
            self.prev_phi[env_idx] = 0.0
            self.previous_grm_obs[env_idx] = self._reference_start_obs(current_obs)
            self.has_prev_phi[env_idx] = True

        previous_obs = self.previous_grm_obs[env_idx] or self._reference_start_obs(
            current_obs
        )
        reference_start = self._reference_start_obs(current_obs)
        goal_entry = self._goal_entry(observations, env_idx)
        task = str(_slice_value(observations["task_descriptions"], env_idx))

        mode_before = {
            "incremental": previous_obs,
            "forward": reference_start,
            "backward": {
                "main_images": goal_entry["main"],
                "wrist_images": goal_entry["wrist"] or goal_entry["main"],
            },
        }
        payloads = [
            self._build_mode_payload(
                task=task,
                reference_start=reference_start,
                goal_entry=goal_entry,
                before=mode_before[mode],
                after=current_obs,
            )
            for mode in self.modes
        ]
        outputs, request_latencies = self._request_grm(payloads)
        valid_phis = []
        prev_phi = float(self.prev_phi[env_idx].item())
        mode_records: dict[str, dict[str, Any]] = {}
        for mode, output in zip(self.modes, outputs):
            parsed = parse_dopamine_grm_score(output)
            mode_record = {
                "valid": parsed.valid,
                "raw_score": parsed.raw_score if parsed.valid else None,
                "phi": None,
            }
            if not parsed.valid:
                mode_records[mode] = mode_record
                continue
            phi = dopamine_grm_score_to_phi(
                mode, parsed.raw_score, prev_phi, self.phi_clip
            )
            mode_record["phi"] = phi
            mode_records[mode] = mode_record
            valid_phis.append(phi)

        phi_by_mode = {
            mode: record["phi"]
            for mode, record in mode_records.items()
            if record["valid"] and record["phi"] is not None
        }
        consistency_applied = False
        fusion_method = "mean_valid"
        global_mean = None
        normalized_discrepancy = None
        confidence = None
        if success:
            # LIBERO's success signal is authoritative. The state is reset immediately
            # after this transition, while terminal_potential keeps PBRS episodic-safe.
            phi_next = 1.0
            fusion_method = "success_override"
        elif self.enable_consistency_check and all(
            mode in phi_by_mode for mode in ("incremental", "forward", "backward")
        ):
            phi_next, global_mean, normalized_discrepancy, confidence = (
                consistency_aware_phi(
                    prev_phi=prev_phi,
                    incremental_phi=float(phi_by_mode["incremental"]),
                    forward_phi=float(phi_by_mode["forward"]),
                    backward_phi=float(phi_by_mode["backward"]),
                    alpha=self.consistency_alpha,
                    epsilon=self.consistency_epsilon,
                    phi_clip=self.phi_clip,
                )
            )
            consistency_applied = True
            fusion_method = "consistency_aware"
        elif not success:
            fused_phi = fuse_valid_phis(valid_phis)
            if fused_phi is None:
                if done:
                    # A zero terminal potential preserves the finite-episode PBRS boundary.
                    reward = self.gamma_eff * self.terminal_potential - prev_phi
                    self._write_metric_record(
                        {
                            "env_idx": env_idx,
                            "call_count": call_count,
                            "task": task,
                            "success": success,
                            "done": done,
                            "valid": False,
                            "skipped": False,
                            "prev_phi": prev_phi,
                            "phi_fused": None,
                            "phi_next": None,
                            "potential_next_for_shaping": self.terminal_potential,
                            "shaping_reward": reward,
                            "fusion_method": "terminal_boundary_invalid_grm",
                            "mode_metrics": mode_records,
                            "valid_mode_count": 0,
                            "invalid_mode_count": len(self.modes),
                            "grm_latency_sec": float(sum(request_latencies)),
                            "grm_latency_by_mode_sec": dict(zip(self.modes, request_latencies)),
                            "total_latency_sec": time.perf_counter() - env_start_time,
                        }
                    )
                    return float(reward)
                self._write_metric_record(
                    {
                        "env_idx": env_idx,
                        "call_count": call_count,
                        "task": task,
                        "success": success,
                        "done": done,
                        "valid": False,
                        "skipped": False,
                        "prev_phi": prev_phi,
                        "phi_fused": None,
                        "phi_next": None,
                        "shaping_reward": self.invalid_reward,
                        "fusion_method": "invalid_grm",
                        "mode_metrics": mode_records,
                        "valid_mode_count": 0,
                        "invalid_mode_count": len(self.modes),
                        "grm_latency_sec": float(sum(request_latencies)),
                        "grm_latency_by_mode_sec": dict(zip(self.modes, request_latencies)),
                        "total_latency_sec": time.perf_counter() - env_start_time,
                    }
                )
                return self.invalid_reward
            phi_next = fused_phi

        potential_next = self.terminal_potential if episode_terminal else phi_next
        reward = self.gamma_eff * potential_next - prev_phi
        if not episode_terminal:
            self.prev_phi[env_idx] = float(phi_next)
            self.previous_grm_obs[env_idx] = current_obs
        valid_mode_count = len(valid_phis)
        self._write_metric_record(
            {
                "env_idx": env_idx,
                "call_count": call_count,
                "task": task,
                "success": success,
                "done": done,
                "episode_terminal": episode_terminal,
                "valid": True,
                "skipped": False,
                "prev_phi": prev_phi,
                "phi_fused": phi_next,
                "phi_next": phi_next,
                "potential_next_for_shaping": potential_next,
                "shaping_reward": float(reward),
                "fusion_method": fusion_method,
                "consistency_applied": consistency_applied,
                "global_mean_phi": global_mean,
                "normalized_discrepancy": normalized_discrepancy,
                "consistency_confidence": confidence,
                "mode_metrics": mode_records,
                "valid_mode_count": valid_mode_count,
                "invalid_mode_count": len(self.modes) - valid_mode_count,
                "grm_latency_sec": float(sum(request_latencies)),
                "grm_latency_by_mode_sec": dict(zip(self.modes, request_latencies)),
                "total_latency_sec": time.perf_counter() - env_start_time,
            }
        )
        return float(reward)

    @torch.no_grad()
    def compute_reward(self, observations: dict[str, Any]) -> torch.Tensor:
        batch_size = self._batch_size(observations)
        if batch_size > self.num_envs:
            raise ValueError(
                "DopamineGRMRewardModel got batch size "
                f"{batch_size}, but num_envs={self.num_envs}"
            )
        success = self._extract_success(observations, batch_size)
        dones = observations.get("dones")
        dones_tensor = (
            torch.as_tensor(dones).reshape(-1).bool().cpu()
            if dones is not None
            else torch.zeros(batch_size, dtype=torch.bool)
        )
        rewards = torch.zeros(batch_size, dtype=torch.float32)
        for env_idx in range(batch_size):
            rewards[env_idx] = self._compute_env_reward(
                observations,
                env_idx,
                bool(success[env_idx].item()),
                bool(dones_tensor[env_idx].item()),
            )
            if bool(dones_tensor[env_idx].item()) or bool(success[env_idx].item()):
                self.prev_phi[env_idx] = 0.0
                self.has_prev_phi[env_idx] = False
                self.previous_grm_obs[env_idx] = None
                self.reward_call_counts[env_idx] = 0
        return rewards
