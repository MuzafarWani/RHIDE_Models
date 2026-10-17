# src/training/train_loop.py
from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.solver.gurobi_solver import RoundMILPSolver, SolverParams
from src.solver.param_feeds import build_round_coeffs
from src.training.round_scheduler import build_round_batches
from src.training.constraints import assert_feasible_decisions
from src.training.reward import compute_rl_reward
from src.training.logging_hooks import TrainingLogger
from src.training.checkpoints import (
    checkpoint_init_state,
    checkpoint_save_round,
)
from src.federated.priority import PriorityRegistry
from src.federated.messaging.bus import MessagingBus
from src.federated.edge_server.aggregator import EdgeAggregator
from src.federated.cloud_server.global_aggregator import CloudAggregator
from src.federated.cloud_server.distributor import Distributor
from src.models.serialization import load_model
from src.models.xeventnet import build_model
from src.models.split_graph import get_allowed_cuts


DeviceID = str
EdgeID = str


@dataclass
class TrainLoopConfig:
    data_cfg: Dict[str, Any]
    train_cfg: Dict[str, Any]
    fed_cfg: Dict[str, Any]
    split_cfg: Dict[str, Any]
    energy_latency_cfg: Dict[str, Any]
    solver_cfg: Dict[str, Any]
    rl_cfg: Dict[str, Any]
    run_dir: str


class TrainLoop:
    """
    Orchestrates:
      batches → coeffs → β-selection + MILP(z,k)
      → device/edge execution → aggregation → RL → logging
    """

    def __init__(self, cfg: TrainLoopConfig):
        self.cfg = cfg
        self.logger = TrainingLogger(run_dir=cfg.run_dir)
        self.priority = PriorityRegistry(cfg.fed_cfg)
        self.bus = MessagingBus()

        # ------------------------------
        # Build global model
        # ------------------------------
        input_shape_2d = tuple(cfg.train_cfg["input_shape_2d"])
        input_shape_3d = tuple(cfg.train_cfg["input_shape_3d"])
        num_classes = int(cfg.train_cfg.get("num_classes", 5))
        self.full_model = build_model(
            input_shape_2d,
            input_shape_3d,
            num_classes,
        )

        resume_path = cfg.train_cfg.get("resume_model_path")
        if resume_path:
            self.full_model = load_model(resume_path)

        # ------------------------------
        # Federated components
        # ------------------------------
        self.edge_nodes: Dict[EdgeID, EdgeAggregator] = {
            e["id"]: EdgeAggregator(edge_id=e["id"], bus=self.bus, cfg=cfg)
            for e in cfg.fed_cfg["edges"]
        }
        self.cloud = CloudAggregator(cfg=cfg)
        self.distributor = Distributor(bus=self.bus, cfg=cfg)

        # ------------------------------
        # Allowed split points
        # ------------------------------
        self.allowed_cuts = get_allowed_cuts(self.full_model, cfg.split_cfg)

        # ------------------------------
        # Init checkpoint metadata
        # ------------------------------
        checkpoint_init_state(
            cfg.run_dir,
            {
                "train_cfg": cfg.train_cfg,
                "fed_cfg": cfg.fed_cfg,
                "split_cfg": cfg.split_cfg,
                "energy_latency_cfg": cfg.energy_latency_cfg,
                "solver_cfg": cfg.solver_cfg,
                "rl_cfg": cfg.rl_cfg,
                "allowed_cuts": [c.name for c in self.allowed_cuts],
            },
        )

    # ==========================================================
    # MAIN TRAIN LOOP
    # ==========================================================
    def run(self) -> None:
        num_rounds = int(self.cfg.train_cfg["num_rounds"])

        for t in range(num_rounds):
            t0 = time.time()

            # --------------------------------------------------
            # 1) Build per-device batches
            # --------------------------------------------------
            device_batches = build_round_batches(
                data_cfg=self.cfg.data_cfg,
                fed_cfg=self.cfg.fed_cfg,
                train_cfg=self.cfg.train_cfg,
                priority=self.priority,
                rng_seed=self.cfg.train_cfg.get("rng_seed", 42),
            )

            # --------------------------------------------------
            # 2) Build round coefficients
            # --------------------------------------------------
            coeffs, per_device_meta = build_round_coeffs(
                device_batches=device_batches,
                full_model=self.full_model,
                allowed_cuts=self.allowed_cuts,
                energy_latency_cfg=self.cfg.energy_latency_cfg,
                fed_cfg=self.cfg.fed_cfg,
                split_cfg=self.cfg.split_cfg,
            )

            # --------------------------------------------------
            # 3) Solve round (β fixed, MILP for z,k)
            # --------------------------------------------------
            solver_params = SolverParams(**self.cfg.solver_cfg)
            solver = RoundMILPSolver(coeffs, solver_params)
            solver_out = solver.solve()

            # --------------------------------------------------
            # 4) Normalize solver output → execution decisions
            # --------------------------------------------------
            decisions: Dict[DeviceID, Dict[str, Any]] = {}

            for device_id in coeffs.statics.keys():
                beta = solver_out.beta_choice.get(device_id, "")
                z = int(solver_out.z.get(device_id, 0))
                k = float(solver_out.k.get(device_id, 0.0))

                # Enforce solver contract defensively
                if beta == "":
                    z = 0  # no feasible split → no local training

                decisions[device_id] = {
                    "device_id": device_id,
                    "beta": beta,                 # "" means raw-transfer only
                    "participate": bool(z),
                    "k_transfer": k,
                    "edge_id": per_device_meta[device_id]["edge_id"],
                    "batch_ids": [b["id"] for b in device_batches[device_id]],
                }

            # --------------------------------------------------
            # 5) Sanity checks (dev/debug mode)
            # --------------------------------------------------
            assert_feasible_decisions(
                decisions=decisions,
                meta=per_device_meta,
                energy_latency_cfg=self.cfg.energy_latency_cfg,
                split_cfg=self.cfg.split_cfg,
            )

            # --------------------------------------------------
            # 6) Execute: device → edge → cloud
            # --------------------------------------------------
            self.bus.publish("work_orders", decisions)

            for edge in self.edge_nodes.values():
                edge.step_round(round_index=t)

            updated = self.cloud.maybe_aggregate_and_update(round_index=t)
            if updated:
                self.distributor.broadcast_to_edges(self.cloud.get_weights())

            # --------------------------------------------------
            # 7) Collect realized accounting
            # --------------------------------------------------
            realized_energy = self.bus.collect("energy_used", default={})
            realized_latency = self.bus.collect("latency_used", default={})
            realized_upload = self.bus.collect("bytes_uploaded", default={})

            # --------------------------------------------------
            # 8) RL reward (edge-level)
            # --------------------------------------------------
            rewards = compute_rl_reward(
                realized_energy=realized_energy,
                realized_latency=realized_latency,
                fed_cfg=self.cfg.fed_cfg,
                rl_cfg=self.cfg.rl_cfg,
            )
            for edge_id, rew in rewards.items():
                self.edge_nodes[edge_id].rl_learn(reward=rew)

            # --------------------------------------------------
            # 9) Logging + checkpoint
            # --------------------------------------------------
            metrics = {
                "mean_energy": float(np.mean(list(realized_energy.values()) or [0.0])),
                "mean_latency": float(np.mean(list(realized_latency.values()) or [0.0])),
                "mean_upload": float(np.mean(list(realized_upload.values()) or [0.0])),
                "round_time_s": time.time() - t0,
            }

            self.logger.log_round(
                round_idx=t,
                decisions=decisions,
                metrics=metrics,
                extras={"rewards": rewards},
            )

            checkpoint_save_round(
                run_dir=self.cfg.run_dir,
                round_idx=t,
                model=self.full_model,
                edge_states={eid: self.edge_nodes[eid].state_dict()
                             for eid in self.edge_nodes},
                solver_warm=None,
                logs={
                    "decisions": decisions,
                    "metrics": metrics,
                },
            )

            print(json.dumps({"round": t, **metrics}, indent=2))

