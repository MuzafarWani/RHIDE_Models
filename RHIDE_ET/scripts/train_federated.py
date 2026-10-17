#!/usr/bin/env python3
# scripts/train_federated_xevenenet.py
from __future__ import annotations

import os
import argparse
import csv
import time
import numpy as np
import tensorflow as tf

from src.utils.io_helpers import read_yaml, ensure_dir
from src.utils.seeding import set_global_seed
from src.data.wrf_reader import WRFReader
from src.data.dataset_builder import build_labeled_arrays
from src.data.normalization import load_stats, apply_channel_stats
from src.data.encoders import SimpleLabelEncoder
from src.models.xeventnet import build_model
from src.models.split_graph import create_submodels
from src.models.serialization import make_device_edge_from_full
from src.federated.priority import PriorityRegistry

# Messaging-driven imports
from src.federated.messaging.bus import MessagingBus
from src.federated.messaging.messages import ModelUpdate, CutGradients  # include CutGradients
from src.federated.device_client.uploader import Uploader
from src.federated.edge_server.aggregator import EdgeAggregator
from src.federated.cloud_server.global_aggregator import CloudAggregator
from src.federated.cloud_server.distributor import Distributor
from src.federated.cloud_server.rl_agent import CloudRLAggregationAgent, CloudRLConfig

# NEW: device-side trainer for split-backward simulation
from src.federated.device_client.trainer import DeviceTrainer

# RL / Edge state
from src.federated.edge_server.edge_state import EdgeState  # noqa: F401

# Solver
from src.solver.mappings import default_betas
from src.solver.param_feeds import DeviceStatic, BetaCoeffs, build_round_coeffs
from src.solver.gurobi_solver import RoundMILPSolver, SolverParams
from src.solver.postprocess import summarize_solution
from src.solver.warm_start import pack_warm_start, save_warm_start, load_warm_start

# Priority-aware scheduler
from src.training.round_scheduler import build_round_batches

# Accounting
from src.accounting.energy_latency_log import EnergyLatencyLog

# Checkpoints (NEW)
from src.training.checkpoints import (
    checkpoint_init_state,
    checkpoint_save_round,
    checkpoint_load_latest,
)

# Optional eval hooks
try:
    from src.eval.reports import evaluate_model as _ext_evaluate_model
    from src.eval.reports import write_round_report as _ext_write_round_report
except Exception:
    _ext_evaluate_model = None
    _ext_write_round_report = None


def _one_hot(idx: int, n: int) -> list[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v

def _print_pre_solve_tables(rc, allowed):
    print("\n[pre-solve] ===== per-device feasibility & costs (per batch) =====")
    for dev_id, st in rc.statics.items():
        bc = rc.beta_coeffs[dev_id]
        Bi = int(rc.batches_per_device.get(dev_id, 0))
        raw_E = float(rc.transfer_energy_J[dev_id])                # k=1 energy
        raw_U = float(st.RBS_bytes)                                # k=1 upload
        raw_T = float(rc.transfer_time_s[dev_id])                  # k=1 time
        feas_raw = dict(
            E=(raw_E <= st.Emax_round_J and raw_E <= st.Erem_J),
            U=(raw_U <= st.Umax_bytes),
            T=(raw_T <= st.Tmax_s),
        )
        print(f"[pre-solve][{dev_id}] Batches={Bi}  Emax={st.Emax_round_J:.3f} Erem={st.Erem_J:.3f} "
              f"Umax={st.Umax_bytes} Tmax={st.Tmax_s:.3f} Mem={st.mem_max_bytes}")
        print(f"  RAW(k=1):  E={raw_E:.6f}J  U={int(raw_U)}B  T={raw_T:.6f}s  "
              f"feas(E={feas_raw['E']},U={feas_raw['U']},T={feas_raw['T']})")

        print("  β candidates (z=1) per-batch (+edge+sync in T):")
        for b_idx, b in enumerate(rc.betas):
            E = bc.Ecomp_J[b_idx] + bc.Ecomm_J[b_idx]
            U = bc.U_bytes[b_idx]
            T = bc.Tcomp_s[b_idx] + bc.Tcomm_s[b_idx] + (st.Tedge_s + st.Tsync_s)
            M = bc.M_bytes[b_idx]
            feas = dict(
                E=(E <= st.Emax_round_J and E <= st.Erem_J),
                U=(U <= st.Umax_bytes),
                T=(T <= st.Tmax_s),
                M=(M <= st.mem_max_bytes),
            )
            print(f"    β={b.name:14s}  E={E:.6f}J  U={int(U)}B  T={T:.6f}s  M={int(M)}B  "
                  f"feas(E={feas['E']},U={feas['U']},T={feas['T']},M={feas['M']})")
    print("[pre-solve] =======================================================\n")


def _fallback_evaluate_model_keras(model: tf.keras.Model, X2, X3, y, batch_size: int = 64) -> dict:
    if X2 is None or X3 is None or y is None or len(y) == 0:
        return {}
    results = model.evaluate({"input_2d": X2, "input_3d": X3}, y, verbose=0, batch_size=batch_size)
    metrics = {}
    if isinstance(results, (list, tuple)):
        names = getattr(model, "metrics_names", None)
        if names and len(names) == len(results):
            metrics = {names[i]: float(results[i]) for i in range(len(results))}
        else:
            if len(results) >= 1:
                metrics["loss"] = float(results[0])
            if len(results) >= 2:
                metrics["accuracy"] = float(results[1])
    else:
        metrics["loss"] = float(results)
    return metrics


def _fallback_write_round_report_csv(path: str, round_idx: int, metrics: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    file_exists = os.path.exists(path)
    import csv as _csv
    with open(path, "a", newline="") as f:
        w = _csv.writer(f)
        if not file_exists:
            w.writerow(["round"] + list(metrics.keys()))
        w.writerow([int(round_idx)] + [metrics[k] for k in metrics.keys()])


def _maybe_np_dtype(x):
    """Helper: accept numpy dtype object or string like 'float16'."""
    if x is None:
        return None
    try:
        return np.dtype(x)
    except Exception:
        return None


# Track split points
def log_split_details(z_i, beta, dev_id, edge_id, batch_id, X2b, X3b, edge_aggs, train_cfg):
    # ✅ HARD GUARD: raw-only case
    if z_i == 1 and not beta:
        print(f"[train][split] Device {dev_id} on Edge {edge_id} - No feasible β, RAW mode")
        return

    if z_i == 1:
        print(f"[train][split] Device {dev_id} on Edge {edge_id} - Split training activated for batch {batch_id} with beta {beta}")

        if beta == "output":
            print(f"[train][split] Device {dev_id} - Output layer training")
            return

        print(f"[train][split] Device {dev_id} - Performing split forward pass using beta {beta}")

        device_net, edge_net = create_submodels(
            full_model=edge_aggs[edge_id].trainer.full_model,
            beta_name=beta,
            input_shape_2d=tuple(train_cfg["input_shape_2d"]),
            input_shape_3d=tuple(train_cfg["input_shape_3d"]),
            num_classes=int(train_cfg["num_classes"]),
        )
        make_device_edge_from_full(edge_aggs[edge_id].trainer.full_model, device_net, edge_net)

        _ = device_net.predict({"input_2d": X2b, "input_3d": X3b}, verbose=0)
        print(f"[train][split] Device {dev_id} - Split activations generated")
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", default="configs/data.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    ap.add_argument("--fed_cfg", default="configs/federated.yaml")
    ap.add_argument("--split_cfg", default="configs/split.yaml")
    ap.add_argument("--el_cfg", default="configs/energy_latency.yaml")
    ap.add_argument("--solver_cfg", default="configs/solver.yaml")
    ap.add_argument("--rl_cfg", default="configs/rl.yaml")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_global_seed(args.seed, deterministic_tf=True)

    data_cfg = read_yaml(args.data_cfg)
    train_cfg = read_yaml(args.train_cfg)
    fed_cfg = read_yaml(args.fed_cfg)
    split_cfg = read_yaml(args.split_cfg)
    el_cfg = read_yaml(args.el_cfg)
    solver_cfg = read_yaml(args.solver_cfg)
    rl_cfg = read_yaml(args.rl_cfg)

    out_dir = train_cfg.get("out_dir", "runs/expA")
    ensure_dir(out_dir)

    # ----- Load dataset & normalize (TRAIN) -----
    wrf = WRFReader(data_cfg["wrf_file"], var_map=data_cfg["var_map"]).read_all()
    enc = SimpleLabelEncoder.load(data_cfg["label_encoder_out"])
    X2_tr, X3_tr, y_tr, _ = build_labeled_arrays(
        wrf=wrf,
        locations_csv=data_cfg["locations_train_csv"],
        window_hw=int(data_cfg["window_half_width"]),
        feature2d_keys=data_cfg["feature2d_keys"],
        feature3d_keys=data_cfg["feature3d_keys"],
        classes=data_cfg["classes"],
        label_encoder=enc,
    )
    stats = load_stats(data_cfg["norm_stats_out"])
    X2_tr, X3_tr = apply_channel_stats(X2_tr, X3_tr, stats)

    # ----- Optional VALIDATION -----
    X2_val = X3_val = y_val = None
    if data_cfg.get("locations_val_csv"):
        X2_val, X3_val, y_val, _ = build_labeled_arrays(
            wrf=wrf,
            locations_csv=data_cfg["locations_val_csv"],
            window_hw=int(data_cfg["window_half_width"]),
            feature2d_keys=data_cfg["feature2d_keys"],
            feature3d_keys=data_cfg["feature3d_keys"],
            classes=data_cfg["classes"],
            label_encoder=enc,
        )
        X2_val, X3_val = apply_channel_stats(X2_val, X3_val, stats)

    # ----- Build global model -----
    full_model = build_model(
        tuple(train_cfg["input_shape_2d"]),
        tuple(train_cfg["input_shape_3d"]),
        num_classes=int(train_cfg["num_classes"]),
    )
    print("[train] model built.")

    # ----- Federation topology & priorities -----
    preg = PriorityRegistry(fed_cfg)
    devs_by_edge = {e["id"]: e.get("devices", []) for e in fed_cfg["edges"]}

    # ----- Split info -----
    allowed = split_cfg["allowed"]
    A_size_map = split_cfg["A_size_bytes"]

    # ----- Solver & ordering safety -----
    betas = default_betas()
    beta_names = [b.name for b in betas]
    if beta_names != list(allowed):
        raise ValueError(
            f"Per-β order mismatch: solver betas={beta_names} vs split.allowed={list(allowed)}."
        )

    # Warm-start
    ws_path = solver_cfg.get("warm_start_path", os.path.join(out_dir, "solver_warm_start.json"))
    use_ws = bool(solver_cfg.get("use_warm_start", True))
    sp = SolverParams(
        eta=float(solver_cfg.get("eta", 0.0)),
        xi=float(solver_cfg.get("xi", 0.0)),
        mip_gap=float(solver_cfg.get("mip_gap", 0.01)),
        time_limit_s=float(solver_cfg.get("time_limit_s", 30)),
        threads=int(solver_cfg.get("threads", 0)),
        warm_start=None,
    )
    if use_ws and os.path.exists(ws_path):
        try:
            sp.warm_start = load_warm_start(ws_path)
            print(f"[solver] loaded warm start from {ws_path}")
        except Exception as ex:
            print(f"[solver] warning: failed to load warm start ({ex}); continuing without it.")

    # ----- Battery state & harvest -----
    e_rem_state = {dev_id: float(dcfg["Erem_J"]) for dev_id, dcfg in el_cfg["devices"].items()}
    harvest_map = {
        dev_id: float(el_cfg["devices"][dev_id].get("harvest_J_per_round", 0.0))
        for dev_id in el_cfg["devices"].keys()
    }
    # ----------------
    # Timing accumulators
    # per-device accumulated training time (seconds) — wall clock measured only for device computation
    device_train_time_accum = {dev_id: 0.0 for dev_id in el_cfg["devices"].keys()}
    # per-round (within loop) aggregate training time (seconds) — reset each round
    round_train_time = 0.0
    # total (global) accumulated training time across all rounds
    total_train_time = 0.0
    

    # Accounting
    energylat_log = EnergyLatencyLog(bus=None)

    # RL reward weights
    rw = rl_cfg.get("reward_weights", {"alpha": 1.0, "beta": 1.0, "gamma": 0.01})
    alpha = float(rw.get("alpha", 1.0))
    beta_w = float(rw.get("beta", 1.0))
    gamma = float(rw.get("gamma", 0.01))

    # Messaging setup
    bus = MessagingBus()
    edge_aggs: dict[str, EdgeAggregator] = {}
    edge_models: dict[str, tf.keras.Model] = {}

    for e in devs_by_edge.keys():
        edge_models[e] = build_model(
            tuple(train_cfg["input_shape_2d"]),
            tuple(train_cfg["input_shape_3d"]),
            num_classes=int(train_cfg["num_classes"]),
        )
        edge_models[e].set_weights(full_model.get_weights())

        class _Cfg: ...
        _cfg = _Cfg()
        _cfg.train_cfg = train_cfg
        _cfg.rl_cfg = rl_cfg

        agg = EdgeAggregator(edge_id=e, bus=bus, cfg=_cfg)
        agg.set_full_model(edge_models[e])
        edge_aggs[e] = agg

    cloud = CloudAggregator(cfg=type("C", (), {"rl_cfg": rl_cfg})())
    cloud.bind_bus(bus)
    distributor = Distributor(bus=bus, cfg=type("D", (), {})())

    # -------- Per-link compression knobs (NEW) --------
    links_cfg = (fed_cfg.get("links") or {}) if isinstance(fed_cfg, dict) else {}
    d2e_cfg = links_cfg.get("device_to_edge", {}) or {}

    # Prefer per-link toggles; fall back to global use_zstd if present; else None -> default inside Uploader
    use_zstd_activ = d2e_cfg.get("use_zstd_activ", d2e_cfg.get("use_zstd", None))
    use_zstd_raw   = d2e_cfg.get("use_zstd_raw",   d2e_cfg.get("use_zstd", None))

    uploader = Uploader(
        bus=bus,
        # global defaults
        use_zstd=bool(d2e_cfg.get("use_zstd", True)),
        cast_dtype=_maybe_np_dtype(d2e_cfg.get("cast_dtype")),
        # per-link overrides for device->edge
        activations_dtype=_maybe_np_dtype(d2e_cfg.get("activations_dtype")),
        raw_dtype=_maybe_np_dtype(d2e_cfg.get("raw_dtype")),
        zstd_level_activ=int(d2e_cfg.get("zstd_level_activ", d2e_cfg.get("zstd_level", 3))),
        zstd_level_raw=int(d2e_cfg.get("zstd_level_raw", d2e_cfg.get("zstd_level", 3))),
        use_zstd_activ=use_zstd_activ,
        use_zstd_raw=use_zstd_raw,
    )

    edge_states = {e: edge_aggs[e].state for e in edge_aggs.keys()}

    # -------- Cloud-level RL (NEW) --------
    cloud_actions = tuple(rl_cfg.get("cloud_actions", [1, 2, 4, 8]))
    cloud_obs_keys = tuple(rl_cfg.get("cloud_obs_keys",
                         ["global_mean_latency", "global_mean_energy", "global_queue"]))
    cloud_cfg = CloudRLConfig(
        obs_keys=cloud_obs_keys,
        actions=cloud_actions,
        policy_hidden=tuple(rl_cfg.get("cloud_policy_hidden", [64, 64])),
        value_hidden=tuple(rl_cfg.get("cloud_value_hidden", [64, 64])),
        clip_ratio=float(rl_cfg.get("cloud_clip_ratio", 0.2)),
        gamma=float(rl_cfg.get("cloud_gamma", 0.99)),
        lam=float(rl_cfg.get("cloud_lambda", 0.95)),
        policy_lr=float(rl_cfg.get("cloud_policy_lr", 3e-4)),
        value_lr=float(rl_cfg.get("cloud_value_lr", 1e-3)),
        train_iters=int(rl_cfg.get("cloud_train_iters", 4)),
        minibatch_size=int(rl_cfg.get("cloud_minibatch", 64)),
        entropy_coef=float(rl_cfg.get("cloud_entropy_coef", 0.01)),
    )
    cloud_rl = CloudRLAggregationAgent(cloud_cfg, seed=args.seed)
    rounds_since_cloud = 0

    # Distinct reward weights for cloud agent (optional)
    crw = rl_cfg.get("cloud_reward_weights", {"alpha": 1.0, "beta": 1.0, "gamma": 0.01})
    c_alpha = float(crw.get("alpha", 1.0))
    c_beta  = float(crw.get("beta", 1.0))
    c_gamma = float(crw.get("gamma", 0.01))

    # ----- Device-side trainers (NEW: split backward simulation) -----
    # One DeviceTrainer per device id; initialize from that device's edge model.
    device_trainers: dict[str, DeviceTrainer] = {}
    for e_id, dev_list in devs_by_edge.items():
        for d in dev_list:
            device_trainers[d] = DeviceTrainer(full_model=edge_models[e_id])

    # Eval cadence
    eval_every = int(train_cfg.get("eval_every", 0))
    eval_batch_size = int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 32)))
    eval_report_csv = train_cfg.get("eval_report_csv", os.path.join(out_dir, "val_metrics.csv"))

    # ---------- Checkpoint options (NEW) ----------
    ckpt_every = int(train_cfg.get("ckpt_every", 1))  # save every N rounds
    resume_from_latest = bool(train_cfg.get("resume_from_latest", False))
    ckpt_rl_save_replay = bool(train_cfg.get("ckpt_rl_save_replay", False))

    # Initialize checkpointing (creates meta.json if absent)
    checkpoint_init_state(out_dir, meta={
        "exp": os.path.basename(out_dir.rstrip("/")),
        "cfgs": {
            "data": args.data_cfg,
            "train": args.train_cfg,
            "fed": args.fed_cfg,
            "split": args.split_cfg,
            "el": args.el_cfg,
            "solver": args.solver_cfg,
            "rl": args.rl_cfg,
        }
    })

    # Helper to grab a save-capable RL agent per edge (best-effort)
    def _collect_rl_agents_for_ckpt() -> dict[str, object]:
        agents = {}
        for e_id, agg in edge_aggs.items():
            if hasattr(agg, "rl") and hasattr(agg.rl, "save"):
                agents[f"edge::{e_id}"] = agg.rl
                continue
            for attr in ("_agent", "agent"):
                inner = getattr(agg.rl, attr, None)
                if inner is not None and hasattr(inner, "save"):
                    agents[f"edge::{e_id}"] = inner
                    break
        # Add cloud-level agent
        agents["cloud"] = cloud_rl
        return agents

    rl_agents_for_ckpt = _collect_rl_agents_for_ckpt()

    # Optional: resume RL agents from latest checkpoint (best-effort)
    if resume_from_latest:
        latest = checkpoint_load_latest(out_dir)
        if latest and rl_agents_for_ckpt:
            rl_root = os.path.join(latest["ckpt_dir"], "rl")
            for e_id, agent in rl_agents_for_ckpt.items():
                edge_dir = os.path.join(rl_root, str(e_id))
                if os.path.isdir(edge_dir) and hasattr(agent, "load"):
                    try:
                        agent.load(edge_dir)
                        print(f"[ckpt] restored RL agent '{e_id}' from {edge_dir}")
                    except TypeError:
                        try:
                            agent.load(edge_dir)
                            print(f"[ckpt] restored RL agent '{e_id}' from {edge_dir}")
                        except Exception as ex:
                            print(f"[ckpt] warning: failed to restore RL agent {e_id}: {ex}")
                    except Exception as ex:
                        print(f"[ckpt] warning: failed to restore RL agent {e_id}: {ex}")

    # Optional: resume cloud agent
    if resume_from_latest:
        latest = checkpoint_load_latest(out_dir)
        if latest:
            rl_root = os.path.join(latest["ckpt_dir"], "rl")
            cloud_dir = os.path.join(rl_root, "cloud")
            if os.path.isdir(cloud_dir):
                try:
                    cloud_rl.load(cloud_dir)
                    print(f"[ckpt] restored Cloud RL agent from {cloud_dir}")
                except Exception as ex:
                    print(f"[ckpt] warning: failed to restore Cloud RL agent: {ex}")

    # RNG for sampling windows per batch descriptor
    rng = np.random.default_rng(args.seed)

    # ======== ROUNDS ========
    N = len(y_tr)
    for t in range(args.rounds):
        print(f"\n[train][round {t}] scheduling & solving MILP ...")
        round_train_time = 0.0

        # ---- Priority-aware scheduling (build per-device batch descriptors) ----
        device_batches = build_round_batches(
            data_cfg=data_cfg,
            fed_cfg=fed_cfg,
            train_cfg=train_cfg,
            priority=preg,               # currently unused inside, but passed for API symmetry
            rng_seed=args.seed + t,      # vary per round
        )
        # Translate to counts for the solver
        batches_per_device = {d: len(blist) for d, blist in device_batches.items()}

        # ---- Round statics & per-β coeffs ----
        devices_static = []
        per_dev_beta = {}
        for dev_id, dcfg in el_cfg["devices"].items():
            devices_static.append(
                DeviceStatic(
                    device_id=dev_id,
                    edge_id=dcfg["edge_id"],
                    priority=preg.device_priority(dev_id),
                    mem_max_bytes=dcfg["mem_max_bytes"],
                    uplink_rate_Bps=dcfg["uplink_rate_Bps"],
                    comm_cost_per_byte_J=dcfg["comm_cost_per_byte_J"],
                    Emax_round_J=dcfg["Emax_round_J"],
                    Erem_J=float(e_rem_state[dev_id]),
                    Umax_bytes=dcfg["Umax_bytes"],
                    Tmax_s=dcfg["Tmax_s"],
                    RBS_bytes=dcfg["RBS_bytes"],
                    Tedge_s=dcfg["Tedge_s"],
                    Tsync_s=dcfg["Tsync_s"],
                )
            )
            pb = dcfg["per_beta"]
            L = len(allowed)
            for key in ("Ecomp_J", "Ecomm_J", "Tcomp_s", "Tcomm_s", "M_bytes"):
                if len(pb[key]) != L:
                    raise ValueError(
                        f"[{dev_id}] per_beta.{key} length {len(pb[key])} != len(split.allowed) {L}."
                    )

            per_dev_beta[dev_id] = BetaCoeffs(
                Ecomp_J=list(pb["Ecomp_J"]),
                Ecomm_J=list(pb["Ecomm_J"]),
                U_bytes=[float(A_size_map[name]) for name in allowed],
                Tcomp_s=list(pb["Tcomp_s"]),
                Tcomm_s=list(pb["Tcomm_s"]),
                M_bytes=list(pb["M_bytes"]),
            )

        # ---- Solve MILP ----
        rc = build_round_coeffs(devices_static, batches_per_device, per_dev_beta, betas=betas)
        # NEW: dump the same feasibility table the MILP “sees”
        try:
            _print_pre_solve_tables(rc, allowed)
        except Exception as ex:
            print(f"[pre-solve] table print failed (non-fatal): {ex}")    
        solver = RoundMILPSolver(rc, sp, integer_k=bool(solver_cfg.get("integer_k", True)))
        sol = solver.solve()

        if use_ws:
            try:
                ws = pack_warm_start(sol.z, sol.beta_choice, sol.k)
                save_warm_start(ws_path, ws)
                sp.warm_start = ws
                print(f"[solver] warm start saved to {ws_path}")
            except Exception as ex:
                print(f"[solver] warning: failed to save warm start ({ex})")

        solsum = summarize_solution(sol)
        print("[train] solution:", {k: solsum[k] for k in [
            "status", "obj", "active_devices", "total_energy_J", "total_upload_bytes", "total_latency_s"
        ]})

        stat_by_id = {d.device_id: d for d in devices_static}

        # ================================
        # Device -> Edge: publish traffic
        # ================================
        for e_id, dev_list in devs_by_edge.items():
            for d in dev_list:
                # If scheduler gave no batches to this device in this round, skip.
                blist = device_batches.get(d, [])
                if not blist:
                    continue

                z_i = sol.z.get(d, 0)
                beta = sol.beta_choice.get(d, "")
                k_i = int(round(sol.k.get(d, 0.0)))
                st = stat_by_id[d]
                
                # --- BEGIN: quick reason when RAW (z_i == 0) is chosen ---
                if z_i == 0:
                    bc = per_dev_beta[d]
                    # Best β energy (per-batch) if we had split
                    best_beta = None
                    best_E = float("inf")
                    best_U = best_T = best_M = None
                    for b_idx, b in enumerate(allowed):
                        E = bc.Ecomp_J[b_idx] + bc.Ecomm_J[b_idx]
                        U = bc.U_bytes[b_idx]
                        T = bc.Tcomp_s[b_idx] + bc.Tcomm_s[b_idx] + (st.Tedge_s + st.Tsync_s)
                        M = bc.M_bytes[b_idx]
                        feasible = (E <= st.Emax_round_J and E <= st.Erem_J and U <= st.Umax_bytes and T <= st.Tmax_s and M <= st.mem_max_bytes)
                        
                        if feasible and E < best_E:
                            best_E, best_U, best_T, best_M, best_beta = E, U, T, M, b
                           
                    # RAW costs per batch (k=1) seen by the solver
                    raw_E = float(rc.transfer_energy_J[d])
                    raw_U = float(st.RBS_bytes)
                    raw_T = float(rc.transfer_time_s[d])
                    
                    print(f"[why-no-split][{d}] best_feasible_beta={best_beta} "
                          f"(E={best_E if best_beta else None}, U={best_U if best_beta else None}, "
                          f"T={best_T if best_beta else None})  "
                          f"vs RAW(E={raw_E:.6f}, U={int(raw_U)}, T={raw_T:.6f}); "
                          f"eta*priority={float(sp.eta)*float(stat_by_id[d].priority):.3f}")
        # --- END: quick reason ---        

                # Per-device batch index for stable batch_id
                for bi, bdesc in enumerate(blist):
                    bsize = int(bdesc["size"])
                    # Sample any windows for now; to bias by labels/geo, plug your sampler here.
                    take = rng.choice(N, size=bsize, replace=True)
                    X2b, X3b, yb = X2_tr[take], X3_tr[take], y_tr[take]

                    # Prepare a stable batch_id for this message
                    batch_id = f"t{t}-dev{d}-b{bi}"
                    
                    log_split_details(z_i, beta, d, e_id, batch_id, X2b, X3b, edge_aggs, train_cfg)

                    bytes_uploaded = 0
                    e_comp_real = 0.0
                    t_comp_real = 0.0
                    t_comm_real = 0.0
                    t_extra = 0.0
                    e_comm_real = 0.0
                    e_transfer_real = 0.0
 
                    if z_i == 1 and beta:
                        if beta == "output":
                            t0 = time.time()
                            edge_aggs[e_id].trainer.full_model.fit(
                                {"input_2d": X2b, "input_3d": X3b},
                                yb, epochs=1, batch_size=int(train_cfg["batch_size"]), verbose=0
                            )
                            t_taken = time.time() - t0
                            device_train_time_accum[d] += t_taken
                            round_train_time += t_taken
                            total_train_time += t_taken
                            
                            b_idx = allowed.index("output")
                            e_comp_real = per_dev_beta[d].Ecomp_J[b_idx] / max(1, len(blist))  # apportion per batch
                            t_comp_real = t_taken / max(1, len(blist))
                            #t_comp_real = per_dev_beta[d].Tcomp_s[b_idx] / max(1, len(blist))
                            t_extra = (st.Tedge_s + st.Tsync_s) / max(1, len(blist))
                        else:
                            # ---- Split forward on device to produce activations ----
                            device_net, edge_net = create_submodels(
                                full_model=edge_aggs[e_id].trainer.full_model,
                                beta_name=beta,
                                input_shape_2d=tuple(train_cfg["input_shape_2d"]),
                                input_shape_3d=tuple(train_cfg["input_shape_3d"]),
                                num_classes=int(train_cfg["num_classes"]),
                            )
                            make_device_edge_from_full(edge_aggs[e_id].trainer.full_model, device_net, edge_net)

                            # Cache the *raw* inputs on the device for later split-backward
                            if d in device_trainers:
                                try:
                                    device_trainers[d].cache_last_inputs(
                                        beta_name=beta,
                                        inputs={"input_2d": X2b, "input_3d": X3b},
                                    )
                                except Exception as ex:
                                    print(f"[device-cache] warn dev={d} beta={beta}: {ex}")
                                    
                            t0 = time.time()   
                            cut_out = device_net.predict({"input_2d": X2b, "input_3d": X3b}, verbose=0)
                            t_taken = time.time() - t0
                            device_train_time_accum[d] += t_taken
                            round_train_time += t_taken
                            total_train_time += t_taken
                            
                            activ = {"a2d": cut_out[0], "a3d": cut_out[1]} if isinstance(cut_out, (list, tuple)) else {"joint": cut_out}

                            # ---- Send activations WITH labels + batch_id; compression knobs already wired ----
                            acc = uploader.send_activations(
                                device_id=d,
                                edge_id=e_id,
                                beta=beta,
                                activations=activ,
                                batch_id=batch_id,
                                labels=yb,
                            )
                            bytes_uploaded = acc["bytes_payload"]

                            b_idx = allowed.index(beta)
                            e_comp_real = per_dev_beta[d].Ecomp_J[b_idx] / max(1, len(blist))
                            t_comp_real = t_taken / max(1, len(blist))
                            #t_comp_real = per_dev_beta[d].Tcomp_s[b_idx] / max(1, len(blist))
                            t_comm_real = float(bytes_uploaded) / max(1.0, float(st.uplink_rate_Bps))
                            t_extra = (st.Tedge_s + st.Tsync_s) / max(1, len(blist))
                    else:
                        if k_i > 0:
                            acc = uploader.send_rawbatch(
                                device_id=d, edge_id=e_id, arrays={"X2": X2b, "X3": X3b}, labels=yb,
                                batch_id=batch_id
                            )
                            bytes_uploaded = acc["bytes_payload"]
                            t_comm_real = float(bytes_uploaded) / max(1.0, float(st.uplink_rate_Bps))

                    e_comm_real = float(bytes_uploaded) * float(st.comm_cost_per_byte_J)
                    if z_i == 0 and k_i > 0:
                        e_transfer_real = e_comm_real
                        e_comm_real = 0.0

                    e_total_real = float(e_comp_real + e_comm_real + e_transfer_real)
                    t_total_real = float(t_comp_real + t_comm_real + t_extra)

                    energylat_log.add_energy(d, e_total_real)
                    energylat_log.add_latency(d, t_total_real)

                    # Edge RL stats
                    est = edge_states[e_id]
                    est.realized_energy_sum += e_total_real
                    est.realized_latency_sum += t_total_real
                    est.step_count += 1
                    if z_i == 1 and beta and beta != "output":
                        est.q_activations += 1
                    if z_i == 0 and k_i > 0:
                        est.q_raw += 1  # per scheduled batch

        # Edge step + cadence
        for e_id, agg in edge_aggs.items():
            agg.step_round(round_index=t)

        # -------- Device-side gradient application (NEW) --------
        # Drain cut-gradients and apply them on the corresponding device trainer.
        cut_msgs = [m for m in bus.drain("edge_to_device_cutgrads") if isinstance(m, CutGradients)]
        for m in cut_msgs:
            dev_id = getattr(m, "device_id", None)
            beta_m = getattr(m, "beta", None)
            if dev_id is None or beta_m is None:
                continue
            dt = device_trainers.get(dev_id)
            if dt is None:
                continue
            try:
                t0 = time.time()
                dt.apply_cut_gradients(beta_name=beta_m, cut_grads=m.grads)
                t_taken = time.time() - t0
                if dev_id in device_train_time_accum:
                    device_train_time_accum[dev_id] += t_taken
                    round_train_time += t_taken
                    total_train_time += t_taken
            except Exception as ex:
                print(f"[device-apply] warn dev={dev_id} beta={beta_m}: {ex}")

        # ----- Build global obs for cloud RL -----
        means_lat = []
        means_eng = []
        qtot = 0.0
        for _eid, st in edge_states.items():
            n = max(1, st.step_count)
            means_lat.append(st.realized_latency_sum / n if n > 0 else 0.0)
            means_eng.append(st.realized_energy_sum / n if n > 0 else 0.0)
            qtot += float(st.q_activations + st.q_raw)

        global_mean_latency = float(np.mean(means_lat)) if means_lat else 0.0
        global_mean_energy  = float(np.mean(means_eng)) if means_eng else 0.0
        global_queue        = float(qtot)

        cloud_obs = {
            "global_mean_latency": global_mean_latency,
            "global_mean_energy":  global_mean_energy,
            "global_queue":        global_queue,
        }

        # ----- Cloud RL chooses O_e (cadence) -----
        Oe = cloud_rl.select_action(cloud_obs, deterministic=False)
        rounds_since_cloud += 1

        did_cloud_update = False
        if rounds_since_cloud >= int(Oe):
            direct_cloud_apply = bool(train_cfg.get("direct_cloud_apply", False))
            updated = cloud.maybe_aggregate_and_update(round_index=t)
            if updated:
                weights = cloud.get_weights()
                if direct_cloud_apply:
                    # DIRECT APPLY MODE
                    for e_id in edge_models.keys():
                        edge_models[e_id].set_weights(weights)
                    full_model.set_weights(weights)
                    print(f"[round {t}] cloud aggregated (O_e={Oe}) & directly applied to edges/global.")
                else:
                    # MESSAGE-DRIVEN MODE
                    distributor.broadcast_to_edges(weights)
                    print(f"[round {t}] cloud aggregated (O_e={Oe}) & broadcasted.")
                did_cloud_update = True
            rounds_since_cloud = 0

        # ----- Cloud RL: reward & learn -----
        cloud_reward = - (c_alpha * global_mean_latency + c_beta * global_mean_energy + c_gamma * global_queue)
        next_cloud_obs = {
            "global_mean_latency": global_mean_latency,
            "global_mean_energy":  global_mean_energy,
            "global_queue":        global_queue,
        }
        try:
            cloud_rl.learn(cloud_reward, next_cloud_obs, done=False)
        except Exception as ex:
            print(f"[cloud-rl] learn warning: {ex}")

        # RL reward (edge agents)
        for e_id, st in edge_states.items():
            obs = st.obs_vector()
            reward = - (alpha * obs["mean_latency"] + beta_w * obs["mean_energy"] + gamma * obs["queue_depth"])
            edge_aggs[e_id].rl_learn(reward)

        # Eval (cadenced)
        if eval_every and ((t + 1) % eval_every == 0):
            if X2_val is not None and len(y_val or []) > 0:
                if _ext_evaluate_model and _ext_write_round_report:
                    metrics = _ext_evaluate_model(full_model, X2_val, X3_val, y_val, batch_size=eval_batch_size)
                    _ext_write_round_report(eval_report_csv, t + 1, metrics)
                else:
                    metrics = _fallback_evaluate_model_keras(full_model, X2_val, X3_val, y_val, batch_size=eval_batch_size)
                    if metrics:
                        _fallback_write_round_report_csv(eval_report_csv, t + 1, metrics)
                if metrics:
                    print(f"[eval][round {t}] val:", {k: round(float(v),4) for k, v in metrics.items()})

        # Snapshot & battery evolution
        e_snap, l_snap = energylat_log.snapshot()
        print("[train] realized totals so far:",
              "energy_J=", {k: round(v, 4) for k, v in e_snap.items()},
              "latency_s=", {k: round(v, 4) for k, v in l_snap.items()})

        for d in e_rem_state.keys():
            spent = float(sol.energy_J.get(d, 0.0))
            e_rem_state[d] = max(0.0, e_rem_state[d] - spent + harvest_map[d])
        print("[train] updated E_rem (J):", {k: round(v, 4) for k, v in e_rem_state.items()})
        print("[train][timing] per-device (cumulative seconds):")
        for dev_id, sec in device_train_time_accum.items():
            print(f"  {dev_id}: {sec:.6f}s")
        print(f"[train][timing] round {t} measured compute time (sum): {round_train_time:.6f}s  total so far: {total_train_time:.6f}s")    

        # ------------- Checkpoint save (cadenced) -------------
        if ckpt_every > 0 and ((t + 1) % ckpt_every == 0):
            # Logs we want to persist for analysis
            round_logs = {
                "round": int(t + 1),
                "solver": {
                    "status": solsum.get("status"),
                    "obj": float(solsum.get("obj", 0.0)),
                    "active_devices": int(solsum.get("active_devices", 0)),
                    "total_energy_J": float(solsum.get("total_energy_J", 0.0)),
                    "total_upload_bytes": float(solsum.get("total_upload_bytes", 0.0)),
                    "total_latency_s": float(solsum.get("total_latency_s", 0.0)),
                },
                "realized_totals": {
                    "energy_J": {k: float(v) for k, v in e_snap.items()},
                    "latency_s": {k: float(v) for k, v in l_snap.items()},
                },
                
            }

            # Prepare edge_state snapshot (best-effort)
            edge_state_snapshot: dict[str, dict] = {}
            for e_id, st in edge_states.items():
                if hasattr(st, "state_dict"):
                    try:
                        edge_state_snapshot[e_id] = st.state_dict()
                    except Exception:
                        edge_state_snapshot[e_id] = {
                            "agg_period_Ol": getattr(st, "agg_period_Ol", None),
                            "rounds_since_agg": getattr(st, "rounds_since_agg", None),
                        }
                else:
                    edge_state_snapshot[e_id] = {
                        k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in vars(st).items()
                        if not k.startswith("_")
                    }

            # RL agents to persist this round
            rl_agents_for_ckpt = _collect_rl_agents_for_ckpt()

            # Save checkpoint
            ckpt_dir = checkpoint_save_round(
                run_dir=out_dir,
                round_idx=t + 1,
                model=full_model,
                edge_states=edge_state_snapshot,
                solver_warm=sp.warm_start if use_ws else None,
                logs=round_logs,
                rl_agents=rl_agents_for_ckpt,
            )
            print(f"[ckpt] saved round {t+1} -> {ckpt_dir}")

        # Reset round counters on edge states
        for e_id in edge_states.keys():
            edge_states[e_id].reset_round_counters()

    # Save final model
    ensure_dir(train_cfg.get("savedmodel_path", os.path.join(out_dir, "savedmodel")))
    full_model.save(train_cfg.get("savedmodel_path", os.path.join(out_dir, "savedmodel")))
    print("[train] saved final SavedModel:", train_cfg.get("savedmodel_path", os.path.join(out_dir, "savedmodel")))


if __name__ == "__main__":
    main()

