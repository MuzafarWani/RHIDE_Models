
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run all generated XEventNet scenarios and log outputs to per-scenario text files.

Example:
  python run_all_scenarios_xeventnet.py --scenarios-dir outxevent --configs-dir configs_xeventnet --rounds 10 --extra "--seed 1"
"""

import argparse, csv, os, shlex, subprocess, sys, time
from datetime import datetime
from pathlib import Path


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scenarios-dir", required=True,
                    help="Directory containing scenario subfolders (out/*)")

    ap.add_argument("--configs-dir", default="configs",
                    help="XEventNet configs directory")

    ap.add_argument("--train-script",
                    default="scripts/train_federated_xeventnet.py",
                    help="XEventNet training script")

    ap.add_argument("--rounds", type=int, default=5)

    ap.add_argument("--python", default=sys.executable)

    ap.add_argument("--extra", default="",
                    help='Extra args as one string, e.g. "--seed 1"')

    ap.add_argument("--logs-dir", default="logs")

    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


# ------------------------------------------------------------
# Scenario discovery
# ------------------------------------------------------------
def list_scenarios(root: Path):
    for p in sorted(root.iterdir()):
        if (
            p.is_dir()
            and (p / "federated.yaml").is_file()
            and (p / "energy_latency.yaml").is_file()
        ):
            yield p.name, p


# ------------------------------------------------------------
# Command builder (XEventNet)
# ------------------------------------------------------------
def build_cmd(py, train_script, cfg_dir: Path, scen_dir: Path,
              rounds: int, extra: str):

    cmd = [
        py, str(train_script),

        "--data_cfg",   str(cfg_dir / "data.yaml"),   # ✅ REQUIRED
        "--train_cfg",  str(cfg_dir / "train.yaml"),
        "--fed_cfg",    str(scen_dir / "federated.yaml"),
        "--el_cfg",     str(scen_dir / "energy_latency.yaml"),
        "--split_cfg",  str(cfg_dir / "split.yaml"),
        "--solver_cfg", str(cfg_dir / "solver.yaml"),
        "--rl_cfg",     str(cfg_dir / "rl.yaml"),

        "--rounds", str(rounds),
    ]

    extra = extra.strip()
    if extra:
        cmd.extend(shlex.split(extra))

    return cmd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    scen_root   = Path(args.scenarios_dir).resolve()
    cfg_dir     = Path(args.configs_dir).resolve()
    train_script = Path(args.train_script).resolve()

    project_root = train_script.parent.parent.resolve()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_root = ensure_dir(Path(args.logs_dir) / stamp)
    manifest_path = logs_root / "manifest.csv"

    with manifest_path.open("w", newline="") as mf:

        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "scenario",
                "return_code",
                "seconds",
                "log_path",
                "command",
            ],
        )
        writer.writeheader()

        any_found = False

        for scen_name, scen_dir in list_scenarios(scen_root):

            any_found = True

            cmd = build_cmd(
                args.python,
                train_script,
                cfg_dir,
                scen_dir,
                args.rounds,
                args.extra,
            )

            log_path = logs_root / f"{scen_name}.log"

            # Match your PYTHONPATH setup
            env = os.environ.copy()
            prev = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{project_root}:{project_root / 'src'}"
                + (f":{prev}" if prev else "")
            )

            print(f"\n[RUN] {scen_name}")
            print("CMD:", " ".join(shlex.quote(x) for x in cmd))
            print("LOG:", log_path)
            print("CWD:", project_root)
            print("PYTHONPATH:", env["PYTHONPATH"])

            if args.dry_run:
                writer.writerow({
                    "scenario": scen_name,
                    "return_code": "DRY_RUN",
                    "seconds": 0.0,
                    "log_path": str(log_path),
                    "command": " ".join(shlex.quote(x) for x in cmd),
                })
                continue

            t0 = time.time()

            with log_path.open("w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=str(project_root),
                    env=env,
                )

                for line in proc.stdout:
                    sys.stdout.write(line)
                    lf.write(line)

                proc.wait()
                rc = proc.returncode

            dt = time.time() - t0

            writer.writerow({
                "scenario": scen_name,
                "return_code": rc,
                "seconds": round(dt, 3),
                "log_path": str(log_path),
                "command": " ".join(shlex.quote(x) for x in cmd),
            })

        if not any_found:
            print(f"[WARN] No scenarios found in {scen_root}")

    print(f"\n[OK] Finished XEventNet runs. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

