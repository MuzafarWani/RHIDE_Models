# src/solver/gurobi_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math
import os

# Gurobi import check
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

from .param_feeds import RoundCoeffs


# ==========================
# Dataclasses
# ==========================

@dataclass
class SolverParams:
    """
    Hyperparameters for the MILP solver.

    eta : reward weight for activating a device (z_i = 1)
    xi  : reward weight for transferring raw batches (k_i)
    mip_gap : acceptable optimality gap
    time_limit_s : solver time limit
    threads : number of threads for Gurobi
    warm_start : optional initial solution
    """
    eta: float = 0.0
    xi: float = 0.0
    mip_gap: float = 0.01
    time_limit_s: float = 30.0
    threads: int = 0
    warm_start: Optional[Dict[str, Any]] = None


@dataclass
class SolverOutput:
    """
    Structured result of one MILP solve.
    """
    beta_choice: Dict[str, str]     # device -> chosen β name ("" if none)
    z: Dict[str, int]               # z_i ∈ {0,1} : participation
    k: Dict[str, float]             # k_i ∈ [0, B_i] : raw batches sent
    obj_value: float
    status: str
    energy_J: Dict[str, float]
    upload_bytes: Dict[str, float]
    latency_s: Dict[str, float]


# ==========================
# Debug helpers
# ==========================

def _debug_enabled() -> bool:
    """Enable debug prints unless MILP_DEBUG=0."""
    return os.environ.get("MILP_DEBUG", "1") not in ("0", "false", "False", "")


# ==========================
# β SELECTION (OUTSIDE MILP)
# ==========================

def select_beta_min_energy(coeffs: RoundCoeffs) -> Dict[str, Optional[int]]:
    """
    For each device i:
      - Enumerate all β in coeffs.betas
      - Check feasibility under:
            Energy, Remaining Energy, Latency, Memory
      - Pick the β with minimum energy
      - Return None if no β is feasible

    Output:
        device_id -> index of chosen β (or None)
    """
    beta_sel: Dict[str, Optional[int]] = {}

    # Loop over devices (i ∈ devices)
    #This loop is iterating by device name, like d0, d1 ....
    for dev, st in coeffs.statics.items():
        bc = coeffs.beta_coeffs[dev] # fetches the BetaCoeffs object for that same device which has which has Ecomp_J, Ecomm_J, U_bytes, Tcomp_s, Tcomm_s, M_bytes

        best_idx = None
        best_energy = math.inf

        # Loop over all split points β
        for b_idx, _ in enumerate(coeffs.betas):
            # Per-round energy if β is used
            E = bc.Ecomp_J[b_idx] + bc.Ecomm_J[b_idx]

            # Per-round latency if β is used
            T = (
                bc.Tcomp_s[b_idx]
                + bc.Tcomm_s[b_idx]
                + st.Tedge_s[b_idx]
                + st.Tsync_s
            )

            # Peak memory on device
            M = bc.M_bytes[b_idx]

            # Feasibility checks (AND of all constraints)
            if (
              #  E <= st.Emax_round_J and
                E <= st.Erem_J and
                T <= st.Tmax_s and
                M <= st.mem_max_bytes
            ):
                if E < best_energy:
                    best_energy = E
                    best_idx = b_idx

        beta_sel[dev] = best_idx

        if _debug_enabled():
            if best_idx is None:
                print(f"[beta-select][{dev}] NO feasible β → raw transfer only")
            else:
                print(
                    f"[beta-select][{dev}] chosen β={coeffs.betas[best_idx].name} "
                    f"E={best_energy:.6f}J"
                )

    return beta_sel


# ==========================
# TRAIN-TIME MILP (z, k ONLY)
# ==========================

class RoundMILPSolver:
    """
    Train-time MILP with FIXED β.

    Decision variables:
        z_i ∈ {0,1}   : device participation
        k_i ∈ [0, B_i]: number of raw batches sent upstream

    Semantics:
        z_i = 1  → device trains using split β
        z_i = 0  → device does not train

        k_i = number of raw batches sent (only allowed if z_i = 0)

    Core exclusivity rule:
        k_i ≤ B_i * (1 - z_i)

        If z_i = 1 → RHS = 0 → k_i = 0
        If z_i = 0 → RHS = B_i → 0 ≤ k_i ≤ B_i
    """

    def __init__(self, coeffs: RoundCoeffs, params: SolverParams, integer_k: bool = True):
        if gp is None:
            raise ImportError("gurobipy not installed")
        self.c = coeffs
        self.p = params
        self.integer_k = integer_k

    def solve(self) -> SolverOutput:
        # Device set: e.g., ["d0", "d1", "d2"]
        devices = list(self.c.statics.keys())

        # ---- FIXED β selection (outside MILP)
        beta_idx = select_beta_min_energy(self.c)

        # Create MILP model
        m = gp.Model("iot_weather_train_fixed_beta")

        # Solver configuration
        if self.p.threads:
            m.Params.Threads = self.p.threads
        m.Params.MIPGap = self.p.mip_gap
        if self.p.time_limit_s > 0:
            m.Params.TimeLimit = self.p.time_limit_s

        # Decision variables
        z, k = {}, {}

        # For each device i
        for i in devices:
            Bi = int(self.c.batches_per_device.get(i, 0))  # B_i

            # z_i ∈ {0,1}
            z[i] = m.addVar(vtype=GRB.BINARY, name=f"z[{i}]")

            # k_i ∈ [0, B_i]
            k[i] = m.addVar(
                vtype=GRB.INTEGER if self.integer_k else GRB.CONTINUOUS,
                lb=0.0,
                ub=float(Bi),
                name=f"k[{i}]"
            )

        m.update()

        # ---- Constraints
        for i in devices:
            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]
            Bi = int(self.c.batches_per_device.get(i, 0))
            b = beta_idx[i]

            # If no feasible β, all β-dependent terms collapse to 0
            E_beta = 0.0 if b is None else (bc.Ecomp_J[b] + bc.Ecomm_J[b])
            T_beta = 0.0 if b is None else (
                bc.Tcomp_s[b]
                + bc.Tcomm_s[b]
                + st.Tedge_s[b]
                + st.Tsync_s
            )
            M_beta = 0.0 if b is None else bc.M_bytes[b]

            # Energy constraints (round budget & battery)
           #m.addConstr(E_beta * z[i] + self.c.transfer_energy_J[i] * k[i] <= st.Emax_round_J)
            m.addConstr(E_beta * z[i] + self.c.transfer_energy_J[i] * k[i] <= st.Erem_J)

            # Latency constraint
            m.addConstr(T_beta * z[i] + self.c.transfer_time_s[i] * k[i] <= st.Tmax_s)

            # Memory constraint (only applies when z_i = 1)
            m.addConstr(M_beta * z[i] <= st.mem_max_bytes)

            # Exclusivity: train OR transfer
            m.addConstr(k[i] <= Bi * (1 - z[i]))

            # If β is feasible and data exists → force training
            if b is not None and Bi > 0:
                m.addConstr(z[i] == 1)

        # ---- Objective
        obj = []
        for i in devices:
            st = self.c.statics[i]

            # Cost for transferring raw data
           # obj.append(self.c.transfer_energy_J[i] * k[i])

            # Reward for participation
            if self.p.eta:
                obj.append(-self.p.eta * st.priority * z[i])

            # Optional reward for transfer
            if self.p.xi:
                obj.append(-self.p.xi * st.priority * k[i])

        m.setObjective(gp.quicksum(obj), GRB.MINIMIZE)
        m.optimize()

        # ---- Status
        status = m.Status
        status_str = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
        }.get(status, f"STATUS_{status}")

        # ---- Extract results
        beta_choice, z_sol, k_sol = {}, {}, {}
        energy, upload, latency = {}, {}, {}

        for i in devices:
            z_i = int(round(z[i].X))
            k_i = float(k[i].X)

            b = beta_idx[i]
            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]

            e = u = t = 0.0
            if z_i == 1 and b is not None:
                e += bc.Ecomp_J[b] + bc.Ecomm_J[b]
                u += bc.U_bytes[b]
                t += (
                    bc.Tcomp_s[b]
                    + bc.Tcomm_s[b]
                    + st.Tedge_s[b]
                    + st.Tsync_s
                )
                beta_choice[i] = self.c.betas[b].name
            else:
                beta_choice[i] = ""

            # Raw transfer contributions
            e += self.c.transfer_energy_J[i] * k_i
            u += st.RBS_bytes * k_i
            t += self.c.transfer_time_s[i] * k_i

            z_sol[i] = z_i
            k_sol[i] = k_i
            energy[i] = e
            upload[i] = u
            latency[i] = t

        return SolverOutput(
            beta_choice=beta_choice,
            z=z_sol,
            k=k_sol,
            obj_value=float(m.ObjVal) if m.SolCount else math.inf,
            status=status_str,
            energy_J=energy,
            upload_bytes=upload,
            latency_s=latency,
        )

