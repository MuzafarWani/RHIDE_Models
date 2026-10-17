from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

# ---------------------------------------------------------
# Gurobi import (check)
# ---------------------------------------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

# RoundCoeffs bundles *all* numeric parameters for one round:
#  - per-device static limits (energy, latency, memory, etc.)
#  - per-device, per-β coefficients (energy, time, upload, memory)
#  - number of batches B_i for each device
#  - transfer costs for raw data
#  - list of available split points β
from .param_feeds import RoundCoeffs


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class SolverParams:
    """
    Hyperparameters controlling the MILP behavior.

    eta : reward for training participation
          Appears as: -eta * priority_i * z_i

    xi  : reward for offloading raw batches
          Appears as: -xi * priority_i * k_i

    mip_gap : acceptable optimality gap (e.g., 0.01 = 1%)

    time_limit_s : solver time limit in seconds

    threads : number of Gurobi threads

    warm_start : optional initial solution (unused here)
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
    Structured output of one MILP solve.
    """
    beta_choice: Dict[str, str]     # device -> chosen β name ("" if none)
    z: Dict[str, int]               # z_i ∈ {0,1} (train or not)
    k: Dict[str, float]             # k_i ∈ [0, B_i] (raw batches offloaded)
    obj_value: float
    status: str
    energy_J: Dict[str, float]
    upload_bytes: Dict[str, float]
    latency_s: Dict[str, float]
    batches_used: Dict[str, int]
    reasons: Dict[str, str]   ## Tag 1


class RoundMILPSolver:
    """
    SPHERE semantics with shared edge capacity:

    - A device may train only if at least one β is locally feasible.
    - If no β is feasible, it may only offload raw batches (k_i >= 1),
      provided raw transfer fits energy and latency.
    - The solver is free to *drop* otherwise-feasible devices when
      edge-level capacity (Umax) is exceeded.
    """

    def __init__(self, coeffs: RoundCoeffs, params: SolverParams, integer_k: bool = True):
        if gp is None:
            raise ImportError("gurobipy not installed")

        # All numeric inputs for this round
        self.c = coeffs

        # Solver hyperparameters (eta, xi, etc.)
        self.p = params

        # Whether k_i is integer (batches) or continuous
        self.integer_k = integer_k

    def solve(self) -> SolverOutput:
        # -------------------------------------------------
        # Index sets
        # -------------------------------------------------
        devices = list(self.c.statics.keys())
        n_betas = len(self.c.betas)

        # Create MILP model
        m = gp.Model("sphere_train_joint_beta")

        # Solver configuration
        if self.p.threads:
            m.Params.Threads = self.p.threads
        m.Params.MIPGap = self.p.mip_gap
        if self.p.time_limit_s > 0:
            m.Params.TimeLimit = self.p.time_limit_s

        # -------------------------------------------------
        # Decision variables
        # -------------------------------------------------
        z, k, beta = {}, {}, {}

        for i in devices:
            num_batches = int(self.c.batches_per_device.get(i, 0))

            z[i] = m.addVar(vtype=GRB.BINARY, name=f"z[{i}]")

            k[i] = m.addVar(
                vtype=GRB.INTEGER if self.integer_k else GRB.CONTINUOUS,
                lb=0.0,
                ub=float(num_batches),
                name=f"k[{i}]"
            )

            beta[i] = {}
            for b in range(n_betas):
                beta[i][b] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"beta[{i},{self.c.betas[b].name}]"
                )

        m.update()

        # -------------------------------------------------
        # Per-device constraints
        # -------------------------------------------------
        for i in devices:
            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]
            num_batches = int(self.c.batches_per_device.get(i, 0))

            # Exactly one β if active, none if inactive:
            m.addConstr(gp.quicksum(beta[i][b] for b in range(n_betas)) == z[i])

            # Energy budget:
            m.addConstr(
                gp.quicksum(
                    (bc.Ecomp_J[b] + bc.Ecomm_J[b]) * beta[i][b] * z[i]
                    for b in range(n_betas)
                )
                + self.c.transfer_energy_J[i] * k[i] * (1-z[i])
                <= st.Erem_J
            )

            # Device memory constraint:
            m.addConstr(
                gp.quicksum(
                    bc.M_bytes[b] * beta[i][b] * z[i]
                    for b in range(n_betas)
                )
                <= st.mem_max_bytes
            )

            # Mutual exclusion:
            m.addConstr(k[i] <= num_batches * (1 - z[i]))

            # Feasibility checking:
            feasible = False
            for b in range(n_betas):
                e = bc.Ecomp_J[b] + bc.Ecomm_J[b]
                t = bc.Tcomp_s[b] + bc.Tcomm_s[b] + st.Tedge_s[b] + st.Tsync_s
                mreq = bc.M_bytes[b]
                if e <= st.Erem_J and t <= st.Tmax_s and mreq <= st.mem_max_bytes:
                    feasible = True
                    break

            if not feasible:
                can_transfer = (
                    num_batches > 0
                    and self.c.transfer_energy_J[i] <= st.Erem_J
                    and self.c.transfer_time_s[i] <= st.Tmax_s
                )
                if can_transfer:
                    m.addConstr(z[i] == 0)
                    m.addConstr(k[i] >= 1)
                else:
                    m.addConstr(z[i] == 0)
                    m.addConstr(k[i] == 0)

        # -------------------------------------------------
        # Edge-level aggregate uplink / memory constraint
        # -------------------------------------------------
        devices_by_edge: Dict[str, list[str]] = {}
        for i, st in self.c.statics.items():
            devices_by_edge.setdefault(st.edge_id, []).append(i)

        # For each edge e:
        #   sum_i (activation bytes if training + raw bytes if offloading)
        #   <= Umax_e
        for e, devs in devices_by_edge.items():
            # >>> NEW: use edge RAM from federated.yaml via RoundCoeffs
            Umax_e = self.c.edge_ram_bytes[e]

            m.addConstr(
                gp.quicksum(
                    gp.quicksum(
                        self.c.beta_coeffs[i].U_bytes[b] * beta[i][b] * z[i]
                        for b in range(n_betas)
                    )
                    + self.c.statics[i].RBS_bytes * k[i] * (1-z[i])
                    for i in devs
                )
                <= Umax_e,
                name=f"edge_uplink[{e}]"
            )

        # -------------------------------------------------
        # Objective
        # -------------------------------------------------
        obj = []
        for i in devices:
            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]

            obj.append(
                gp.quicksum(
                    (bc.Ecomp_J[b] + bc.Ecomm_J[b]) * beta[i][b]
                    for b in range(n_betas)
                )
            )

            if self.p.eta:
                obj.append(-self.p.eta * st.priority * z[i])

            if self.p.xi:
                obj.append(-self.p.xi * st.priority * k[i])

        m.setObjective(gp.quicksum(obj), GRB.MINIMIZE)
        m.optimize()

        # -------------------------------------------------
        # Extract results
        # -------------------------------------------------
        beta_choice, z_sol, k_sol = {}, {}, {}
        energy, upload, latency = {}, {}, {}
        batches_used = {}
        reasons = {}  ## Tag 1

        status = m.Status
        status_str = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
        }.get(status, f"STATUS_{status}")

        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            for i in devices:
                z_i = int(round(z[i].X))
                k_i = int(round(k[i].X))

                st = self.c.statics[i]
                bc = self.c.beta_coeffs[i]
                num_batches = int(self.c.batches_per_device.get(i, 0)) ## Tag 1

                chosen_b = None
                for b in range(n_betas):
                    if beta[i][b].X > 0.5:
                        chosen_b = b
                        break

                e = u = t = 0.0
                if z_i == 1 and chosen_b is not None:
                    e += bc.Ecomp_J[chosen_b] + bc.Ecomm_J[chosen_b]
                    u += bc.U_bytes[chosen_b]
                    t += (
                        bc.Tcomp_s[chosen_b]
                        + bc.Tcomm_s[chosen_b]
                        + st.Tedge_s[chosen_b]
                        + st.Tsync_s
                    )
                    beta_choice[i] = self.c.betas[chosen_b].name
                    batches_used[i] = self.c.batches_per_device.get(i, 0)
                    reasons[i] = "TRAIN"   ## Tag 1
                else:
                    beta_choice[i] = ""
                    batches_used[i] = k_i

                e += self.c.transfer_energy_J[i] * k_i
                u += st.RBS_bytes * k_i
                t += self.c.transfer_time_s[i] * k_i

                z_sol[i] = z_i
                k_sol[i] = k_i
                energy[i] = e
                upload[i] = u
                latency[i] = t

                ## Tag 1 block
                if z_i == 0:
                    locally_feasible = False
                    min_beta_energy = float("inf")
                    min_beta_upload = float("inf")

                    for b in range(n_betas):
                        e_b = bc.Ecomp_J[b] + bc.Ecomm_J[b]
                        t_b = bc.Tcomp_s[b] + bc.Tcomm_s[b] + st.Tedge_s[b] + st.Tsync_s
                        m_b = bc.M_bytes[b]
                        if e_b <= st.Erem_J and t_b <= st.Tmax_s and m_b <= st.mem_max_bytes:
                            locally_feasible = True
                            min_beta_energy = min(min_beta_energy, e_b)
                            min_beta_upload = min(min_beta_upload, bc.U_bytes[b])

                    if not locally_feasible:
                        reasons[i] = "NO_FEASIBLE_SPLIT (energy/latency/memory)"
                    else:
                        edge = st.edge_id
                        used = 0.0
                        for j in devices:
                            if self.c.statics[j].edge_id == edge:
                                for b in range(n_betas):
                                    used += self.c.beta_coeffs[j].U_bytes[b] * beta[j][b].X
                                used += self.c.statics[j].RBS_bytes * k[j].X

                        # >>> NEW: use edge RAM here too
                        Umax = self.c.edge_ram_bytes[edge]

                        if used + min_beta_upload > Umax + 1e-6:
                            reasons[i] = "DROPPED_BY_EDGE_CAPACITY"
                        else:
                            reasons[i] = "DROPPED_BY_OBJECTIVE (energy vs reward)"

        return SolverOutput(
            beta_choice=beta_choice,
            z=z_sol,
            k=k_sol,
            obj_value=float(m.ObjVal) if m.SolCount else float("inf"),
            status=status_str,
            energy_J=energy,
            upload_bytes=upload,
            latency_s=latency,
            batches_used=batches_used,
            reasons=reasons,    ## Tag 1
        )

