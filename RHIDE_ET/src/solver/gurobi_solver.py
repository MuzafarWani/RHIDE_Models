#Execution_time_constraint solver code
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

from .param_feeds import RoundCoeffs


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class SolverParams:

    lambda_energy: float = 1.0
    gamma_edge: float = 1.0

    eta: float = 0.0
    xi: float = 0.0

    mip_gap: float = 0.01
    time_limit_s: float = 30.0
    threads: int = 0
    warm_start: Optional[Dict[str, Any]] = None


@dataclass
class SolverOutput:
    beta_choice: Dict[str, str]
    z: Dict[str, int]
    k: Dict[str, float]
    obj_value: float
    status: str
    energy_J: Dict[str, float]
    upload_bytes: Dict[str, float]
    latency_s: Dict[str, float]
    batches_used: Dict[str, int]
    reasons: Dict[str, str]


# =========================================================
# Solver
# =========================================================

class RoundMILPSolver:

    EDGE_RECV_COST_PER_BYTE = 1e-9

    def __init__(self, coeffs: RoundCoeffs, params: SolverParams, integer_k: bool = True):

        if gp is None:
            raise ImportError("gurobipy not installed")

        self.c = coeffs
        self.p = params
        self.integer_k = integer_k

    def solve(self) -> SolverOutput:

        devices = list(self.c.statics.keys())
        n_betas = len(self.c.betas)

        print("\n================ SOLVER INPUT DEBUG ================")
        print("Devices:", devices)
        print("Number of betas:", n_betas)
        print("lambda_energy:", self.p.lambda_energy)
        print("gamma_edge:", self.p.gamma_edge)
        print("eta:", self.p.eta)
        print("xi:", self.p.xi)
        print("====================================================\n")

        m = gp.Model("sphere_train_joint_beta")

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

            print(f"\n----- DEVICE {i} PROFILE -----")
            print("Remaining energy:", st.Erem_J)
            print("Memory limit:", st.mem_max_bytes)
            print("Latency limit:", st.Tmax_s)
            print("Batches:", num_batches)

            # Only one beta if z=1
            m.addConstr(gp.quicksum(beta[i][b] for b in range(n_betas)) == z[i])

            # Energy constraint
            m.addConstr(
                gp.quicksum(
                    (bc.Ecomp_J[b] + bc.Ecomm_J[b]) * beta[i][b]
                    for b in range(n_betas)
                )
                + self.c.transfer_energy_J[i] * k[i]
                <= st.Erem_J
            )

            # Memory constraint
            m.addConstr(
                gp.quicksum(
                    bc.M_bytes[b] * beta[i][b]
                    for b in range(n_betas)
                )
                <= st.mem_max_bytes
            )

            # Transfer constraint
            m.addConstr(k[i] <= num_batches * (1 - z[i]))

            # Must either train or transfer
            m.addConstr(z[i] + k[i] <= 1)

            # -------------------------------------------------
            #  NEW: LATENCY CONSTRAINT
            # -------------------------------------------------

            m.addConstr(
                gp.quicksum(
                    (
                        bc.Tcomp_s[b]
                        + bc.Tcomm_s[b]
                        + st.Tedge_s[b]
                        + st.Tsync_s
                    ) * beta[i][b]
                    for b in range(n_betas)
                )
                <= st.Tmax_s,
                name=f"latency[{i}]"
            )

        # -------------------------------------------------
        # Group devices by edge
        # -------------------------------------------------

        devices_by_edge: Dict[str, list[str]] = {}

        for i, st in self.c.statics.items():
            devices_by_edge.setdefault(st.edge_id, []).append(i)

        # -------------------------------------------------
        # Edge RAM constraint
        # -------------------------------------------------

        for e, devs in devices_by_edge.items():

            Mmax_em = None
            if self.c.edge_ram_bytes is not None:
                Mmax_em = self.c.edge_ram_bytes.get(e, None)

            print("\nEDGE", e, "RAM LIMIT:", Mmax_em)

            if Mmax_em is None:
                continue

            m.addConstr(
                gp.quicksum(
                    gp.quicksum(
                        self.c.beta_coeffs[i].U_bytes[b] * beta[i][b]
                        for b in range(n_betas)
                    )
                    + self.c.statics[i].RBS_bytes * k[i]
                    for i in devs
                )
                <= Mmax_em
            )

        # -------------------------------------------------
        # Edge Energy Budget Constraint
        # -------------------------------------------------

        for e, devs in devices_by_edge.items():

            Emax_em = None
            if self.c.edge_energy_budget is not None:
                Emax_em = self.c.edge_energy_budget.get(e, None)

            print("\nEDGE", e, "ENERGY BUDGET:", Emax_em)

            if Emax_em is None:
                continue

            m.addConstr(
                gp.quicksum(
                    gp.quicksum(
                        (
                            (
                                (self.c.beta_coeffs[i].Ecomp_J[-1] +
                                 self.c.beta_coeffs[i].Ecomm_J[-1])
                                -
                                (self.c.beta_coeffs[i].Ecomp_J[b] +
                                 self.c.beta_coeffs[i].Ecomm_J[b])
                            )
                            +
                            self.c.beta_coeffs[i].U_bytes[b] *
                            self.EDGE_RECV_COST_PER_BYTE
                        )
                        * beta[i][b]
                        for b in range(n_betas)
                    )
                    +
                    k[i] * (
                        self.c.beta_coeffs[i].Ecomp_J[-1] +
                        self.c.beta_coeffs[i].Ecomm_J[-1]
                    )
                    for i in devs
                )
                <= Emax_em,
                name=f"edge_energy[{e}]"
            )

        # -------------------------------------------------
        # Objective
        # -------------------------------------------------

        obj = []

        for i in devices:

            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]

            total_model_energy = bc.Ecomp_J[-1] + bc.Ecomm_J[-1]

            obj.append(
                gp.quicksum(
                    (
                        self.p.lambda_energy *
                        (bc.Ecomp_J[b] + bc.Ecomm_J[b])
                        +
                        self.p.gamma_edge *
                        (
                            (total_model_energy - (bc.Ecomp_J[b] + bc.Ecomm_J[b]))
                            +
                            bc.U_bytes[b] * self.EDGE_RECV_COST_PER_BYTE
                        )
                    )
                    * beta[i][b]
                    for b in range(n_betas)
                )
            )

            if self.p.eta:
                obj.append(-self.p.eta * st.priority * z[i])

            if self.p.xi:
                obj.append(-self.p.xi * st.priority * k[i])

        m.setObjective(gp.quicksum(obj), GRB.MINIMIZE)

        print("\n========= STARTING SOLVER =========\n")

        m.optimize()

        # -------------------------------------------------
        # Extract results
        # -------------------------------------------------

        beta_choice, z_sol, k_sol = {}, {}, {}
        energy, upload, latency = {}, {}, {}
        batches_used = {}
        reasons = {}

        status = m.Status

        status_str = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
        }.get(status, f"STATUS_{status}")

        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):

            for i in devices:

                st = self.c.statics[i]
                bc = self.c.beta_coeffs[i]

                z_i = int(round(z[i].X))
                k_i = int(round(k[i].X))

                chosen_b = None

                for b in range(n_betas):
                    if beta[i][b].X > 0.5:
                        chosen_b = b
                        break

                z_sol[i] = z_i
                k_sol[i] = k_i

                e = u = t = 0.0

                if z_i == 1 and chosen_b is not None:

                    beta_name = self.c.betas[chosen_b].name
                    beta_choice[i] = beta_name

                    e = bc.Ecomp_J[chosen_b] + bc.Ecomm_J[chosen_b]
                    u = bc.U_bytes[chosen_b]

                    t = (
                        bc.Tcomp_s[chosen_b]
                        + bc.Tcomm_s[chosen_b]
                        + st.Tedge_s[chosen_b]
                        + st.Tsync_s
                    )

                    batches_used[i] = self.c.batches_per_device.get(i, 0)
                    reasons[i] = "TRAIN"

                else:

                    beta_choice[i] = ""

                    e = self.c.transfer_energy_J[i] * k_i
                    u = st.RBS_bytes * k_i
                    t = self.c.transfer_time_s[i] * k_i

                    batches_used[i] = k_i
                    reasons[i] = "TRANSFER"

                energy[i] = e
                upload[i] = u
                latency[i] = t

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
            reasons=reasons,
        )
