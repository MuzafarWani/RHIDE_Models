#multi_round optimization solver code
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

# ===================== DEBUG FLAG =====================
DEBUG_SOLVER = True
# =====================================================


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class SolverParams:

    lambda_energy: float = 1.0
    gamma_edge: float = 1.0
    lambda_comm: float = 0.0
    lambda_latency: float = 0.0

    eta: float = 0.0
    xi: float = 0.0

    alpha_future: float = 0.0
    remaining_rounds: int = 1

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

        # ================= DEBUG: ROUND INPUT =================
        if DEBUG_SOLVER:
            print("\n================ DEBUG: ROUND INPUT =================")
            for i in devices:
                st = self.c.statics[i]
                bc = self.c.beta_coeffs[i]

                Erem = max(1e-6, st.Erem_J)
                E_allowed = max(1e-6, st.Erem_J / max(1, self.p.remaining_rounds))

                print(f"[Device {i}] Erem={Erem:.4f}, E_allowed={E_allowed:.4f}, "
                      f"alpha/E_allowed={self.p.alpha_future / max(1e-6, E_allowed):.4f}")
            print("====================================================\n")

        m = gp.Model("sphere_train_joint_beta")

        if self.p.threads:
            m.Params.Threads = self.p.threads

        m.Params.MIPGap = self.p.mip_gap

        if self.p.time_limit_s > 0:
            m.Params.TimeLimit = self.p.time_limit_s
        m.Params.NumericFocus = 3    

        # -------------------------------------------------
        # Variables
        # -------------------------------------------------

        z, k, beta, penalty_var = {}, {}, {}, {}

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
            penalty_var[i] = {}

            for b in range(n_betas):
                beta[i][b] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"beta[{i},{self.c.betas[b].name}]"
                )

                #Penality Variable
                penalty_var[i][b] = m.addVar(lb=0.0, name=f"penalty[{i},{b}]")

        m.update()

        # -------------------------------------------------
        # Device-level constraints
        # -------------------------------------------------

        for i in devices:

            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]

            if DEBUG_SOLVER:
                print(f"[Device Constraint][{i}] Erem={st.Erem_J:.4f}, mem={st.mem_max_bytes}")

            m.addConstr(gp.quicksum(beta[i][b] for b in range(n_betas)) == z[i])

            m.addConstr(
                gp.quicksum(
                    (bc.Ecomp_J[b] + bc.Ecomm_J[b]) * beta[i][b]
                    for b in range(n_betas)
                )
                + self.c.transfer_energy_J[i] * k[i]
                <= st.Erem_J
            )

            m.addConstr(
                gp.quicksum(
                    bc.M_bytes[b] * beta[i][b]
                    for b in range(n_betas)
                )
                <= st.mem_max_bytes
            )

            m.addConstr(k[i] <= int(self.c.batches_per_device.get(i, 0)) * (1 - z[i]))
            m.addConstr(z[i] + k[i] >= 1)

        # -------------------------------------------------
        #  NEW: Penalty constraints
        # -------------------------------------------------

        for i in devices:
            bc = self.c.beta_coeffs[i]
            st = self.c.statics[i]

            Erem = max(1e-6, st.Erem_J)
            E_allowed = max(1e-6, st.Erem_J / max(1, self.p.remaining_rounds))

            for b in range(n_betas):

                ratio_expr = (bc.Ecomp_J[b] + bc.Ecomm_J[b]) / max(1e-6, E_allowed)

                m.addConstr(
                    penalty_var[i][b] >= ratio_expr - 1,
                    name=f"penalty_lb[{i},{b}]"
                )

        # -------------------------------------------------
        # Edge Memory constraint
        # -------------------------------------------------

        if self.c.edge_ram_bytes is not None:

            if DEBUG_SOLVER:
                print("\n================ DEBUG: EDGE RAM =================")

            for e, devs in self.c.devices_by_edge.items():

                Mmax_em = self.c.edge_ram_bytes.get(e, None)

                if Mmax_em is None:
                    continue

                if DEBUG_SOLVER:
                    print(f"[Edge {e}] RAM budget={Mmax_em}")

                m.addConstr(
                    gp.quicksum(
                        gp.quicksum(
                            self.c.beta_coeffs[i].U_bytes[b] * beta[i][b] * z[i]
                            for b in range(n_betas)
                        )
                        + self.c.statics[i].RBS_bytes * k[i] * (1 - z[i])
                        for i in devs
                    )
                    <= Mmax_em,
                    name=f"edge_ram[{e}]"
                )

        # -------------------------------------------------
        # Edge Energy constraint
        # -------------------------------------------------

        if self.c.edge_energy_budget is not None:

            if DEBUG_SOLVER:
                print("\n================ DEBUG: EDGE ENERGY =================")

            for e, devs in self.c.devices_by_edge.items():

                Emax_em = self.c.edge_energy_budget.get(e, None)

                if Emax_em is None:
                    continue

                if DEBUG_SOLVER:
                    print(f"[Edge {e}] Energy budget={Emax_em}")

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

        if DEBUG_SOLVER:
            print("\n================ DEBUG: OBJECTIVE =================")

        for i in devices:

            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]

            total_model_energy = bc.Ecomp_J[-1] + bc.Ecomm_J[-1]
            Erem = max(1e-6, st.Erem_J)
            E_allowed = max(1e-6, st.Erem_J / max(1, self.p.remaining_rounds))

            for b in range(n_betas):

                if DEBUG_SOLVER:
                    E_device = bc.Ecomp_J[b] + bc.Ecomm_J[b]
                    norm = E_device / total_model_energy
                    #ratio = ((bc.Ecomp_J[b] + bc.Ecomm_J[b]) - E_allowed) / max(1e-6, E_allowed)
                    #penalty = max(0.0, ratio - 1.0)
                    #penalty = max(0.0, (bc.Ecomp_J[b] + bc.Ecomm_J[b]) - E_allowed)
                    #energy_term = self.p.lambda_energy * norm + self.p.alpha_future * penalty
                    violation_ratio = E_device / max(1e-6, E_allowed)
                    violation_excess = max(0.0, violation_ratio - 1.0)
                    penalty = max(0.0, (E_device - E_allowed) / E_allowed)
                    penalty = min(penalty, 2.0)
                    energy_term = (self.p.lambda_energy * norm + self.p.alpha_future * penalty)
                    #energy_term = self.p.lambda_energy * norm * (1 + self.p.alpha_future * violation_excess)
                    #print(f"[Device {i} | Beta {b}] energy_term={energy_term:.4f}, ratio={ratio:.4f}, penalty={penalty:.4f}")
                    print(f"[Device {i} | Beta {b}] energy_term={energy_term:.4f}, "f"norm={norm:.4f}, violation_ratio={violation_ratio:.4f}")

            obj.append(
                gp.quicksum(
                    (
                        self.p.lambda_energy * ((bc.Ecomp_J[b] + bc.Ecomm_J[b]) / total_model_energy)
                        +
                        #self.p.alpha_future * penalty_var[i][b]
                        (
                        self.p.lambda_energy * self.p.alpha_future * ((bc.Ecomp_J[b] + bc.Ecomm_J[b]) / total_model_energy) * penalty_var[i][b] 
                        )
                        +
                        self.p.gamma_edge *
                        ((total_model_energy - (bc.Ecomp_J[b] + bc.Ecomm_J[b])) + bc.U_bytes[b] * self.EDGE_RECV_COST_PER_BYTE)
                        +
                        self.p.lambda_comm * bc.U_bytes[b]
                    )
                    * beta[i][b]
                    for b in range(n_betas)
                )
            )

            if self.p.lambda_comm:
                obj.append(self.p.lambda_comm * st.RBS_bytes * k[i])
             
            if self.p.eta:
                obj.append(-self.p.eta * st.priority * z[i])

            if self.p.xi:
                obj.append(-self.p.xi * st.priority * k[i])

        m.setObjective(gp.quicksum(obj), GRB.MINIMIZE)

        m.optimize()

        # -------------------------------------------------
        # Debug solution
        # -------------------------------------------------

        if DEBUG_SOLVER:
            print("\n================ DEBUG: SOLUTION =================")

        for i in devices:

            if m.SolCount > 0:
                z_i = int(round(z[i].X))
                k_i = int(round(k[i].X))
            else:
                print(f"[WARNING] No solution available for device {i}")
                z_i = 0
                k_i = 0    

            if DEBUG_SOLVER:
                print(f"[Device {i}] z={z_i}, k={k_i}")

                if z_i == 0:
                    print("  ❌ DROPPED (energy too high OR reward too low)")

        if DEBUG_SOLVER:
            print("================================================\n")


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

                    beta_choice[i] = self.c.betas[chosen_b].name

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
