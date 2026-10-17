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
    eta: float = 0.0
    xi: float = 0.0
    mip_gap: float = 0.01
    time_limit_s: float = 30.0
    threads: int = 0
    warm_start: Optional[Dict[str, Any]] = None


@dataclass
class SolverOutput:
    layer_num: Dict[str, int]
    z: Dict[str, int]
    k: Dict[str, int]
    obj_value: float
    status: str


class RoundMILPSolver:
    def __init__(self, coeffs: RoundCoeffs, params: SolverParams, integer_k: bool = True):
        if gp is None:
            raise ImportError("gurobipy not installed")

        self.c = coeffs
        self.p = params
        self.integer_k = integer_k

    def solve(self) -> SolverOutput:
        devices = list(self.c.statics.keys())
        n_layers = len(self.c.betas)

        m = gp.Model("sphere_layernum_milp")

        m.Params.MIPGap = self.p.mip_gap
        if self.p.time_limit_s > 0:
            m.Params.TimeLimit = self.p.time_limit_s
        if self.p.threads:
            m.Params.Threads = self.p.threads

        # -------------------------------------------------
        # Decision variables
        # -------------------------------------------------
        z = {}
        k = {}
        layer_num = {}
        y = {}   # selector binaries

        for i in devices:
            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]
            Bi = int(self.c.batches_per_device.get(i, 0))

            z[i] = m.addVar(vtype=GRB.BINARY, name=f"z[{i}]")

            k[i] = m.addVar(
                vtype=GRB.INTEGER if self.integer_k else GRB.CONTINUOUS,
                lb=0,
                ub=Bi,
                name=f"k[{i}]"
            )

            layer_num[i] = m.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=n_layers - 1,
                name=f"layer_num[{i}]"
            )

            y[i] = {}
            for b in range(n_layers):
                y[i][b] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"y[{i},{b}]"
                )

        m.update()

        # -------------------------------------------------
        # Per-device constraints
        # -------------------------------------------------
        for i in devices:
            st = self.c.statics[i]
            bc = self.c.beta_coeffs[i]
            Bi = int(self.c.batches_per_device.get(i, 0))

            # Select exactly one layer if active
            m.addConstr(gp.quicksum(y[i][b] for b in range(n_layers)) == z[i])

            # Define layer_num from selectors
            m.addConstr(
                layer_num[i] == gp.quicksum(b * y[i][b] for b in range(n_layers))
            )

            # Energy constraint
            m.addConstr(
                gp.quicksum(
                    (bc.Ecomp_J[b] + bc.Ecomm_J[b]) * y[i][b]
                    for b in range(n_layers)
                )
                + self.c.transfer_energy_J[i] * k[i]
                <= st.Erem_J
            )

            # Memory constraint
            m.addConstr(
                gp.quicksum(
                    bc.M_bytes[b] * y[i][b]
                    for b in range(n_layers)
                )
                <= st.mem_max_bytes
            )

            # Mutual exclusion
            m.addConstr(k[i] <= Bi * (1 - z[i]))

        # -------------------------------------------------
        # Edge-level uplink constraint
        # -------------------------------------------------
        devices_by_edge: Dict[str, list[str]] = {}
        for i, st in self.c.statics.items():
            devices_by_edge.setdefault(st.edge_id, []).append(i)

        for e, devs in devices_by_edge.items():
            Umax_e = self.c.statics[devs[0]].Umax_bytes

            m.addConstr(
                gp.quicksum(
                    gp.quicksum(
                        self.c.beta_coeffs[i].U_bytes[b] * y[i][b]
                        for b in range(n_layers)
                    )
                    + self.c.statics[i].RBS_bytes * k[i]
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
                    (bc.Ecomp_J[b] + bc.Ecomm_J[b]) * y[i][b]
                    for b in range(n_layers)
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
        layer_sol = {}
        z_sol = {}
        k_sol = {}

        for i in devices:
            z_sol[i] = int(round(z[i].X))
            k_sol[i] = int(round(k[i].X))
            layer_sol[i] = int(round(layer_num[i].X))

        return SolverOutput(
            layer_num=layer_sol,
            z=z_sol,
            k=k_sol,
            obj_value=float(m.ObjVal) if m.SolCount else float("inf"),
            status=str(m.Status),
        )

