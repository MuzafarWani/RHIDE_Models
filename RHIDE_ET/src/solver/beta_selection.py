#beta_selection.py
def select_beta_min_energy(
    beta_coeffs,
    device_static,
    allowed_betas,
):
    """
    Returns beta_name or None if no feasible beta exists.
    Inference-safe version: robust to YAML string types.
    """

    feasible = []

    for idx, beta in enumerate(allowed_betas):
        # ---- Force numeric types (YAML safety) ----
        Ecomp = float(beta_coeffs.Ecomp_J[idx])
        Ecomm = float(beta_coeffs.Ecomm_J[idx])
        U     = float(beta_coeffs.U_bytes[idx])
        Tcomp = float(beta_coeffs.Tcomp_s[idx])
        Tcomm = float(beta_coeffs.Tcomm_s[idx])
        M     = float(beta_coeffs.M_bytes[idx])

        E = Ecomp + Ecomm
        T = Tcomp + Tcomm + device_static.Tedge_s + device_static.Tsync_s

        if (
            E <= device_static.Emax_round_J and
            E <= device_static.Erem_J and
            U <= device_static.Umax_bytes and
            T <= device_static.Tmax_s and
            M <= device_static.mem_max_bytes
        ):
            feasible.append((E, beta.name))

    if not feasible:
        return None

    # Minimum-energy feasible β
    return min(feasible, key=lambda x: x[0])[1]

