import torch


def euler_solver(
    init_cond,
    t_span,
    dt,
    model=None,
    conds=None,
    class_conds=None,
    cfg_scale=None,
    device="cpu",
):
    """
    Euler method solver for ODE: dZ/dt = v(Z, t)

    Parameters:
        init_cond: Initial condition for Z
        t_span: Tuple (t0, tf) specifying initial and final time
        dt: Step size
        model: Model state for apply_fn
        conds: Conditions for the model
        cfg_scale: Configuration scale

    Returns:
        Z: solutions
    """
    t0, tf = t_span
    num_steps = abs(int((tf - t0) / dt) + 1)  # Number of time steps
    t = torch.linspace(t0, tf, num_steps, device=device)  # Time array
    Z = init_cond.to(device)

    # Simple wrapper to make less cluttered on ODE loop
    def _func_cfg(init_cond, t):
        #
        with torch.no_grad():
            cond_vector = model(init_cond, t, conds, class_conds)
            uncond_vector = model(
                init_cond, t, None, None
            )  # NOTE: unsure how uncond works, assuming it's simply not being conditioned at all instead of padding tensor
        return uncond_vector + (cond_vector - uncond_vector) * cfg_scale

    # Euler method iteration
    for i in range(1, num_steps):
        Z = Z - _func_cfg(Z, t[i - 1].unsqueeze(0).expand_as(Z)) * dt

    return Z
