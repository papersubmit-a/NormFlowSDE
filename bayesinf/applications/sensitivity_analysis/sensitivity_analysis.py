import jax.numpy as jnp
from jax import jit, vmap, lax
import jax.random as jr
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.ticker as ticker
import numpy as np
from functools import partial
from sklearn.feature_selection import mutual_info_regression


@jit
def mass_spring_ode_f(t, state, args):
    """
    The first-order ODE for the mass-spring system with external force.
    state = (x, v)
    args = (params,), where params = (m, k, c)
    """
    x, v = state
    params, = args
    m, k, c = params
    F_t = 10.0 * jnp.sin(3.0 * t)

    dx_dt = v
    dv_dt = F_t / m- (k / m) * x - (c / m) * v

    return jnp.array([dx_dt, dv_dt])

@jit
def mass_spring_ode_l(t, state, args):
    """
    The first-order ODE for the mass-spring system.
    state = (x, v)
    args = (params,), where params = (m, k, c)
    """
    x, v = state
    params, = args
    m, k, c = params

    dx_dt = v
    dv_dt = - (k / m) * x - (c / m) * v

    return jnp.array([dx_dt, dv_dt])

@jit
def mass_spring_ode2(t, state, args):
    """
    System: Wall -- k1 -- m -- k2 -- m -- k3 -- Wall
    We assume m1 = m2 = m.
    k2 is the coupling spring (known).
    k1 and k3 are the unknown parameters we want to infer.
    """
    m=1.0
    k2=2.0
    c=0.1
    x1, v1, x2, v2 = state
    params,  = args
    k1, k3 = params
    # Equations of motion
    # m*a1 = -k1*x1 + k2*(x2 - x1) - c*v1
    # m*a2 = -k3*x2 - k2*(x2 - x1) - c*v2

    dx1dt = v1
    dv1dt = (-k1 * x1 + k2 * (x2 - x1) - c * v1) / m

    dx2dt = v2
    dv2dt = (-k3 * x2 - k2 * (x2 - x1) - c * v2) / m

    return jnp.array([dx1dt, dv1dt, dx2dt, dv2dt])

@partial(jit, static_argnums=(0,))
def rk4_step(ode_func, state, params, t, dt):
    k1 = ode_func(t, state, (params,))
    k2 = ode_func(t + 0.5 * dt, state + 0.5 * dt * k1, (params,))
    k3 = ode_func(t + 0.5 * dt, state + 0.5 * dt * k2, (params,))
    k4 = ode_func(t + dt, state + dt * k3, (params,))
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@partial(jit, static_argnums=(0, 3))
def simulate_jax(ode_func, params, y0, obs_func=None):
    dt = 0.05
    steps = 21

    # We use lax.scan instead of a python loop for maximum speed
    def step_fn(carry, i):
        state, t = carry
        next_state = rk4_step(ode_func, state, params, t, dt)
        return (next_state, t + dt), next_state

    init_state = (y0, 0.0)
    _, traj = lax.scan(step_fn, init_state, jnp.arange(steps))
    if obs_func is not None:
        traj = obs_func(traj)
    return jnp.sqrt(jnp.mean(jnp.square(traj)))

vmap_sim = vmap(simulate_jax, in_axes=(None, 0, None, None))

def run_comparative_analysis(ode_func, problem, noise_levels, N=1024, *, seed=0, return_details=False):
    names = problem['names']
    # Noise is an independent input in the Saltelli design. Physical-parameter
    # paired rows then share the same noise realization, as required by the
    # stochastic-model total-order estimand.
    noise_problem = dict(problem)
    noise_problem.update(
        num_vars=problem['num_vars'] + 1,
        names=list(names) + ['observation_noise'],
        bounds=list(problem['bounds']) + [[0., 1.]],
        dists=list(problem.get('dists', ['unif'] * problem['num_vars'])) + ['norm'],
    )
    if 'groups' in problem:
        raise ValueError("Grouped Sobol designs require an explicit noise group")
    design = saltelli.sample(noise_problem, N)
    param_values, noise_draws = design[:, :-1], design[:, -1]
    j_params = jnp.array(param_values)

    # 1. Generate Clean Simulations
    print(f"Running {len(param_values)} simulations...")
    y0 = jnp.array([1.0, 0.0])
    Y_clean = vmap_sim(ode_func, j_params, y0, None)

    sobol_results = {name: [] for name in names}
    mi_results = {name: [] for name in names}

    details = []
    for sigma in noise_levels:
        print(f"Analyzing Noise Level: σ = {sigma}")

        if sigma < 0:
            raise ValueError("Noise standard deviations must be nonnegative")
        Y_np = np.asarray(Y_clean) + noise_draws * sigma
        Si = sobol.analyze(noise_problem, Y_np, print_to_console=False, seed=seed)
        st_indices = np.asarray(Si['ST'])[:-1]
        details.append({"sigma": float(sigma), "names": noise_problem['names'],
                        "S1": np.asarray(Si['S1']).tolist(),
                        "ST": np.asarray(Si['ST']).tolist(),
                        "S1_conf": np.asarray(Si['S1_conf']).tolist(),
                        "ST_conf": np.asarray(Si['ST_conf']).tolist()})
        # Preserve raw estimates and confidence intervals; finite-sample
        # overshoots are estimator uncertainty, not a reason to rescale.
        for i, name in enumerate(names):
            sobol_results[name].append(st_indices[i]) # Using Total Index ST

        # 4. Compute Mutual Information
        for i, name in enumerate(names):
            # We use a subset for MI to speed up if N is very large,
            # but for 1024 * (2*3+2) = 8192, it's fine.
            score = mutual_info_regression(
                param_values[:, i:i+1],
                Y_np,
                discrete_features=False,
                n_neighbors=5, random_state=seed
            )
            mi_results[name].append(score[0])

    return (sobol_results, mi_results, details) if return_details else (sobol_results, mi_results)

# mass-spring-damper model with external force and noise
problem = {
    'num_vars': 3, # m, k, c
    'names': [r'$m$', r'$k$', r'$c$'],
    'bounds': [[0.8, 1.2], [5.0, 10.0], [0.5, 3]]
}

N = 1024
noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def plot_mi_st_comparison(sobol_res, mi_res, problem):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    FS_LABEL = 18
    FS_TITLE = 18
    FS_TICK = 18
    FS_LEGEND = 18


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for i, name in enumerate(problem['names']):
        # Sobol Plot
        ax1.plot(noise_levels, sobol_res[name], label=name,
                color=colors[i], marker=markers[i], linewidth=2, markersize=8)
        # MI Plot
        ax2.plot(noise_levels, mi_res[name], label=name,
                color=colors[i], marker=markers[i], linewidth=2, markersize=8)

    # Format Left Plot (Sobol)
    ax1.set_xscale('log')
    ax1.set_xticks(noise_levels)

    # Use ScalarFormatter so 0.5 shows as "0.5" instead of "5x10^-1"
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # Clean up formatting so it doesn't use scientific notation for small decimals
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # Grid: both major and minor for log-scale context
    ax1.grid(True, which="major", ls="-", alpha=0.4, color='gray')
    ax1.grid(True, which="minor", ls="--", alpha=0.1, color='gray')

    # Titles and Labels
    ax1.set_title('Variance-Based Sensitivity', fontsize=FS_TITLE, pad=25, fontweight='bold')
    ax1.set_xlabel(r'Noise Standard Deviation ($\sigma$)', fontsize=FS_LABEL, labelpad=15)
    ax1.set_ylabel('Total-Order Sobol\' Index', fontsize=FS_LABEL, labelpad=15)

    # Tick parameters
    ax1.tick_params(axis='both', which='major', labelsize=FS_TICK, length=10, width=1.5)
    ax1.tick_params(axis='x', which='minor', length=5, width=1)
    ax1.legend(fontsize=FS_LEGEND)
    # Format Right Plot (MI)
    ax2.set_xscale('log')
    ax2.set_xticks(noise_levels)

    # Use ScalarFormatter so 0.5 shows as "0.5" instead of "5x10^-1"
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # Clean up formatting so it doesn't use scientific notation for small decimals
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # Grid: both major and minor for log-scale context
    ax2.grid(True, which="major", ls="-", alpha=0.4, color='gray')
    ax2.grid(True, which="minor", ls="--", alpha=0.1, color='gray')
    ax2.set_ylabel('Mutual Information', fontsize=FS_LABEL, labelpad=15)
    ax2.set_xlabel(r'Noise Standard Deviation ($\sigma$)', fontsize=FS_LABEL, labelpad=15)
    ax2.set_title('Information-Theoretic Identifiability', fontsize=FS_TITLE, pad=25, fontweight='bold')
    # Tick parameters
    ax2.tick_params(axis='both', which='major', labelsize=FS_TICK, length=10, width=1.5)
    ax2.tick_params(axis='x', which='minor', length=5, width=1)
    ax2.legend(fontsize=FS_LEGEND)

    plt.tight_layout()
    plt.savefig('sensitivity_identifiability.pdf')

# # mass-spring-damper with external force
# problem = {
#     'num_vars': 3,
#     'names': [r'$m$', r'$k$', r'$c$'],
#     'bounds': [[0.8, 1.2], [5.0, 10.0], [0.5, 3]]
# }

# N = 1024
# # Generate Low-Discrepancy Saltelli Samples
# param_values = saltelli.sample(problem, N)
# # Convert to JAX array
# j_param_values = jnp.array(param_values)

# print(f"Running {len(param_values)} simulations on JAX...")
# start = time.time()

# Y_jax = vmap_sim(mass_spring_ode_f, j_param_values, jnp.array([1.0, 0.0]), None)

# Y_jax.block_until_ready()
# print(f"JAX Time: {time.time() - start:.4f} seconds")

# Si = sobol.analyze(problem, np.array(Y_jax), print_to_console=True)

# # mass-spring-damper with linear correlation
# N = 1024
# # Generate Low-Discrepancy Saltelli Samples
# param_values = saltelli.sample(problem, N)
# j_param_values = jnp.array(param_values)

# print(f"Running {len(param_values)} simulations on JAX...")
# start = time.time()

# Y_jax = vmap_sim(mass_spring_ode_l, j_param_values, jnp.array([1.0, 0.0]), None)

# # Block until finished (JAX is asynchronous)
# Y_jax.block_until_ready()
# print(f"JAX Time: {time.time() - start:.4f} seconds")

# Sil = sobol.analyze(problem, np.array(Y_jax), print_to_console=True)

# # Coupled mass-spring-damper
# problem2 = {
#     'num_vars': 2,
#     'names': [r'$k_1$', r'$k_3$'],
#     'bounds': [[1.0, 10.0], [1.0, 10.0]]
# }
# N = 1024
# # Generate Low-Discrepancy Saltelli Samples
# param_values = saltelli.sample(problem2, N)
# j_param_values = jnp.array(param_values)

# print(f"Running {len(param_values)} simulations on JAX...")
# start = time.time()

# def obs_func(y):
#     return jnp.stack([y[..., 0]+y[..., 2], y[..., 1]+y[..., 3]], axis=-1)

# Y_jax = vmap_sim(mass_spring_ode2, j_param_values, jnp.array([1.0, 0.0, 1.0, 0.0]), obs_func)


# # Block until finished (JAX is asynchronous)
# Y_jax.block_until_ready()
# print(f"JAX Time: {time.time() - start:.4f} seconds")

# Si2 = sobol.analyze(problem2, np.array(Y_jax), print_to_console=True)

def plot_sensitivity_comparison(Si_linear, Si_coupled, names_linear, names_coupled):
    """
    Generates a (1,2) subplot comparing Global Sensitivity Analysis results.

    Args:
        Si_linear: SALib result dictionary for the Linear (External Force) case.
        Si_coupled: SALib result dictionary for the Coupled Two-Mass case.
        names_linear: List of strings (e.g., ['m', 'k', 'c'])
        names_coupled: List of strings (e.g., ['k1', 'k2', 'k3', 'c1', 'c2'])
    """
    plt.rcParams.update({'font.size': 24})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    width = 0.35

    # --- SUBPLOT 1: LINEAR CASE ---
    x1 = np.arange(len(names_linear))
    rects1_s1 = ax1.bar(x1 - width/2, Si_linear['S1'], width, yerr=Si_linear['S1_conf'],
                        label='First Order ($S_1$)', color='#3498db', capsize=5)
    rects1_st = ax1.bar(x1 + width/2, Si_linear['ST'], width, yerr=Si_linear['ST_conf'],
                        label='Total Order ($S_T$)', color='#3498db', alpha=0.4, capsize=5)

    ax1.set_title('Linear Case (Mass-Spring-Damper)', fontsize=24, pad=20)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(names_linear, fontsize=24)
    ax1.set_ylabel('Sensitivity Index', fontsize=24)
    ax1.set_ylim(0, 1.2) # Extra space for labels
    ax1.legend(fontsize=24)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Add labels on top of bars for Plot 1
    ax1.bar_label(rects1_s1, fmt='%.3f', padding=5, fontsize=20, fontweight='bold')
    ax1.bar_label(rects1_st, fmt='%.3f', padding=5, fontsize=20, fontweight='bold')

    # --- SUBPLOT 2: COUPLED CASE ---
    x2 = np.arange(len(names_coupled))
    rects2_s1 = ax2.bar(x2 - width/2, Si_coupled['S1'], width, yerr=Si_coupled['S1_conf'],
                        label='First Order ($S_1$)', color='#e74c3c', capsize=5)
    rects2_st = ax2.bar(x2 + width/2, Si_coupled['ST'], width, yerr=Si_coupled['ST_conf'],
                        label='Total Order ($S_T$)', color='#e74c3c', alpha=0.4, capsize=5)

    ax2.set_title('Coupled Case (Symmetric Two-Mass)', fontsize=24, pad=20)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names_coupled, fontsize=24)
    ax2.set_ylim(0, 1.2) # Extra space for labels
    ax2.legend(fontsize=24)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Add labels on top of bars for Plot 2
    ax2.bar_label(rects2_s1, fmt='%.3f', padding=5, fontsize=20, fontweight='bold')
    ax2.bar_label(rects2_st, fmt='%.3f', padding=5, fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import json
    from pathlib import Path
    sobol_res, mi_res, details = run_comparative_analysis(
        mass_spring_ode_f, problem, noise_levels, N=1024, return_details=True)
    Path("sensitivity_identifiability.json").write_text(json.dumps(details, indent=2))
    plot_mi_st_comparison(sobol_res, mi_res, problem)
