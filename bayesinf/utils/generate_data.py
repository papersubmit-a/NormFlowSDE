import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx


def mass_spring_ode(t, state, args):
        """
        The first-order ODE for the mass-spring system.
        state = (x, v)
        args = (params,), where params = (m, k)
        """
        x, v = state
        params, = args
        m, k, c = params

        dx_dt = v
        dv_dt = - (k / m) * x - (c / m) * v

        return jnp.array([dx_dt, dv_dt])

def generate_mass_spring_data(key: jax.Array,
                              n_samples: int = 2000,
                              dt0: float = 0.05,
                              t1: float = 5.0,
                              noise_std: float = 0.05,
                              saveat='all'):
    key_y0, key_p, key_n = jr.split(key, 3)

    # Sample initial state (x0, v0)
    # Start all trajectories at a random initial displacement and velocity
    x0_batch = jr.uniform(key_y0, (n_samples, 1), minval=0.5, maxval=2.0)
    v0_batch = jnp.zeros((n_samples, 1))  # Start with zero velocity
    y0_batch = jnp.concatenate([x0_batch, v0_batch], axis=-1)  # Shape: (n_samples, 2)

    # Sample physical parameters (m, k, c)
    key_m, key_k, key_c = jr.split(key_p, 3)
    boundary = jnp.array([[0.8, 1.2], [5.0, 10.0], [0.5, 3]])
    m_batch = jr.uniform(key_m, (n_samples, 1), minval=boundary[0, 0], maxval=boundary[0, 1])  # Mass around 1.0
    k_batch = jr.uniform(key_k, (n_samples, 1), minval=boundary[1, 0], maxval=boundary[1, 1])  # Stiffness
    c_batch = jr.uniform(key_c, (n_samples, 1), minval=boundary[2, 0], maxval=boundary[2, 1])  # damping

    # Params stack: (m, k, c) - Shape: (n_samples, 3)
    params = jnp.concatenate([m_batch, k_batch, c_batch], axis=-1)

    term = dfx.ODETerm(mass_spring_ode)
    solver = dfx.Dopri5()

    # Solves from t=0 to t=t1, saving only the final point
    if saveat == 'last':
        sol_func = lambda y0, p: dfx.diffeqsolve(term, solver, t0=0.0, t1=t1, dt0=dt0, y0=y0, args=(p,),
                                                 saveat=dfx.SaveAt(t1=True)).ys[-1]
    else:
        sol_func = lambda y0, p: dfx.diffeqsolve(
            term, solver, t0=0.0, t1=t1, dt0=dt0, y0=y0, args=(p,), saveat=dfx.SaveAt(ts=jnp.linspace(0.0, t1, int(t1 / dt0) + 1))).ys

    # Batch solve
    data = jax.vmap(sol_func)(y0_batch, params)  # Final states z(t1) = (x, v). Shape: (n_samples, 2)

    # # Calculate the derivatives at the final state z(t1)
    # ddata = jax.vmap(lambda s, p: jax.vmap(lambda _s: mass_spring_ode(None, _s, (p,)))(s))(data, params)

    key_n1, key_n2 = jr.split(key_n, 2)

    # Noise for state z(t1)
    if noise_std > 0:
        noise = jr.normal(key_n1, data.shape) * noise_std
        data = data + noise

        # # Noise for derivative dz/dt(t1)
        # noise_d = jr.normal(key_n2, ddata.shape) * noise_std
        # ddata = ddata + noise_d

    return data, params

def mass_spring_ode_f(t, state, args, beta=0.0, alpha=0.0):
        """
        The first-order ODE for the mass-spring system with external force.
        state = (x, v)
        args = (params,), where params = (m, k, c)
        """
        x, v = state
        params, = args
        m, k, c = params
        F_t = 10.0 * jnp.sin(3.0 * t)

        if beta > 0:
           f_spring = - (k * x + beta * jnp.power(x, 3))
        else:
            f_spring = - k * x
        if alpha > 0:
            f_damping = - (c * v + alpha * v * jnp.abs(v))
        else:
            f_damping = - c * v

        dx_dt = v
        dv_dt = (F_t + f_spring + f_damping) / m

        return jnp.array([dx_dt, dv_dt])

def generate_mass_spring_data_f(key: jax.Array,
                              n_samples: int = 2000,
                              dt0: float = 0.05,
                              t1: float = 5.0,
                              noise_std: float = 0.05,
                              saveat='all', beta=0, alpha=0, shift=[0, 0, 0]):

    key_y0, key_p, key_n = jr.split(key, 3)

    # Sample initial state (x0, v0)
    # Start all trajectories at a random initial displacement and velocity
    x0_batch = jr.uniform(key_y0, (n_samples, 1), minval=0.5, maxval=2.0)
    v0_batch = jnp.zeros((n_samples, 1))  # Start with zero velocity
    y0_batch = jnp.concatenate([x0_batch, v0_batch], axis=-1)  # Shape: (n_samples, 2)

    # Sample physical parameters (m, k, c)
    key_m, key_k, key_c = jr.split(key_p, 3)
    boundary = jnp.array([[0.8+shift[0], 1.2+shift[0]], [5.0+shift[1], 10.0+shift[1]], [0.5+shift[2], 3+shift[2]]])
    m_batch = jr.uniform(key_m, (n_samples, 1), minval=boundary[0, 0], maxval=boundary[0, 1])  # Mass around 1.0
    k_batch = jr.uniform(key_k, (n_samples, 1), minval=boundary[1, 0], maxval=boundary[1, 1])  # Stiffness
    c_batch = jr.uniform(key_c, (n_samples, 1), minval=boundary[2, 0], maxval=boundary[2, 1])  # damping

    # Params stack: (m, k, c) - Shape: (n_samples, 3)
    params = jnp.concatenate([m_batch, k_batch, c_batch], axis=-1)

    term = dfx.ODETerm(lambda t, state, args: mass_spring_ode_f(t, state, args, beta=beta, alpha=alpha))
    solver = dfx.Dopri5()

    # Solves from t=0 to t=t1, saving only the final point
    if saveat == 'last':
        sol_func = lambda y0, p: dfx.diffeqsolve(term, solver, t0=0.0, t1=t1, dt0=dt0, y0=y0, args=(p,),
                                                 saveat=dfx.SaveAt(t1=True)).ys[-1]
    else:
        sol_func = lambda y0, p: dfx.diffeqsolve(
            term, solver, t0=0.0, t1=t1, dt0=dt0, y0=y0, args=(p,), saveat=dfx.SaveAt(ts=jnp.linspace(0.0, t1, int(t1 / dt0) + 1))).ys

    # Batch solve
    data = jax.vmap(sol_func)(y0_batch, params)  # Final states z(t1) = (x, v). Shape: (n_samples, 2)

    # # Calculate the derivatives at the final state z(t1)
    # ddata = jax.vmap(lambda s, p: jax.vmap(lambda _s: mass_spring_ode(None, _s, (p,)))(s))(data, params)

    key_n1, key_n2 = jr.split(key_n, 2)

    # Noise for state z(t1)
    if noise_std > 0:
        noise = jr.normal(key_n1, data.shape) * noise_std
        data = data + noise

        # # Noise for derivative dz/dt(t1)
        # noise_d = jr.normal(key_n2, ddata.shape) * noise_std
        # ddata = ddata + noise_d

    return data, params

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

def generate_mass_spring_data_l(key: jax.Array,
                              n_samples: int = 2000,
                              dt0: float = 0.05,
                              t1: float = 5.0,
                              noise_std: float = 0.05,
                              saveat='all'):

    key_y0, key_p, key_n = jr.split(key, 3)

    # Sample initial state (x0, v0)
    # Start all trajectories at a random initial displacement and velocity
    x0_batch = jr.uniform(key_y0, (n_samples, 1), minval=0.5, maxval=2.0)
    v0_batch = jnp.zeros((n_samples, 1))  # Start with zero velocity
    y0_batch = jnp.concatenate([x0_batch, v0_batch], axis=-1)  # Shape: (n_samples, 2)

    # Sample physical parameters (m, k, c)
    key_m, key_c = jr.split(key_p)
    boundary = jnp.array([[0.8, 1.2], [5.0, 10.0], [0.5, 3]])
    m_batch = jr.uniform(key_m, (n_samples, 1), minval=boundary[0, 0], maxval=boundary[0, 1])  # Mass around 1.0
    k_batch = m_batch * 10
    # Params stack: (m, k) - Shape: (n_samples, 3)
    c_batch = jr.uniform(key_c, (n_samples, 1), minval=boundary[2, 0], maxval=boundary[2, 1])  # damping
    params = jnp.concatenate([m_batch, k_batch, c_batch], axis=-1)

    term = dfx.ODETerm(mass_spring_ode_l)
    solver = dfx.Dopri5()

    # Solves from t=0 to t=t1, saving only the final point
    if saveat == 'last':
        sol_func = lambda y0, p: dfx.diffeqsolve(term, solver, t0=0.0, t1=t1, dt0=dt0, y0=y0, args=(p,),
                                                 saveat=dfx.SaveAt(t1=True)).ys[-1]
    else:
        sol_func = lambda y0, p: dfx.diffeqsolve(
            term, solver, t0=0.0, t1=t1, dt0=dt0, y0=y0, args=(p,), saveat=dfx.SaveAt(ts=jnp.linspace(0.0, t1, int(t1 / dt0) + 1))).ys

    # Batch solve
    data = jax.vmap(sol_func)(y0_batch, params)  # Final states z(t1) = (x, v). Shape: (n_samples, 2)

    # # Calculate the derivatives at the final state z(t1)
    # ddata = jax.vmap(lambda s, p: jax.vmap(lambda _s: mass_spring_ode(None, _s, (p,)))(s))(data, params)

    key_n1, key_n2 = jr.split(key_n, 2)

    # Noise for state z(t1)
    if noise_std > 0:
        noise = jr.normal(key_n1, data.shape) * noise_std
        data = data + noise

        # # Noise for derivative dz/dt(t1)
        # noise_d = jr.normal(key_n2, ddata.shape) * noise_std
        # ddata = ddata + noise_d

    return data, params

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

def generate_mass_spring_data2(key: jax.Array,
                               n_samples: int = 2000,
                               dt0: float = 0.05,
                               t1: float = 5.0,
                               noise_std: float = 0.05,
                               saveat='all'):
    
    key_y0, key_p, key_n = jr.split(key, 3)
    key_y01, key_y02 = jr.split(key_y0)
    x10_batch = jr.uniform(key_y01, (n_samples, 1), minval=0.5, maxval=2.0)
    v10_batch = jnp.zeros((n_samples, 1))  # Start with zero velocity
    x20_batch = x10_batch
    v20_batch = v10_batch  # Start with zero velocity
    y0_batch = jnp.concatenate([x10_batch, v10_batch, x20_batch, v20_batch], axis=-1)  # Shape: (n_samples, 4)

    # Sample physical parameters (k1, k3)
    key_k1, key_k3 = jr.split(key_p)
    boundary = jnp.array([[1.0, 10.0], [1.0, 10.0]])
    k1_batch = jr.uniform(key_k1, (n_samples, 1), minval=boundary[0, 0], maxval=boundary[0, 1])  # k1
    k3_batch = jr.uniform(key_k3, (n_samples, 1), minval=boundary[1, 0], maxval=boundary[1, 1])  # k3
    params = jnp.concatenate([k1_batch, k3_batch], axis=-1)

    term = dfx.ODETerm(mass_spring_ode2)
    solver = dfx.Dopri5()

    # Solves from t=0 to t=t1, saving only the final point
    if saveat == 'last':
        sol_func = lambda y0, p: dfx.diffeqsolve(term, solver, t0=0.0, t1=t1, dt0=dt0, y0=y0, args=(p,),
                                                 saveat=dfx.SaveAt(t1=True)).ys[-1]
    else:
        sol_func = lambda y0, p: dfx.diffeqsolve(
            term, solver, t0=0.0, t1=t1, dt0=dt0, y0=y0, args=(p,), saveat=dfx.SaveAt(ts=jnp.linspace(0.0, t1, int(t1 / dt0) + 1))).ys

    # Batch solve
    data = jax.vmap(sol_func)(y0_batch, params)  # Final states z(t1) = (x, v). Shape: (n_samples, 4)
    
    # Noise for state z(t1)
    if noise_std > 0:
        noise = jr.normal(key_n, data.shape) * noise_std
        data = data + noise

    return jnp.stack([data[..., 0]+data[..., 2], data[..., 1]+data[..., 3]], axis=-1), params


def mass_spring_ode_nf(t, state, args):
    """
    Non-linear Mass-Spring-Damper (Duffing + Quadratic Damping)
    """
    x, v = state
    params, = args
    # m: mass, k: linear stiffness, c: linear damping, 
    # beta: non-linear stiffness, alpha: non-linear damping
    m, k, c = params 
    
    F_t = 10.0 * jnp.sin(3.0 * t)

    dx_dt = v
    
    # Physics: F = ma -> a = (F_ext + F_spring + F_damping) / m
    beta = 0.1
    alpha = 0.1
    f_spring = - (k * x + beta * jnp.power(x, 3))
    f_damping = - (c * v + alpha * v * jnp.abs(v))
    
    dv_dt = (F_t + f_spring + f_damping) / m

    return jnp.array([dx_dt, dv_dt])


def normalize_data(data, mean=None, std=None):
    if mean is None:
        mean = jnp.mean(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
        std = jnp.std(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
        return (data - mean) / std, mean, std
    else:
        return (data - mean) / std

def denormalize_data(data, mean, std):
    return data * std + mean

if __name__ == '__main__':
    data, param = generate_mass_spring_data_f(jr.PRNGKey(0), n_samples=2, dt0=0.01, t1=1.0, noise_std=0.0)
    print(data.shape)
