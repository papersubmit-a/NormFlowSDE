"""Unit checks of the signature-kernel PDE solver in direct_baselines.py (run inside the container)."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.special import i0

from experiments.normflowsde_comparison.direct_baselines import (_goursat, linear_signature_kernel,
                                                                  rbf_signature_kernel)


def naive(z):
    lx, ly = z.shape
    k = np.ones((lx + 1, ly + 1))
    for i in range(lx):
        for j in range(ly):
            a = 1 + z[i, j] / 2 + z[i, j] ** 2 / 12
            k[i + 1, j + 1] = (k[i + 1, j] + k[i, j + 1]) * a - k[i, j] * (1 - z[i, j] ** 2 / 12)
    return k[-1, -1]


# 1) associative-scan solver == naive double loop
z = np.asarray(jr.normal(jr.PRNGKey(0), (37, 23))) * 0.3
print("scan vs naive:", float(_goursat(jnp.asarray(z))), naive(z))
assert abs(float(_goursat(jnp.asarray(z))) - naive(z)) < 1e-8 * max(1, abs(naive(z)))

# 2) straight lines, linear static kernel: k = I0(2 sqrt(<a,b>)) (refined grid converges)
a, b = np.array([0.7, -0.4, 1.1]), np.array([0.9, 0.5, 0.8])
c = a @ b
for n in (8, 64, 256):
    s = np.linspace(0, 1, n + 1)[:, None]
    val = float(linear_signature_kernel(jnp.asarray(s * a), jnp.asarray(s * b)))
    print(f"n={n:4d} pde={val:.10f} closed={i0(2 * np.sqrt(c)):.10f}")
assert abs(val - i0(2 * np.sqrt(c))) < 1e-4 * i0(2 * np.sqrt(c))

# 3) symmetry and finite gradients for the RBF lift used in training (float32, horizon 250)
jax.config.update("jax_enable_x64", False)
x = jnp.cumsum(jr.normal(jr.PRNGKey(1), (251, 13)) * 0.1, 0)
y = jnp.cumsum(jr.normal(jr.PRNGKey(2), (251, 13)) * 0.1, 0)
kxy, kyx = rbf_signature_kernel(x, y, 3.6), rbf_signature_kernel(y, x, 3.6)
g = jax.grad(lambda p: rbf_signature_kernel(p, y, 3.6))(x)
print("rbf k(x,y)", float(kxy), "k(y,x)", float(kyx), "k(x,x)", float(rbf_signature_kernel(x, x, 3.6)),
      "grad finite", bool(jnp.isfinite(g).all()))
assert abs(float(kxy) - float(kyx)) < 1e-3 * abs(float(kxy)) and bool(jnp.isfinite(g).all())
print("CHECK_OK")
