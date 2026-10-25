import jax
import diffrax
from typing import Optional
from collections.abc import Callable
from typing import Any, ClassVar
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm, MultiTerm
from diffrax._custom_types import VF
from diffrax._custom_types import RealScalarLike, Y, Args, BoolScalarLike, DenseInfo
from diffrax._solver.base import _SolverState
from jaxtyping import PyTree
from equinox.internal import ω

class ReversibleHeun(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        del t1
        vf0 = terms.vf(t0, y0, args)
        return y0, vf0

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        yhat0, vf0 = solver_state
        vf0 = terms.vf(t0, y0, args)
        control = terms.contr(t0, t1)
        yhat1 = (2 * y0 ** ω - yhat0 ** ω + terms.prod(vf0, control) ** ω).ω
        vf1 = terms.vf(t1, yhat1, args)
        y1 = (y0 ** ω + 0.5 * terms.prod((vf0 ** ω + vf1 ** ω).ω, control) ** ω).ω
        y1_error = (0.5 * terms.prod((vf1 ** ω - vf0 ** ω).ω, control) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        solver_state = (yhat1, vf1)
        return y1, y1_error, dense_info, solver_state, RESULTS.successful


class Midpoint(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        dw = diffusion.contr(t0, t1)
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0_prod = diffusion.vf_prod(t0, y0, args, dw)
        tm = t0 + dt * 0.5
        ym = (y0 ** ω + 0.5 * f0_dt ** ω + 0.5 * g0_prod ** ω).ω
        fm_dt = drift.vf_prod(tm, ym, args, dt)
        gm_prod = diffusion.vf_prod(tm, ym, args, dw)
        y1 = (y0 ** ω + fm_dt ** ω + gm_prod ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

class Euler_dW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0_prod = diffusion.vf(t0, y0, args)
        y1 = (y0 ** ω + f0_dt ** ω + g0_prod ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

class Heun_dW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0_prod = diffusion.vf(t0, y0, args)
        yp = (y0 ** ω + f0_dt ** ω + g0_prod ** ω).ω
        fp_dt = drift.vf_prod(t1, yp, args, dt)
        gp_prod = diffusion.vf(t1, yp, args)
        y1 = (y0 ** ω + 0.5 * (f0_dt ** ω + fp_dt ** ω + g0_prod ** ω + gp_prod ** ω)).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

class EulerHeun_dW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0_prod = diffusion.vf(t0, y0, args)
        yp = (y0 ** ω + g0_prod ** ω).ω
        gp_prod = diffusion.vf(t1, yp, args)
        y1 = (y0 ** ω + f0_dt ** ω + 0.5 * (g0_prod ** ω + gp_prod ** ω)).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

class ReversibleHeun_dW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        del t1
        vf0 = terms.vf(t0, y0, args)
        return y0, vf0

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        yhat0, vf0 = solver_state
        f0, g0_prod = terms.vf(t0, y0, args)
        yhat1 = (2 * y0 ** ω - yhat0 ** ω + drift.prod(f0, dt) ** ω + g0_prod ** ω).ω
        f1, g1_prod = terms.vf(t1, yhat1, args)
        y1 = (y0 ** ω + 0.5 * drift.prod((f0 ** ω + f1 ** ω).ω, dt) ** ω + 0.5 * g0_prod ** ω + 0.5 * g1_prod ** ω).ω
        y1_error = (y0 ** ω + 0.5 * drift.prod((f1 ** ω - f0 ** ω).ω, dt) ** ω + 0.5 * g1_prod ** ω - 0.5 * g0_prod ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        solver_state = (yhat1, (f1, g1_prod))
        return y1, y1_error, dense_info, solver_state, RESULTS.successful

class Midpoint_dW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0_prod = diffusion.vf(t0, y0, args)
        tm = t0 + dt * 0.5
        ym = (y0 ** ω + 0.5 * f0_dt ** ω + 0.5 * g0_prod ** ω).ω
        fm_dt = drift.vf_prod(tm, ym, args, dt)
        gm_prod = diffusion.vf(tm, ym, args)
        y1 = (y0 ** ω + fm_dt ** ω + gm_prod ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

class Milstein_dW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, None, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        f0_prod = drift.vf_prod(t0, y0, args, dt)
        g0_prod = diffusion.vf(t0, y0, args)

        def _to_jvp(_y0):
            return diffusion.vf(t0, _y0, args)

        _, v0_prod = jax.jvp(_to_jvp, (y0,), (g0_prod,))
        y1 = (y0**ω + f0_prod**ω + g0_prod**ω + 0.5 * v0_prod**ω).ω

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)


class Euler_dtdW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        args["dt"] = dt
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)
        y1 = (y0 ** ω + f0_dt ** ω + g0 ** ω * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class Heun_dtdW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        args["dt"] = dt
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)
        yp = (y0 ** ω + f0_dt ** ω + g0 ** ω * dt).ω
        fp_dt = drift.vf_prod(t1, yp, args, dt)
        gp = diffusion.vf(t1, yp, args)
        y1 = (y0 ** ω + 0.5 * (f0_dt ** ω + fp_dt ** ω) + 0.5 * (g0 ** ω + gp ** ω) * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class EulerHeun_dtdW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        args["dt"] = dt
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)
        yp = (y0 ** ω + g0 ** ω * dt).ω
        gp = diffusion.vf(t1, yp, args)
        y1 = (y0 ** ω + f0_dt ** ω + 0.5 * (g0 ** ω + gp ** ω) * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class ReversibleHeun_dtdW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        del t1
        vf0 = terms.vf(t0, y0, args)
        return y0, vf0

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        args["dt"] = dt
        yhat0, vf0 = solver_state
        f0, g0 = terms.vf(t0, y0, args)
        yhat1 = (2 * y0 ** ω - yhat0 ** ω + drift.prod(f0, dt) ** ω + g0 ** ω * dt).ω
        f1, g1 = terms.vf(t1, yhat1, args)
        y1 = (y0 ** ω + 0.5 * drift.prod((f0 ** ω + f1 ** ω).ω, dt) ** ω +
              0.5 * (g0 ** ω + g1 ** ω) * dt).ω
        y1_error = (y0 ** ω + 0.5 * drift.prod((f1 ** ω - f0 ** ω).ω, dt) ** ω +
                    0.5 * (g1 ** ω - g0 ** ω) * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        solver_state = (yhat1, (f1, g1))
        return y1, y1_error, dense_info, solver_state, RESULTS.successful


class Midpoint_dtdW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        args["dt"] = dt
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)
        tm = t0 + dt * 0.5
        ym = (y0 ** ω + 0.5 * f0_dt ** ω + 0.5 * g0 ** ω * dt).ω
        fm_dt = drift.vf_prod(tm, ym, args, dt)
        gm = diffusion.vf(tm, ym, args)
        y1 = (y0 ** ω + fm_dt ** ω + gm ** ω * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class Milstein_dtdW(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def step(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, None, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dw"] = diffusion.contr(t0, t1)
        args["dt"] = dt
        f0_prod = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)

        def _to_jvp(_y0):
            return diffusion.vf(t0, _y0, args)

        _, v0 = jax.jvp(_to_jvp, (y0,), (g0,))
        y1 = (y0 ** ω + f0_prod ** ω + g0 ** ω * dt + 0.5 * v0 ** ω * dt).ω

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)


class Euler_dt(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dt"] = dt
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)
        y1 = (y0 ** ω + f0_dt ** ω + g0 ** ω * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class Heun_dt(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dt"] = dt
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)
        yp = (y0 ** ω + f0_dt ** ω + g0 ** ω * dt).ω
        fp_dt = drift.vf_prod(t1, yp, args, dt)
        gp = diffusion.vf(t1, yp, args)
        y1 = (y0 ** ω + 0.5 * (f0_dt ** ω + fp_dt ** ω) + 0.5 * (g0 ** ω + gp ** ω) * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class EulerHeun_dt(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dt"] = dt
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)
        yp = (y0 ** ω + g0 ** ω * dt).ω
        gp = diffusion.vf(t1, yp, args)
        y1 = (y0 ** ω + f0_dt ** ω + 0.5 * (g0 ** ω + gp ** ω) * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class ReversibleHeun_dt(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        del t1
        vf0 = terms.vf(t0, y0, args)
        return y0, vf0

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dt"] = dt
        yhat0, vf0 = solver_state
        f0, g0 = terms.vf(t0, y0, args)
        yhat1 = (2 * y0 ** ω - yhat0 ** ω + drift.prod(f0, dt) ** ω + g0 ** ω * dt).ω
        f1, g1 = terms.vf(t1, yhat1, args)
        y1 = (y0 ** ω + 0.5 * drift.prod((f0 ** ω + f1 ** ω).ω, dt) ** ω +
              0.5 * (g0 ** ω + g1 ** ω) * dt).ω
        y1_error = (y0 ** ω + 0.5 * drift.prod((f1 ** ω - f0 ** ω).ω, dt) ** ω +
                    0.5 * (g1 ** ω - g0 ** ω) * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        solver_state = (yhat1, (f1, g1))
        return y1, y1_error, dense_info, solver_state, RESULTS.successful


class Midpoint_dt(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)

    def step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dt"] = dt
        f0_dt = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)
        tm = t0 + dt * 0.5
        ym = (y0 ** ω + 0.5 * f0_dt ** ω + 0.5 * g0 ** ω * dt).ω
        fm_dt = drift.vf_prod(tm, ym, args, dt)
        gm = diffusion.vf(tm, ym, args)
        y1 = (y0 ** ω + fm_dt ** ω + gm ** ω * dt).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class Milstein_dt(diffrax.AbstractStratonovichSolver):
    term_structure: ClassVar = MultiTerm[
        tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]
    ]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def init(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> _SolverState:
        return None

    def step(
            self,
            terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y0: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, None, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        args["dt"] = dt
        f0_prod = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf(t0, y0, args)

        def _to_jvp(_y0):
            return diffusion.vf(t0, _y0, args)

        _, v0 = jax.jvp(_to_jvp, (y0,), (g0,))
        y1 = (y0 ** ω + f0_prod ** ω + g0 ** ω * dt + 0.5 * v0 ** ω * dt).ω

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
            self,
            terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
            t0: RealScalarLike,
            y0: Y,
            args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)