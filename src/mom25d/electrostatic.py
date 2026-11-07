"""Electrostatic images in a laterally infinite dielectric stack, no PEC.

Each ray in the static transmission/reflection expansion contributes c/R.
The source-interface launch is 2/(eps_below+eps_above); a dielectric step
reflects (eps_in-eps_out)/(eps_in+eps_out), transmitting 1+reflection.
Complex eps=er*(1-j*tand) includes constant dielectric loss. Images retain
analytic triangle integration, including close reflections. Explicit copper
is the only conductor; the exterior half spaces default to air.
"""
import time

import numpy as np
from scipy.interpolate import CubicSpline

from .static_backends import FillEngine, bounded_map


class DielectricStack:
    def __init__(self, z, eps, tol=1e-8, max_states=200_000):
        self.z, self.eps = np.asarray(z, float), np.asarray(eps, complex)
        if (len(self.eps) != len(self.z)+1 or len(self.z) < 1 or
                not np.isfinite(self.z).all() or (np.diff(self.z) <= 0).any() or
                not np.isfinite(self.eps).all() or (self.eps.real <= 0).any() or (self.eps.imag > 0).any()):
            raise ValueError('stack needs increasing boundaries and passive positive permittivities')
        if not np.isfinite(tol) or tol <= 0 or tol >= 1:
            raise ValueError('dielectric_tol must be between 0 and 1')
        self.tol, self.max_states, self._cache = tol, max_states, {}

    @classmethod
    def from_board(cls, board, tol=1e-8):
        slabs = sorted(board['stack']['layers'], key=lambda l: l['z0'])
        if not slabs:
            raise ValueError('stack electrostatics requires dielectric layers')
        for a, b in zip(slabs, slabs[1:]):
            if abs(a['z1']-b['z0']) > 1e-6:
                raise ValueError('dielectric slabs must be contiguous and non-overlapping')
        z = np.array([slabs[0]['z0']] + [s['z1'] for s in slabs])*1e-3
        eps = [board['stack'].get('bottom_er', 1.0)] + [
            s['er']*(1-1j*s.get('tand', 0.0)) for s in slabs] + [board['stack'].get('top_er', 1.0)]
        return cls(z, eps, tol)

    def interface(self, z):
        i = int(np.argmin(abs(self.z-z)))
        if abs(self.z[i]-z) > 1e-9:
            raise ValueError(f'copper at z={z:g} m is not on a dielectric interface')
        return i

    def images(self, source, observation, field=False):
        """Scalar: [(path distance, coefficient)] for interface observation.
        Field: [(image z, coefficient)] at an observation z off all interfaces.
        `tol` cuts off individual ray amplitudes; check tolerance convergence.
        Excessive image growth raises rather than silently truncating.
        """
        key = source, observation, field
        if key in self._cache:
            return self._cache[key]
        if field and np.min(abs(self.z-observation)) < 1e-12:
            raise ValueError('evaluate layered E away from dielectric interfaces')
        n = len(self.z)-1
        launch = 2/(self.eps[source]+self.eps[source+1])
        states = {(source, 1, 0.): launch, (source, -1, 0.): launch}
        images = {0.: launch} if not field and source == observation else {}
        count = 0

        def add(table, key, value):
            table[key] = table.get(key, 0.) + value

        for _ in range(4096):
            following = {}
            for (i, direction, distance), amp in states.items():
                if abs(amp) < self.tol*abs(launch):
                    continue
                count += 1
                if count > self.max_states:
                    raise RuntimeError('dielectric image expansion exceeded max_states; simplify stack or relax tolerance')
                medium = i if direction > 0 else i-1
                boundary = i+direction
                exterior = medium < 0 or medium >= n
                if field:
                    travel = direction*(observation-self.z[i])
                    length = np.inf if exterior else abs(self.z[boundary]-self.z[i])
                    if 0 < travel < length:
                        image_z = round(observation-direction*(distance+travel), 14)
                        add(images, image_z, amp)
                if exterior:
                    continue
                distance = round(distance+abs(self.z[boundary]-self.z[i]), 14)
                ein, eout = self.eps[medium+1], self.eps[medium+1+direction]
                reflection = (ein-eout)/(ein+eout)
                transmitted = amp*(1+reflection)
                if not field and boundary == observation:
                    add(images, distance, transmitted)
                add(following, (boundary, direction, distance), transmitted)
                add(following, (boundary, -direction, distance), amp*reflection)
            states = following
            if not states:
                self._cache[key] = list(sorted(images.items()))
                return self._cache[key]
        raise RuntimeError('dielectric images did not converge after 4096 interface crossings')


def potential_moments(model):
    """Triangle-pair int int sum(c/R), before the 1/(4*pi*eps0) factor."""
    stack = model._dielectric_stack
    started = time.perf_counter()
    engine = FillEngine(model.nodes, model.tris, model.z_of, model.near, model.outer_levels,
                        **model._fill_options, prefer_outer=True)
    g = engine.geometry
    nodes, cent, radius, points, area = g.nodes, g.cent, g.radius, g.points, g.area
    v = nodes[g.tris]
    M = np.zeros((len(v), len(v)), complex)
    rho_max = np.linalg.norm(np.ptp(nodes, axis=0))
    layers = list(model.layer_tris.values())
    image_count = 0
    for li, pa in enumerate(layers):
        for pb in layers[li:]:
            o, s = stack.interface(model.z_of[pa[0]]), stack.interface(model.z_of[pb[0]])
            images = stack.images(s, o)
            image_count += len(images)
            positive = [(d, c) for d, c in images if d > 0]
            smooth = None
            if len(positive) > 8 and np.isfinite(model.near):
                # Sum the smooth images once, rather than re-evaluating
                # thousands of weak reflections for every triangle pair.
                # Grading resolves the shortest image distance near rho=0.
                ngrid = max(257, int(np.ceil(32*np.sqrt(rho_max/min(d for d, _ in positive)))))
                rho = rho_max*np.linspace(0, 1, ngrid)**2
                kernel = np.zeros(ngrid, complex)
                for d, c in positive:
                    kernel += c/np.hypot(rho, d)
                smooth = CubicSpline(rho, kernel)
            def evaluate_row(p):
                qs = pb[pb >= p] if pa is pb else pb
                row = np.empty(len(qs), complex)
                for start in range(0, len(qs), engine.chunk):
                    q = qs[start:start+engine.chunk]
                    pp = np.full(len(q), p, int)
                    values = np.zeros(len(q), complex)
                    rho2 = ((cent[q]-cent[p])**2).sum(-1)
                    delta = points[p, :, None]-points[q, None, :]
                    quadrature_rho2 = (delta*delta).sum(-1)
                    if smooth is not None:
                        values += area[p]*area[q]*smooth(np.sqrt(quadrature_rho2)).mean((1, 2))
                    for d, coefficient in images:
                        dist = np.sqrt(rho2+d*d)
                        close = dist <= model.near*(radius[p]+radius[q])
                        if smooth is not None and d > 0:
                            if not close.any():
                                continue
                            # Replace the smooth product rule by analytic
                            # source integration wherever an image is close.
                            approximate = 1/np.sqrt(quadrature_rho2[close]+d*d)
                            values[close] -= coefficient*area[p]*area[q[close]]*approximate.mean((1, 2))
                        refine = close & (dist <= radius[p]+radius[q])
                        for mask, level in ((close & ~refine, 0), (refine, model.outer_levels)):
                            if mask.any():
                                kinds = np.full(mask.sum(), level, np.int32)
                                values[mask] += coefficient*engine.evaluate(
                                    pp[mask], q[mask], d, kinds, scalar_only=True)[:, 0]
                        far = ~close if d == 0 or smooth is None else np.zeros(len(q), bool)
                        if far.any():
                            kinds = np.full(far.sum(), -1, np.int32)
                            values[far] += coefficient*engine.evaluate(
                                pp[far], q[far], d, kinds, scalar_only=True)[:, 0]
                    row[start:start+len(q)] = values
                return p, qs, row

            for p, q, values in bounded_map(evaluate_row, pa, engine.tile_workers):
                M[p, q] = M[q, p] = values
    model.fill_info['capacitance'] = dict(engine.info, image_terms=image_count, scalar_only=True,
                                         seconds=time.perf_counter()-started)
    return M
