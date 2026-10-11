"""
Stage-2 demo: broadside-coupled stripline quarter-wave coupler (two metal layers, PEC cover).

   PEC cover
   D3  0.20 mm  er 4.3
   M2  ---- top trace: section + feeds going +y (ports P1, P2)
   D2  0.10 mm  er 4.3
   M1  ---- bottom trace: section + feeds going -y (ports P3, P4)
   D1  0.20 mm  er 4.3
   ground

Lossless and fully enclosed -> no radiation, so sum_i |S_i1|^2 must be 1 (strict check).
Homogeneous dielectric -> even/odd modes have equal velocity -> P4 (isolated) should be very low.
Outputs: coupler_dashboard.png, coupler.s4p
Run: python demo_ml.py   (a few minutes)
"""
import numpy as np
from mom25d.multilayer import *
import mom25d.viz as viz

er, w = 4.3, 0.15e-3
st = LayerStack([Layer(0.2e-3, er, name='core'), Layer(0.1e-3, er, name='prepreg'),
                 Layer(0.2e-3, er, name='core')], top='pec')
f0 = 5e9
Lc = C0 / f0 / np.sqrt(er) / 4                     # quarter wave (homogeneous stripline)
gd, step, s0, ref = 0.5e-3, 1.0e-3, 1.5e-3, 6.5e-3
Lf = 2e-3 + ref + gd                               # reference planes 2 mm from the section
M2 = [(0, -w / 2, Lc, w / 2), (0, -w / 2, w, Lf), (Lc - w, -w / 2, Lc, Lf)]
M1 = [(0, -w / 2, Lc, w / 2), (0, -Lf, w, w / 2), (Lc - w, -Lf, Lc, w / 2)]
kw = dict(s0=s0, step=step)
ports = [MLPort(2, 'y', Lf - gd, (0, w), -1, ref, name='P1', **kw),
         MLPort(2, 'y', Lf - gd, (Lc - w, Lc), -1, ref, name='P2', **kw),
         MLPort(1, 'y', -Lf + gd, (0, w), +1, ref, name='P3', **kw),
         MLPort(1, 'y', -Lf + gd, (Lc - w, Lc), +1, ref, name='P4', **kw)]

if __name__ == '__main__':
    freqs = np.linspace(3e9, 7e9, 9)
    print("feed-line characterisation (M1 and M2 should agree by symmetry)")
    g1, z1 = line_char_ml(st, 1, w, [f0])
    g2, z2 = line_char_ml(st, 2, w, [f0])
    print(f"  Zc M1 {z1[0].real:.2f}  M2 {z2[0].real:.2f} ohm   eps_eff {(g1[0].imag / (2*np.pi*f0/C0))**2:.4f}")
    zc = 0.5 * (z1[0].real + z2[0].real)
    for p in ports:
        p.zc = zc
    sim = MultiLayerMoM(st, {1: M1, 2: M2}, ports, dmax=0.3e-3, ncross=6,
                        title='Broadside-coupled stripline λ/4 coupler (stage 2 demo)')
    res = sim.sweep(freqs, currents_at=f0)
    k = np.argmin(abs(freqs - f0))
    S = res.S[k]
    print(f"\nat {freqs[k]/1e9:.1f} GHz (ref. {zc:.1f} ohm):")
    for i in range(4):
        print(f"  |S{i+1}1| = {20*np.log10(abs(S[i,0])):7.2f} dB")
    print(f"  sum |S_i1|^2 = {np.sum(abs(S[:,0])**2):.4f}   max|S-S^T| = {res.recip.max():.1e}")
    with open('coupler.s4p', 'w') as fh:
        fh.write(f"! broadside stripline coupler, mom25d_ml\n# Hz S RI R {zc:.3f}\n")
        for fi, s in zip(res.f, res.S):
            vals = [s[i, j] for i in range(4) for j in range(4)]
            for r in range(4):
                row = vals[4*r:4*r+4]
                fh.write((f"{fi:.6e} " if r == 0 else " " * 13) +
                         " ".join(f"{v.real:.6e} {v.imag:.6e}" for v in row) + "\n")
    print('wrote', viz.dashboard(sim, res, 'coupler_dashboard.png'))
