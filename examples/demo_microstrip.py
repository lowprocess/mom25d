"""
Demo / validation for mom25d.py (stage 1):
  1. Green's function vs exact image solution (er = 1)
  2. Uniform 50-ohm microstrip: Z0 and eps_eff from the 3D solver
  3. Open-end extension vs Hammerstad
  4. Shunt open-stub sweep -> stub_sweep.png and stub.s2p (Touchstone)
Run:  python demo_mom25d.py        (~1.5 min)
"""
import numpy as np
from mom25d.base import *

w, h, er, tand = 0.35e-3, 0.2e-3, 4.3, 0.02          # 50-ohm line on 0.2 mm FR4-like
st = Stackup(h, er, tand)

# ---- 1. Green's function check ------------------------------------------------
f0 = 5e9; k0 = 2 * np.pi * f0 / C0
gt = GreenTable(f0, Stackup(h, 1.0, 0.0), 20e-3)
rho = np.array([1e-4, 5e-4, 2e-3, 8e-3, 15e-3]); ri = np.sqrt(rho ** 2 + 4 * h ** 2)
exact = gt.cA * ((np.exp(-1j * k0 * rho) - 1) / rho) - gt.cA * np.exp(-1j * k0 * ri) / ri
print(f"[1] Green's function (er=1) max rel. error vs exact images: "
      f"{np.max(np.abs(gt.A(rho) - exact) / np.abs(exact)):.1e}")

# ---- 2. Line characterisation --------------------------------------------------
fc = np.array([2e9, 5e9, 10e9])
gam, zc = line_characterization(st, w, fc, length=20e-3, dx=0.5e-3, dy=w / 6)
print("[2] uniform line (2D quasi-static reference: Z0 = 53.4 ohm, eps_eff = 3.18)")
for f, g, z in zip(fc, gam, zc):
    print(f"    {f/1e9:4.0f} GHz  eps_eff = {(g.imag / (2*np.pi*f/C0))**2:.3f}   Zc = {z.real:.1f} ohm   "
          f"alpha = {g.real*8.686:.2f} dB/m")

# ---- 3. Open-end extension ----------------------------------------------------
dx, dy = 0.175e-3, w / 4
Lf, gd, xr = 8e-3, 1e-3, -3e-3
p = Port('x', -Lf + gd, (-w/2, w/2), +1, xr - (-Lf + gd), s0=1.5e-3, step=0.7e-3, name='P1')
sim = PlanarMoM(st, [(-Lf, -w/2, 0, w/2)], [p], dx, dy, verbose=False)
gi = np.interp(5e9, fc, gam.imag); p.zc = np.interp(5e9, fc, zc.real)
S11 = sim.sparams(5e9)[0, 0]
ph = -np.angle(S11); n = np.round((2*gi*(-xr) - ph) / (2*np.pi))
print(f"[3] open-end extension at 5 GHz: {((ph + 2*np.pi*n)/(2*gi) + xr)*1e3:.3f} mm   (Hammerstad 0.077 mm)")

# ---- 4. Shunt open stub sweep --------------------------------------------------
Ls, Lh = 4e-3, 8e-3
rects = [(-Lh, -w/2, Lh, w/2), (-w/2, w/2, w/2, w/2 + Ls)]       # line + stub
ref = Lh - gd - 2e-3                                              # ref planes at x = -/+ 2 mm
p1 = Port('x', -Lh + gd, (-w/2, w/2), +1, ref, s0=1.5e-3, step=0.7e-3, name='P1')
p2 = Port('x',  Lh - gd, (-w/2, w/2), -1, ref, s0=1.5e-3, step=0.7e-3, name='P2')
sim = PlanarMoM(st, rects, [p1, p2], dx, dy)
f = np.linspace(2e9, 14e9, 25)
gs, zs_ = line_characterization(st, w, f[::6], length=20e-3, dx=0.5e-3, dy=dy)
zline = np.interp(f, f[::6], zs_.real)
S = np.zeros((len(f), 2, 2), complex)
for k, (fi, z) in enumerate(zip(f, zline)):
    p1.zc = p2.zc = z
    S[k] = renormalize(sim.sparams(fi), [z, z], 50.0)
print(f"[4] stub notch at {f[np.argmin(abs(S[:,1,0]))]/1e9:.2f} GHz")

with open('stub.s2p', 'w') as fh:                                 # Touchstone, RI, 50 ohm
    fh.write("# Hz S RI R 50\n")
    for fi, s in zip(f, S):
        fh.write(f"{fi:.6e} " + " ".join(f"{s[i,j].real:.6e} {s[i,j].imag:.6e}"
                 for i, j in ((0, 0), (1, 0), (0, 1), (1, 1))) + "\n")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(f/1e9, 20*np.log10(abs(S[:, 1, 0])), 'o-', label='|S21|')
    ax.plot(f/1e9, 20*np.log10(abs(S[:, 0, 0])), 's-', label='|S11|')
    ax.set_xlabel('GHz'); ax.set_ylabel('dB'); ax.set_ylim(-40, 1); ax.grid(alpha=.3); ax.legend()
    ax.set_title('Shunt open stub (4 mm), 2.5D MoM'); fig.tight_layout(); fig.savefig('stub_sweep.png', dpi=140)
except ImportError:
    pass
