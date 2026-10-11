"""
Writes a small KiCad 8 board for testing the import path (no KiCad needed):
  4-layer stackup with solder mask; F.Cu microstrip -> blind via -> In2.Cu stripline;
  In1.Cu GND plane with an antipad (filled zone), GND stitching vias to B.Cu;
  two port footprints 'mom25d:RF_Port' carrying Fab text P1 / P2; and the pick-and-place file.
"""
import math, sys, uuid

def u(): return str(uuid.uuid4())

def circle_pts(cx, cy, r, n=32):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]

def make(path, pos_path):
    W, H = 24.0, 10.0                     # board (mm)
    yc = 5.0
    via = (12.0, yc)
    # return-path vias around the signal via, and a GND fence along the stripline (In2.Cu)
    stitch = [(via[0] + 1.3 * math.cos(a), yc + 1.3 * math.sin(a)) for a in (math.radians(d) for d in (45, 135, 225, 315))]
    stitch += [(x, yc + s) for x in [13.5 + 1.2 * i for i in range(9)] for s in (-1.0, 1.0)]
    L = []
    L.append('(kicad_pcb (version 20240108) (generator "mom25d") (generator_version "8.0")')
    L.append('  (general (thickness 0.7))')
    L.append('  (paper "A4")')
    L.append('  (layers (0 "F.Cu" signal) (1 "In1.Cu" signal) (2 "In2.Cu" signal) (31 "B.Cu" signal)'
             ' (37 "F.SilkS" user) (39 "F.Mask" user) (44 "Edge.Cuts" user) (49 "F.Fab" user))')
    L.append('  (setup (stackup')
    L.append('    (layer "F.SilkS" (type "Top Silk Screen"))')
    L.append('    (layer "F.Mask" (type "Top Solder Mask") (thickness 0.015) (epsilon_r 3.3) (loss_tangent 0.025))')
    L.append('    (layer "F.Cu" (type "copper") (thickness 0.035))')
    L.append('    (layer "dielectric 1" (type "prepreg") (thickness 0.2) (material "FR4") (epsilon_r 4.3) (loss_tangent 0.02))')
    L.append('    (layer "In1.Cu" (type "copper") (thickness 0.035))')
    L.append('    (layer "dielectric 2" (type "core") (thickness 0.2) (material "FR4") (epsilon_r 4.3) (loss_tangent 0.02))')
    L.append('    (layer "In2.Cu" (type "copper") (thickness 0.035))')
    L.append('    (layer "dielectric 3" (type "prepreg") (thickness 0.2) (material "FR4") (epsilon_r 4.3) (loss_tangent 0.02))')
    L.append('    (layer "B.Cu" (type "copper") (thickness 0.035))')
    L.append('    (copper_finish "None") (dielectric_constraints no))')
    L.append('    (pad_to_mask_clearance 0))')
    L.append('  (net 0 "") (net 1 "GND") (net 2 "RF")')
    # microstrip on F.Cu, stripline on In2.Cu
    L.append(f'  (segment (start 2.0 {yc}) (end {via[0]} {yc}) (width 0.35) (layer "F.Cu") (net 2) (uuid "{u()}"))')
    L.append(f'  (segment (start {via[0]} {yc}) (end 22.0 {yc}) (width 0.15) (layer "In2.Cu") (net 2) (uuid "{u()}"))')
    L.append(f'  (via blind (at {via[0]} {via[1]}) (size 0.5) (drill 0.25) (layers "F.Cu" "In2.Cu") (net 2) (uuid "{u()}"))')
    for x, y in stitch:
        L.append(f'  (via (at {x:.4f} {y:.4f}) (size 0.5) (drill 0.25) (layers "F.Cu" "B.Cu") (net 1) (uuid "{u()}"))')
    # In1.Cu GND plane with antipad (filled polygon: outer outline + hole as a slit polygon, like KiCad)
    outer = [(0, 0), (W, 0), (W, H), (0, H)]
    hole = circle_pts(via[0], via[1], 0.6)[::-1]
    pts = outer + [outer[0], (via[0] + 0.6, via[1])] + hole + [(via[0] + 0.6, via[1]), outer[0]]
    L.append(f'  (zone (net 1) (net_name "GND") (layer "In1.Cu") (uuid "{u()}") (hatch edge 0.5)')
    L.append('    (connect_pads (clearance 0.2)) (min_thickness 0.2) (filled_areas_thickness no)')
    L.append('    (fill yes (thermal_gap 0.5) (thermal_bridge_width 0.5))')
    L.append(f'    (polygon (pts {" ".join(f"(xy {x} {y})" for x, y in outer)}))')
    L.append(f'    (filled_polygon (layer "In1.Cu") (pts {" ".join(f"(xy {x:.4f} {y:.4f})" for x, y in pts)})))')
    # port footprints (no copper): Fab text P1/P2, rotation = direction into the circuit
    for ref, num, x, rot, lay in (("TP1", 1, 2.0, 0, None), ("TP2", 2, 22.0, 180, "In2.Cu")):
        L.append(f'  (footprint "mom25d:RF_Port" (layer "F.Cu") (uuid "{u()}") (at {x} {yc} {rot})')
        L.append(f'    (property "Reference" "{ref}" (at 0 -1.5 {rot}) (layer "F.SilkS") (uuid "{u()}"))')
        L.append(f'    (property "Value" "RF_Port" (at 0 1.5 {rot}) (layer "F.Fab") (uuid "{u()}"))')
        if lay:
            L.append(f'    (property "mom.layer" "{lay}" (at 0 0 0) (layer "F.Fab") (hide yes) (uuid "{u()}"))')
        L.append(f'    (fp_text user "P{num}" (at 0 0.8 {rot}) (layer "F.Fab") (uuid "{u()}"))')
        L.append(f'    (fp_line (start -0.5 0) (end 0.5 0) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{u()}"))')
        L.append('  )')
    L.append(f'  (gr_rect (start 0 0) (end {W} {H}) (stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts") (uuid "{u()}"))')
    L.append(')')
    open(path, 'w').write("\n".join(L) + "\n")
    with open(pos_path, 'w') as fh:
        fh.write("### Footprint positions - created by mom25d demo ###\n## Unit = mm, Angle = deg.\n## Side : top\n")
        fh.write("# Ref     Val       Package                PosX       PosY       Rot  Side\n")
        fh.write(f"TP1       P1        RF_Port              2.0000    -{yc:.4f}     0.0000  top\n")
        fh.write(f"TP2       P2        RF_Port             22.0000    -{yc:.4f}   180.0000  top\n")

if __name__ == '__main__' and not (len(sys.argv) > 2 and sys.argv[2] == 'diff'):
    out = sys.argv[1] if len(sys.argv) > 1 else 'demo_transition'
    make(out + '.kicad_pcb', out + '-top.pos')
    print('wrote', out + '.kicad_pcb', out + '-top.pos')


def make_diff(path, pos_path):
    """2-layer board: edge-coupled microstrip pair (w 0.25, gap 0.2 mm) on F.Cu over B.Cu with a 45 deg
    jog (unequal inner/outer path lengths -> some differential-to-common conversion), and four port
    footprints with Fab text P1+ / P1- / P2+ / P2-."""
    w, s, yc = 0.25, 0.2, 6.0
    off = (w + s) / 2
    L = []
    L.append('(kicad_pcb (version 20240108) (generator "mom25d") (generator_version "8.0")')
    L.append('  (general (thickness 0.2))\n  (paper "A4")')
    L.append('  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (39 "F.Mask" user) (44 "Edge.Cuts" user) (49 "F.Fab" user))')
    L.append('  (setup (stackup')
    L.append('    (layer "F.Mask" (type "Top Solder Mask") (thickness 0.015) (epsilon_r 3.3) (loss_tangent 0.025))')
    L.append('    (layer "F.Cu" (type "copper") (thickness 0.035))')
    L.append('    (layer "dielectric 1" (type "core") (thickness 0.2) (material "FR4") (epsilon_r 4.3) (loss_tangent 0.02))')
    L.append('    (layer "B.Cu" (type "copper") (thickness 0.035))')
    L.append('    (copper_finish "None") (dielectric_constraints no))\n    (pad_to_mask_clearance 0))')
    L.append('  (net 0 "") (net 1 "GND") (net 2 "D_P") (net 3 "D_N")')
    # centre line: (2, yc) -> (9, yc) -> (11, yc - 2) -> (22, yc - 2); each line offset by +-off
    # (a mitred 45-degree jog: the offsets are applied perpendicular to each segment)
    pts = [(2.0, yc), (9.0, yc), (11.0, yc - 2.0), (22.0, yc - 2.0)]
    import math
    def offset_path(sign):
        out = []
        n = len(pts)
        for i in range(n):
            if i == 0 or i == n - 1:
                (x0, y0), (x1, y1) = (pts[0], pts[1]) if i == 0 else (pts[-2], pts[-1])
                dx, dy = x1 - x0, y1 - y0; l = math.hypot(dx, dy)
                nx, ny = -dy / l, dx / l
                x, y = pts[i]
                out.append((x + sign * off * nx, y + sign * off * ny))
            else:
                (xa, ya), (xb, yb), (xc, yc_) = pts[i - 1], pts[i], pts[i + 1]
                d1 = ((xb - xa), (yb - ya)); l1 = math.hypot(*d1); d1 = (d1[0] / l1, d1[1] / l1)
                d2 = ((xc - xb), (yc_ - yb)); l2 = math.hypot(*d2); d2 = (d2[0] / l2, d2[1] / l2)
                n1 = (-d1[1], d1[0]); n2 = (-d2[1], d2[0])
                bis = (n1[0] + n2[0], n1[1] + n2[1]); bl = math.hypot(*bis)
                cosh = (bis[0] * n1[0] + bis[1] * n1[1]) / bl
                k = off / cosh
                out.append((xb + sign * k * bis[0] / bl, yb + sign * k * bis[1] / bl))
        return out
    for net, sign in ((2, -1), (3, +1)):             # KiCad y down: P line at smaller y (top)
        p = offset_path(sign)
        for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
            L.append(f'  (segment (start {x0:.4f} {y0:.4f}) (end {x1:.4f} {y1:.4f}) (width {w}) (layer "F.Cu") (net {net}) (uuid "{u()}"))')
    pP, pN = offset_path(-1), offset_path(+1)
    ports = [("TP1", "P1+", pP[0], 0), ("TP2", "P1-", pN[0], 0), ("TP3", "P2+", pP[-1], 180), ("TP4", "P2-", pN[-1], 180)]
    for ref, txt, (x, y), rot in ports:
        L.append(f'  (footprint "mom25d:RF_Port" (layer "F.Cu") (uuid "{u()}") (at {x:.4f} {y:.4f} {rot})')
        L.append(f'    (property "Reference" "{ref}" (at 0 -1.5 {rot}) (layer "F.SilkS") (uuid "{u()}"))')
        L.append(f'    (property "Value" "RF_Port" (at 0 1.5 {rot}) (layer "F.Fab") (uuid "{u()}"))')
        L.append(f'    (fp_text user "{txt}" (at 0 0.8 {rot}) (layer "F.Fab") (uuid "{u()}"))\n  )')
    L.append(f'  (gr_rect (start 0 0) (end 24 10) (stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts") (uuid "{u()}"))\n)')
    open(path, 'w').write("\n".join(L) + "\n")
    with open(pos_path, 'w') as fh:
        fh.write("### Footprint positions ###\n## Unit = mm, Angle = deg.\n# Ref Val Package PosX PosY Rot Side\n")
        for ref, txt, (x, y), rot in ports:
            fh.write(f"{ref} {txt} RF_Port {x:.4f} {-y:.4f} {rot:.4f} top\n")


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[2] == 'diff':
    make_diff(sys.argv[1] + '.kicad_pcb', sys.argv[1] + '-top.pos')
    print('wrote', sys.argv[1] + '.kicad_pcb')
