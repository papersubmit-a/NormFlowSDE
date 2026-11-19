"""Convert the robot models used by interactive/robot-view into compact browser assets.

Sources (fetched separately, licences copied next to the output):
  ALOHA      gym-aloha  gym_aloha/assets  (ViperX 300 6DOF arms of the ALOHA simulation, Apache-2.0;
             the STL meshes are the vendor's ViperX meshes, BSD-3-Clause)   -> the simulator the dataset was recorded in
  REASSEMBLE mujoco_menagerie franka_emika_panda (Apache-2.0)
For every body of the MJCF kinematic tree: parent, position, orientation, joint (hinge/slide, axis). Visual meshes are
transformed into their body frame, merged per (body, colour), decimated (fast-simplification), and stored quantised
(uint16 positions, uint16/uint32 indices, base64) in interactive/robot-view/models/<robot>.js, so that the page works
from file:// without fetch().

Usage: python scripts/build_robot_models.py --aloha <gym-aloha>/gym_aloha/assets --panda <menagerie>/franka_emika_panda
Needs: numpy, fast-simplification.
"""
import argparse
import base64
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import fast_simplification
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'interactive' / 'robot-view' / 'models'


# ----------------------------------------------------------------------------- geometry helpers
def quat_norm(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def euler_xyz(e):
    """MuJoCo default eulerseq 'xyz' (intrinsic): R = Rx(a) Ry(b) Rz(c)."""
    a, b, c = e
    qx = np.array([np.cos(a / 2), np.sin(a / 2), 0, 0])
    qy = np.array([np.cos(b / 2), 0, np.sin(b / 2), 0])
    qz = np.array([np.cos(c / 2), 0, 0, np.sin(c / 2)])
    return quat_mul(quat_mul(qx, qy), qz)


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame(el):
    pos = np.array([float(v) for v in el.get('pos', '0 0 0').split()])
    if el.get('quat'):
        q = quat_norm([float(v) for v in el.get('quat').split()])
    elif el.get('euler'):
        q = euler_xyz([float(v) for v in el.get('euler').split()])
    else:
        q = np.array([1.0, 0, 0, 0])
    return pos, q


def read_stl(path):
    data = path.read_bytes()
    n = int.from_bytes(data[80:84], 'little')
    if 84 + 50 * n == len(data):  # binary
        rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
        tri = rec['v'].reshape(-1, 3).astype(np.float64)
    else:
        tri = np.array(re.findall(r'vertex\s+(\S+)\s+(\S+)\s+(\S+)', data.decode('ascii', 'ignore')), float)
    v, inv = np.unique(np.round(tri, 7), axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3)


def read_obj(path):
    vs, fs = [], []
    for line in path.read_text(errors='ignore').splitlines():
        if line.startswith('v '):
            vs.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) for t in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(vs) + i for i in idx]
            fs += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    # weld duplicated vertices (OBJ exports repeat positions per normal/uv), so that decimation sees one surface
    v, inv = np.unique(np.round(np.array(vs, float), 7), axis=0, return_inverse=True)
    f = inv.reshape(-1)[np.array(fs, int)]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v, f


def decimate(v, f, target):
    if len(f) <= target:
        return v, f
    v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - target / len(f))
    return v2.astype(np.float64), f2.astype(np.int64)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


# ----------------------------------------------------------------------------- MJCF walk
def walk(body_el, parent, out, meshes, colors, defaults_joint, joint_class=None):
    """Append body_el and its descendants to out (list of dicts)."""
    pos, q = frame(body_el)
    idx = len(out)
    rec = {'name': body_el.get('name'), 'parent': parent, 'pos': pos.round(6).tolist(), 'quat': q.round(7).tolist(),
           'joint': None, 'parts': []}
    for j in body_el.findall('joint'):
        jd = dict(defaults_joint.get(j.get('class') or body_el.get('childclass') or joint_class or '', {}))
        jd.update(j.attrib)
        rec['joint'] = {'name': j.get('name'), 'type': jd.get('type', 'hinge'),
                        'axis': [float(t) for t in jd.get('axis', '0 0 1').split()]}
    pieces = {}
    for g in body_el.findall('geom'):
        if g.get('class') == 'collision' or g.get('mesh') is None:
            continue
        name = g.get('mesh')
        if name not in meshes:
            continue
        gp, gq = frame(g)
        v, f = meshes[name]
        vb = v @ quat_mat(gq).T + gp
        col = colors(g)
        pv, pf = pieces.setdefault(col, ([], []))
        off = sum(len(a) for a in pv)
        pv.append(vb)
        pf.append(f + off)
    for col, (pv, pf) in pieces.items():
        rec['parts'].append({'color': col, 'v': np.concatenate(pv), 'f': np.concatenate(pf)})
    out.append(rec)
    for child in body_el.findall('body'):
        walk(child, idx, out, meshes, colors, defaults_joint, body_el.get('childclass') or joint_class)
    return out


def finish(bodies, budget):
    """Decimate every part to a share of the triangle budget and quantise."""
    total = sum(len(p['f']) for b in bodies for p in b['parts'])
    stats = [0, 0]
    for b in bodies:
        for p in b['parts']:
            target = max(60, int(budget * len(p['f']) / total))
            v, f = decimate(p['v'], p['f'], target)
            lo, hi = v.min(0), v.max(0)
            span = np.maximum(hi - lo, 1e-9)
            qv = np.round((v - lo) / span * 65535).astype('<u2')
            fi = f.astype('<u2' if len(v) < 65536 else '<u4')
            p.update({'lo': lo.round(6).tolist(), 'span': span.round(6).tolist(), 'pos': b64(qv), 'idx': b64(fi),
                      'wide': bool(len(v) >= 65536)})
            stats[0] += len(f)
            stats[1] += len(v)
            del p['v'], p['f']
    return stats


def write(name, model, licences):
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / f'{name}.js'
    js.write_text('/* Generated by scripts/build_robot_models.py; see models/LICENSES.md. */\n'
                  f'(window.ROBOT_MODELS = window.ROBOT_MODELS || {{}})[{json.dumps(name)}] = '
                  + json.dumps(model, separators=(',', ':')) + ';\n', encoding='utf-8')
    for src, dst in licences:
        shutil.copy(src, OUT / dst)
    print(f'{js.relative_to(ROOT)}: {js.stat().st_size / 1e6:.2f} MB')


def build_aloha(assets, budget):
    dep = ET.parse(assets / 'vx300s_dependencies.xml').getroot()
    meshes = {}
    for m in dep.iter('mesh'):
        v, f = read_stl(assets / m.get('file'))
        meshes[m.get('name')] = (v * float(m.get('scale', '1').split()[0]), f)
    grey = {'vx300s_10_gripper_finger_left': '#4a4f55', 'vx300s_10_gripper_finger_right': '#4a4f55'}
    colors = lambda g: grey.get(g.get('mesh'), '#2b2f33')  # noqa: E731  (ViperX arms are black)
    bodies = []
    for side in ('left', 'right'):
        root = ET.parse(assets / f'vx300s_{side}.xml').getroot().find('body')
        walk(root, -1, bodies, meshes, colors, {})
    tris, verts = finish(bodies, budget)
    model = {'name': 'ALOHA (2 x ViperX 300 6DOF, ALOHA simulation)', 'up': 'z', 'bodies': bodies, 'triangles': tris}
    print(f'aloha: {len(bodies)} bodies, {tris} triangles, {verts} vertices')
    return model


def build_panda(src, budget):
    root = ET.parse(src / 'panda.xml').getroot()
    mats = {m.get('name'): m.get('rgba') for m in root.iter('material')}
    defaults = {}
    for d in root.iter('default'):
        j = d.find('joint')
        if j is not None and d.get('class'):
            defaults[d.get('class')] = dict(j.attrib)
    meshes = {}
    for m in root.find('asset').iter('mesh'):
        f = m.get('file')
        if f.endswith('.obj'):
            meshes[m.get('name') or Path(f).stem] = read_obj(src / 'assets' / f)

    def colors(g):
        r, gg, b, _ = [float(t) for t in mats.get(g.get('material'), '1 1 1 1').split()]
        return '#%02x%02x%02x' % (int(r * 255), int(gg * 255), int(b * 255))
    body = root.find('worldbody').find('body')
    bodies = walk(body, -1, [], meshes, colors, defaults, None)
    tris, verts = finish(bodies, budget)
    model = {'name': 'Franka Emika Panda (MuJoCo Menagerie)', 'up': 'z', 'bodies': bodies, 'triangles': tris}
    print(f'panda: {len(bodies)} bodies, {tris} triangles, {verts} vertices')
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--aloha', type=Path, required=True)
    ap.add_argument('--panda', type=Path, required=True)
    ap.add_argument('--aloha-tris', type=int, default=24000)
    ap.add_argument('--panda-tris', type=int, default=36000)
    a = ap.parse_args()
    write('aloha', build_aloha(a.aloha, a.aloha_tris), [(a.aloha.parents[1] / 'LICENSE', 'LICENSE-gym-aloha.txt')])
    write('panda', build_panda(a.panda, a.panda_tris), [(a.panda / 'LICENSE', 'LICENSE-franka_emika_panda.txt')])
