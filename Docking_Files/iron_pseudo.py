#!/usr/bin/env python
"""Tetrahedral Fe pseudo-atom (TF) for AutoDock4 Fe metalloprotein maps — MBind-style."""
import math
from copy import deepcopy
import sys

def canon_ad4_type(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return t
    u = t.upper()
    specials = {"NA", "OA", "SA", "HD", "HS"}
    if u in specials:
        return u
    if t.isalpha() and len(t) == 2:
        return t[0].upper() + t[1].lower()
    if len(t) == 1 and t.isalpha():
        return t.upper()
    return t

def dist(a, b):
    return math.sqrt(sum([(a[i] - b[i]) ** 2 for i in range(min(len(a), len(b)))]))

def angle(a, b, c):
    d12, d13, d23 = dist(a, b), dist(a, c), dist(b, c)
    return math.acos(round((d12 ** 2 + d13 ** 2 - d23 ** 2) / (2 * d12 * d13), 7))

def angled(a, b, c):
    return angle(a, b, c) * 180 / math.pi

def dihedral(a, b, c, d):
    v1 = [b[i] - a[i] for i in range(3)]
    v2 = [c[i] - b[i] for i in range(3)]
    v3 = [d[i] - c[i] for i in range(3)]
    temp = [dist((0, 0, 0), v2) * v1[i] for i in range(3)]
    y = dotProd(temp, crossProd(v2, v3))
    x = dotProd(crossProd(v1, v2), crossProd(v2, v3))
    return math.atan2(y, x) * (180 / math.pi)

def dotProd(a, b):
    return sum(a[i] * b[i] for i in range(min(len(a), len(b))))

def crossProd(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

def rawVec(a, b):
    return [b[i] - a[i] for i in range(min(len(a), len(b)))]

class PDBQT:
    def __init__(self, line):
        self._parse_common(line)
        self._parse_specific(line)

    def getline(self):
        return self._print_common() + self._print_specific()

    def getcoords(self):
        return (self.x, self.y, self.z)

    def _parse_common(self, line):
        self.keyword = line[0:6]
        self.serial = int(line[6:11])
        self.name = line[12:16]
        self.altLoc = line[16:17]
        self.resName = line[17:20]
        self.chain = line[21:22]
        self.resNum = int(line[22:26])
        self.icode = line[26:27]
        self.x = float(line[30:38])
        self.y = float(line[38:46])
        self.z = float(line[46:54])
        self.occupancy = float(line[54:60])
        self.bfact = float(line[60:66])

    def _parse_specific(self, line):
        self.charge = float(line[68:76])
        self.atype = canon_ad4_type(line[77:79].strip())
        self.atomnr = self.atype_to_atomnr(self.atype)

    def _print_common(self):
        return (
            f"{self.keyword:6s}{self.serial:5d} {self.name:4s}{self.altLoc}{self.resName:3s} "
            f"{self.chain}{self.resNum:4d}{self.icode}   "
            f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}{self.occupancy:6.2f}{self.bfact:6.2f}"
        )

    def _print_specific(self):
        t = f"{self.atype:2s}"[:2]
        return f"  {self.charge:8.3f} {t}\n"

    def dist(self, other):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.getcoords(), other.getcoords())))

    def isbound(self, other_atom, cut_off_percent=0.5):
        th = cut_off_percent * (0.5 * self.atomnr_vdw(self.atomnr) + 0.5 * self.atomnr_vdw(other_atom.atomnr))
        return self.dist(other_atom) < th

    def atomnr_vdw(self, atomnr):
        D = {1: 2.0, 6: 4.0, 7: 3.5, 8: 3.2, 9: 3.1, 12: 1.3, 15: 4.2, 16: 4.0, 17: 4.1,
             20: 2.0, 25: 1.3, 26: 1.3, 29: 1.4, 30: 1.5, 35: 4.3, 53: 4.7}
        return D.get(atomnr, 2.0)

    def atype_to_atomnr(self, atype):
        D = {"H": 1, "HD": 1, "HS": 1, "C": 6, "A": 6, "N": 7, "NA": 7, "NS": 7,
             "OA": 8, "OS": 8, "F": 9, "MG": 12, "S": 16, "SA": 16, "Cl": 17, "CL": 17,
             "CA": 20, "MN": 25, "FE": 26, "ZN": 30, "CU": 29, "Cu": 29, "BR": 35, "Br": 35,
             "I": 53, "G": 6, "J": 6, "P": 15, "Z": 6, "GA": 6, "Q": 6, "TF": -1, "TZ": -1}
        u = atype.strip().upper()
        return D.get(u, D.get(atype.strip(), -1))

def load_pdbqt(filename):
    atoms_list, max_id, num_skip = [], 0, 0
    non_atom_text = {}
    counter = 0
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                atom = PDBQT(line)
                if atom.atype == "TF":
                    num_skip += 1
                    continue
                counter += 1
                if canon_ad4_type(atom.atype) == "Fe":
                    atom.charge = 0.0
                atoms_list.append(atom)
                max_id = max(max_id, atom.serial)
            else:
                non_atom_text.setdefault(counter, []).append(line)
    if counter in non_atom_text:
        non_atom_text["last"] = non_atom_text.pop(counter)
    return atoms_list, num_skip, max_id, non_atom_text

def bruteNearbyAtoms(atoms_list, metal="Fe", cut_off=4.5):
    mc = canon_ad4_type(metal)
    out = []
    for metal_atom in [a for a in atoms_list if canon_ad4_type(a.atype) == mc]:
        idx = sorted(range(len(atoms_list)), key=lambda i: atoms_list[i].dist(metal_atom))
        out.append([atoms_list[i] for i in idx if atoms_list[i].dist(metal_atom) < cut_off])
    return out

class MetalShell:
    def __init__(self, metal, cut_off=2.5, carboxy_exp=0.5):
        self.zn = metal
        self.c = cut_off
        self.e = carboxy_exp
        self.rec = []

    def _getBonds(self, atoms):
        n = len(atoms)
        n_conn = [0] * n
        connect = []
        for i in range(n):
            for j in range(i + 1, n):
                if atoms[i].isbound(atoms[j]):
                    connect.append((i, j))
                    n_conn[i] += 1
                    n_conn[j] += 1
        return connect, n_conn

    def _getCarboxyOxyIndx(self, atoms, connect, n_conn):
        co = []
        for i, j in connect:
            if atoms[i].atype in ["O", "OA"] and atoms[j].atype in ["C", "A"] and n_conn[i] == 1:
                co.append((j, i))
            if atoms[j].atype in ["O", "OA"] and atoms[i].atype in ["C", "A"] and n_conn[j] == 1:
                co.append((i, j))
        coo = []
        for i in range(len(co)):
            for j in range(i + 1, len(co)):
                if co[i][0] == co[j][0]:
                    coo.append((co[i][0], min(co[i][1], co[j][1]), max(co[i][1], co[j][1])))
        return coo

    def _rmCarboxy(self, near_atoms, indx_pairs):
        rm = sorted({x for c, i, j in indx_pairs for x in (c, i, j)}, reverse=True)
        for i in rm:
            near_atoms.pop(i)
        return near_atoms

    def _avgCarboxy(self, oxys):
        if self.e < 0:
            return [o for p in oxys for o in p]
        avgs = []
        for oxypair in oxys:
            A = deepcopy(oxypair[0])
            o1, o2 = oxypair[0].getcoords(), oxypair[1].getcoords()
            zn = self.zn.getcoords()
            d1, d2, oo = dist(o1, zn), dist(o2, zn), dist(o1, o2)
            ratio = ((d2 - d1) / oo) ** self.e if oo > 1e-6 else 0
            w = (1 - ratio) / 2
            v12 = rawVec(o1, o2)
            A.x, A.y, A.z = (o1[i] + v12[i] * w for i in range(3))
            A.name, A.atype = "AVG", "OC"
            avgs.append(A)
        return avgs

    def _build13(self, connect):
        c13 = set()
        for i, j in connect:
            for x, y in connect:
                if j == x:
                    c13.add((min(i, y), max(i, y)))
                if j == y:
                    c13.add((min(i, x), max(i, x)))
        reach = {}
        for i, j in connect + list(c13):
            reach.setdefault(i, []).append(j)
            reach.setdefault(j, []).append(i)
        return reach

    def _rm_too_far(self, atoms):
        return [a for a in atoms if a.dist(self.zn) <= self.c]

    def _selectBinders(self, near_atoms, reach3):
        valid = {canon_ad4_type(t) for t in ["O", "OA", "NA", "N", "S", "SA", "MG", "MN", "ZN", "CA", "FE", "CU"]}
        remove = set()
        for i, atom in enumerate(near_atoms):
            if canon_ad4_type(atom.atype) not in valid or atom.dist(self.zn) > self.c:
                remove.add(i)
            elif i not in remove and i in reach3:
                remove.update(j for j in reach3[i] if j not in remove)
        return [a for i, a in enumerate(near_atoms) if i not in remove]

    def build_shell(self, atoms):
        connect, n_conn = self._getBonds(atoms)
        coo = self._getCarboxyOxyIndx(atoms, connect, n_conn)
        Os = [(atoms[o], atoms[O]) for c, o, O in coo]
        atoms = self._rmCarboxy(list(atoms), coo)
        connect, n_conn = self._getBonds(atoms)
        reach = self._build13(connect)
        atoms = self._selectBinders(atoms, reach)
        n0 = len(atoms)
        atoms = atoms + self._avgCarboxy(Os)
        atoms = self._rm_too_far(atoms)
        return atoms

    def proc_rec(self, near_atoms):
        self.rec = self.build_shell(near_atoms)

def tetra_tf(znobj, d=2.0):
    n = len(znobj.rec)
    zn = znobj.zn.getcoords()
    if n == 3:
        pseudo = deepcopy(znobj.zn)
        pseudo.name, pseudo.atype = "TF", "TF"
        a, b, c = [x.getcoords() for x in znobj.rec]
        w = 2 * (int(dihedral(a, b, c, zn)) > 0) - 1
        nv = crossProd(rawVec(a, b), rawVec(b, c))
        L = dist(nv, (0, 0, 0))
        if L < 1e-6:
            return None
        nvn = [w * d * nv[i] / L for i in range(3)]
        pseudo.x, pseudo.y, pseudo.z = zn[0] + nvn[0], zn[1] + nvn[1], zn[2] + nvn[2]
        pseudo.charge = 0.0
        return pseudo
    return None

def fallback_tf(metal, rec, d=2.0):
    zn = metal.getcoords()
    pseudo = deepcopy(metal)
    pseudo.name, pseudo.atype, pseudo.charge = "TF", "TF", 0.0
    if len(rec) >= 3:
        a, b, c = [x.getcoords() for x in rec[:3]]
        w = 2 * (int(dihedral(a, b, c, zn)) > 0) - 1
        nv = crossProd(rawVec(a, b), rawVec(b, c))
        L = dist(nv, (0, 0, 0))
        if L < 1e-6:
            pseudo.x, pseudo.y, pseudo.z = zn[0] + d, zn[1], zn[2]
        else:
            nvn = [w * d * nv[i] / L for i in range(3)]
            pseudo.x, pseudo.y, pseudo.z = zn[0] + nvn[0], zn[1] + nvn[1], zn[2] + nvn[2]
    elif len(rec) == 2:
        a, b = rec[0].getcoords(), rec[1].getcoords()
        nv = crossProd(rawVec(zn, a), rawVec(zn, b))
        L = dist(nv, (0, 0, 0))
        if L < 1e-6:
            pseudo.x, pseudo.y, pseudo.z = zn[0] + d, zn[1], zn[2]
        else:
            nvn = [d * nv[i] / L for i in range(3)]
            pseudo.x, pseudo.y, pseudo.z = zn[0] + nvn[0], zn[1] + nvn[1], zn[2] + nvn[2]
    else:
        pseudo.x, pseudo.y, pseudo.z = zn[0] + d, zn[1], zn[2]
    return pseudo

def main():
    import getopt
    from os.path import splitext
    opts, _ = getopt.gnu_getopt(sys.argv[1:], "r:o:h", ["help"])
    input_name = output_name = None
    for o, a in opts:
        if o in ("-h", "--help"):
            print("python iron_pseudo.py -r receptor.pdbqt [-o out.pdbqt]")
            sys.exit(0)
        if o == "-r":
            input_name = a
        if o == "-o":
            output_name = a
    if not input_name:
        sys.stderr.write("missing -r receptor\n")
        sys.exit(2)
    if not output_name:
        output_name = splitext(input_name)[0] + "_TF.pdbqt"

    r, _, lastserial, non_atom_text = load_pdbqt(input_name)
    cutoff, carboxy, distance = 2.5, 0.5, 2.0
    tf_list = []
    for alist in bruteNearbyAtoms(r, "Fe"):
        metal = alist[0]
        shell = MetalShell(metal, cutoff, carboxy)
        shell.proc_rec(alist[1:])
        p = tetra_tf(shell, distance)
        if p is None:
            p = fallback_tf(metal, shell.rec, distance)
        tf_list.append(p)

    for p in tf_list:
        lastserial += 1
        p.serial = lastserial
        r.append(p)

    print(f"Wrote {len(tf_list)} TF pseudo-atom(s) -> {output_name}")
    with open(output_name, "w") as out:
        for i, atom in enumerate(r):
            if i in non_atom_text:
                for line in non_atom_text[i]:
                    out.write(line)
            out.write(atom.getline())
        if "last" in non_atom_text:
            for line in non_atom_text["last"]:
                out.write(line)

if __name__ == "__main__":
    main()
