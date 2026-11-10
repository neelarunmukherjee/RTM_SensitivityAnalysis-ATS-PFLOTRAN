"""Read / write single parameter values in the three baseline templates (PFLOTRAN reaction .in, reaction database .dat, ATS XML).
Edits are line-based and depend on the exact template layout, so every lookup fails loudly (TemplateError) when the layout differs."""
import re

class TemplateError(Exception): pass

def _blocks(L):
    starts = [i for i, l in enumerate(L) if l == "MICROBIAL_REACTION"]
    if len(starts) != 10: raise TemplateError(f"expected 10 MICROBIAL_REACTION blocks in the reaction .in, found {len(starts)}")
    return [(s, next(j for j in range(s + 1, len(L)) if L[j] == "/")) for s in starts]
def _sub(L, s, e, kind, species=None, thr=False):
    res = []; i = s
    while i < e:
        if L[i].strip() == kind:
            j = next(q for q in range(i + 1, e + 1) if L[q].strip() == "/"); seg = L[i:j]
            ok = (species is None or any(l.split()[:2] == ["SPECIES_NAME", species] for l in seg)) and (not thr or any(l.split()[:2] == ["TYPE", "THRESHOLD"] for l in seg))
            if ok: res.append((i, j))
            i = j
        i += 1
    return res
def _field(L, i, j, key):
    try: return next(q for q in range(i, j) if L[q].split()[:1] == [key])
    except StopIteration: raise TemplateError(f"keyword {key} not found in lines {i}-{j}")

def in_get_set(L, spec, value=None):
    """Return the current value of a reaction-.in parameter; set it (in place) when `value` is given."""
    name, kind, rx, sp = spec[:4]; blk = _blocks(L)
    if kind == "k":
        s, e = blk[rx - 1]; q = _field(L, s, e, "RATE_CONSTANT"); old = float(L[q].split()[1])
        if value is not None: L[q] = f"  RATE_CONSTANT       {value:.6e}"
        return old
    if kind in ("monod", "inhib"):
        s, e = blk[rx - 1]; hits = _sub(L, s, e, "MONOD" if kind == "monod" else "INHIBITION", sp)
        if len(hits) != 1: raise TemplateError(f"{name}: expected one {kind} term for {sp} in reaction {rx}, found {len(hits)}")
        (i, j), = hits; key = "HALF_SATURATION_CONSTANT" if kind == "monod" else "INHIBITION_CONSTANT"; q = _field(L, i, j, key); old = float(L[q].split()[1])
        if value is not None: L[q] = f"    {key} {value:.6e}"
        return old
    if kind in ("cap_thr", "cap_scale"):
        olds = []
        for s, e in blk:
            for i, j in _sub(L, s, e, "INHIBITION", None, thr=True):
                if kind == "cap_thr":
                    q = _field(L, i, j, "INHIBITION_CONSTANT"); olds.append(-float(L[q].split()[1]))
                    if value is not None: L[q] = f"    INHIBITION_CONSTANT -{value:.6e}"
                else:
                    q = next(q for q in range(i, j) if L[q].split()[:2] == ["TYPE", "THRESHOLD"]); olds.append(float(L[q].split()[2]))
                    if value is not None: L[q] = f"    TYPE THRESHOLD {value:.6e}"
        if len(set(olds)) != 1 or len(olds) != 4: raise TemplateError(f"{name}: expected 4 identical CO2-cap terms, found {olds}")
        return olds[0]
    raise TemplateError(f"{name}: unknown reaction-.in kind {kind!r}")

def db_get_set(L, spec, value=None):
    kind = spec[1]; start, marker = ("'Fe(OH)3' 34.3600 3", "'H2O'") if kind == "logK_FeOH3" else ("'SOM' 100.", "'DOM1'")
    try: q = next(i for i, l in enumerate(L) if l.startswith(start)); w = L[q].split(); k = w.index(marker)
    except (StopIteration, ValueError): raise TemplateError(f"{spec[0]}: line starting {start!r} with {marker} not found in the database")
    old = float(w[k + 1])
    if value is not None: w[k + 1:k + 9] = [f"{value:.4f}"] * 8; L[q] = " ".join(w)
    return old

def xml_get_set(L, spec, value=None):
    """mineral_rate_constant: dof 1 = SOM, dof 2 = Fe(OH)3; the same value is set in the three layers (acrotelm, catotelm, mineral)"""
    dof = 1 if spec[1] == "xml_som" else 2
    try: i0 = next(i for i, l in enumerate(L) if 'name="mineral_rate_constant"' in l)
    except StopIteration: raise TemplateError("mineral_rate_constant not found in the ATS XML")
    idx = [q for q in range(i0, i0 + 400) if 'name="value" type="double"' in L[q] and any(('name="dof %d function"' % dof) in L[r] for r in range(max(i0, q - 4), q))][:3]
    if len(idx) != 3: raise TemplateError(f"{spec[0]}: expected 3 layer values, found {len(idx)}")
    old = float(re.search(r'type="double" value="([^"]+)"', L[idx[0]]).group(1))
    if value is not None:
        for q in idx: L[q] = re.sub(r'(name="value" type="double" value=")[^"]+(")', rf'\g<1>{value:.6e}\g<2>', L[q])
    return old

def apply_values(Lin, Ldb, Lx, values, params):
    """write every parameter in `values` (dict name -> float) into the three line lists, in place"""
    for sp in params:
        v = float(values[sp[0]]); k = sp[1]
        if k.startswith("logK"): db_get_set(Ldb, sp, v)
        elif k.startswith("xml"): xml_get_set(Lx, sp, v)
        else: in_get_set(Lin, sp, v)

def read_lines(path): return open(path).read().split("\n")

def base_values(cfg, params):
    """current value of every parameter, read from the baseline files"""
    Lin, Ldb, Lx = read_lines(cfg.reaction_in), read_lines(cfg.database), read_lines(cfg.xml); out = {}
    for sp in params:
        k = sp[1]; out[sp[0]] = db_get_set(Ldb, sp) if k.startswith("logK") else xml_get_set(Lx, sp) if k.startswith("xml") else in_get_set(Lin, sp)
    return out
