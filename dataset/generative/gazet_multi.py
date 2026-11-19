"""Multi-place training data for gazet's plan model: questions that filter a set of divisions by a second (or third)
named place, plus a fix for "regions of <country>" labels.

Why: in "regions of Peru that touch the Pacific Ocean" the 135M plan model extracted only "Peru". In gazet-dataset,
two-place questions are almost all a geometry operation between two named features ("Kabalo and the Congo basin");
"filter a set by its relation to another named place" barely exists. And gazet's templates select the regions of a
country with ST_Within(region, country), which is false for 31% of regions (137 of 195 countries): coastal regions
reach past the country's land polygon. So "regions of Peru ..." plans silently drop every coastal region. Here a
country container is the plan's "country" filter (the code is in the candidate row); region containers keep "within".

    python gazet_multi.py relations             # per country, in capped subprocesses: regions/counties vs nearby
                                                # Natural Earth features, neighbouring countries, containment
    python gazet_multi.py generate [--n 9000]   # questions + gazet's own candidate search + plans + place lists
    python gazet_multi.py paraphrase            # OpenRouter (DeepSeek): one free rewording per question; place names
                                                # verbatim, numbers/negation/ranking checked, plan and candidates reused
    python gazet_multi.py verify [--n 60]       # compiled plans on the full data return the expected features
    python gazet_multi.py export                # chatml/gazet_plan_v2: plan v1 data (relabelled) + the new data
    python gazet_multi.py answer LABEL PORT     # answer the test prompts with llama-server on PORT (as the demo:
                                                # plans constrained to the plan schema and the candidate IDs)
    python gazet_multi.py score PRED_FILE LABEL # plan exact / same features / places on the held-out multi test

Run with gazet's environment (~/repos/gazet/.venv/bin/python). Outputs go to runs/gazet_multi/.
"""
import argparse
import io
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

OUT = HERE / "runs" / "gazet_multi"
REL = OUT / "rel"
DV = str(REPO / "data" / "overture_normalized" / "divisions_area" / "*.parquet")
NE = str(REPO / "data" / "natural_earth_normalized" / "ne_geography.parquet")

# Natural Earth kinds a question can name, grouped by how people talk about them.
KINDS = {
    "water": ["ocean", "sea", "gulf", "bay", "strait", "sound", "channel", "lagoon", "fjord"],
    "river": ["river"],
    "lake": ["lake", "reservoir"],
    "range": ["range/mtn", "foothills"],
    "area": ["desert", "plateau", "basin", "plain", "lowland", "valley", "depression", "delta", "wetlands",
             "peninsula", "island group"],
}
KIND_OF = {s: k for k, ss in KINDS.items() for s in ss}
NEAR_DEG = 0.6  # distances stored for pairs closer than this (buffers up to ~50 km)
# Countries whose questions only go to the held-out test split (spread over continents; Peru stays in training:
# it is the demo's example).
TEST_COUNTRIES = {"CL", "NG", "VN", "PL", "MX", "KE", "NZ", "TR", "CO", "SE", "MA", "PH"}


# ── relations ────────────────────────────────────────────────────────────────


def worker(cc):
    """One country: JSON with its divisions and their relations to Natural Earth features and neighbours."""
    import duckdb
    con = duckdb.connect()
    con.execute("LOAD spatial")
    import os
    con.execute(f"SET threads={os.environ.get('GAZET_MULTI_THREADS', '2')}")
    con.execute(f"SET memory_limit='{os.environ.get('GAZET_MULTI_MB', '2000')}MB'")
    name = "coalesce(names.common['en'], names.\"primary\")"
    con.execute(f"""CREATE TEMP TABLE d AS SELECT id, {name} AS name, names."primary" AS local, subtype, region,
        geometry AS g, ST_XMin(geometry) x0, ST_XMax(geometry) x1, ST_YMin(geometry) y0, ST_YMax(geometry) y1
        FROM read_parquet('{DV}') WHERE country = '{cc}' AND subtype IN ('country', 'region')""")
    n_cty = con.execute(f"SELECT count(*) FROM read_parquet('{DV}') WHERE country = '{cc}' AND subtype = 'county'").fetchone()[0]
    regions = con.execute("SELECT region, id FROM d WHERE subtype = 'region'").fetchall()
    rnd = random.Random(cc)
    # Counties: all of a small country; otherwise those of a few regions (each region holds 3-150 counties).
    if n_cty <= 400:
        cond = "TRUE"
    else:
        sizes = dict(con.execute(f"SELECT region, count(*) FROM read_parquet('{DV}') WHERE country = '{cc}' AND subtype = 'county' GROUP BY 1").fetchall())
        ok = [r for r, _ in regions if 3 <= sizes.get(r, 0) <= 150]
        pick = rnd.sample(ok, min(5, len(ok)))
        cond = "region IN (" + ",".join(f"'{r}'" for r in pick) + ")" if pick else "FALSE"
    con.execute(f"""INSERT INTO d SELECT id, {name}, names."primary", subtype, region, geometry,
        ST_XMin(geometry), ST_XMax(geometry), ST_YMin(geometry), ST_YMax(geometry)
        FROM read_parquet('{DV}') WHERE country = '{cc}' AND subtype = 'county' AND {cond}""")
    bb = con.execute("SELECT min(x0), max(x1), min(y0), max(y1) FROM d WHERE subtype = 'country'").fetchone()
    if bb[0] is None:
        return {"country": cc, "divisions": []}
    sts = ",".join(f"'{s}'" for s in KIND_OF)
    m = NEAR_DEG
    con.execute(f"""CREATE TEMP TABLE n AS SELECT id, coalesce(names.en, names."primary") AS name, subtype, geometry AS g,
        ST_XMin(geometry) x0, ST_XMax(geometry) x1, ST_YMin(geometry) y0, ST_YMax(geometry) y1
        FROM read_parquet('{NE}') WHERE subtype IN ({sts})
        AND ST_XMax(geometry) >= {bb[0] - m} AND ST_XMin(geometry) <= {bb[1] + m}
        AND ST_YMax(geometry) >= {bb[2] - m} AND ST_YMin(geometry) <= {bb[3] + m}""")
    # Distance (degrees) between each division and each nearby feature; 0 = intersects.
    near = con.execute(f"""SELECT d.id, n.id, CASE WHEN ST_Intersects(d.g, n.g) THEN 0.0 ELSE ST_Distance(d.g, n.g) END
        FROM d JOIN n ON n.x1 >= d.x0 - {m} AND n.x0 <= d.x1 + {m} AND n.y1 >= d.y0 - {m} AND n.y0 <= d.y1 + {m}
        WHERE d.subtype != 'country' OR ST_Intersects(d.g, n.g)""").fetchall()
    # Neighbouring countries, and which of our regions/counties touch each.
    con.execute(f"""CREATE TEMP TABLE e AS SELECT id, country, geometry AS g FROM read_parquet('{DV}')
        WHERE subtype = 'country' AND country != '{cc}' AND ST_XMax(geometry) >= {bb[0]} AND ST_XMin(geometry) <= {bb[1]}
        AND ST_YMax(geometry) >= {bb[2]} AND ST_YMin(geometry) <= {bb[3]}""")
    nb = [r[0] for r in con.execute("SELECT e.id FROM e, d WHERE d.subtype = 'country' AND ST_Touches(d.g, e.g)").fetchall()]
    touch = con.execute("""SELECT d.id, e.id FROM d JOIN e ON ST_Touches(d.g, e.g)
        WHERE d.subtype != 'country' AND e.id IN (SELECT unnest(?::VARCHAR[]))""", [nb]).fetchall() if nb else []
    # Containment: regions in the country, counties in their region (gazet's "within" semantics).
    within = con.execute("""SELECT c.id, p.id FROM d c JOIN d p ON (c.subtype = 'region' AND p.subtype = 'country')
        OR (c.subtype = 'county' AND p.subtype = 'region' AND c.region = p.region) WHERE ST_Within(c.g, p.g)""").fetchall()
    divs = con.execute("SELECT id, name, local, subtype, region, (x1 - x0) * (y1 - y0) FROM d").fetchall()
    nat = con.execute("SELECT id, name, subtype FROM n").fetchall()
    areas = dict(con.execute("SELECT id, ST_Area(g) FROM d").fetchall())
    dist = defaultdict(dict)
    for a, b, x in near:
        if x <= m:
            dist[a][b] = round(x, 5)
    tch, win = defaultdict(list), defaultdict(list)
    for a, b in touch:
        tch[a].append(b)
    for a, b in within:
        win[a].append(b)
    return {
        "country": cc, "neighbours": nb,
        "divisions": [{"id": i, "name": nm, "local": lo, "subtype": st, "region": rg, "area": areas[i],
                       "near": dist.get(i, {}), "touches": tch.get(i, []), "within": win.get(i, [])}
                      for i, nm, lo, st, rg, _ in divs],
        "natural": {i: {"name": nm, "subtype": st} for i, nm, st in nat},
    }


def relations(only):
    import duckdb

    import harness
    REL.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    ccs = [r[0] for r in con.execute(f"SELECT DISTINCT country FROM read_parquet('{DV}') WHERE subtype = 'country' ORDER BY 1").fetchall()]
    for cc in only or ccs:
        f = REL / f"{cc}.json"
        if f.exists():
            continue
        rc, out, err, killed = harness.run_capped([sys.executable, "-I", __file__, "worker", cc], "", 900, 3500)
        if killed or rc != 0:
            print(f"{cc}: FAILED {killed or err.strip().splitlines()[-1:]}", flush=True)
            continue
        f.write_text(out)
        r = json.loads(out)
        print(f"{cc}: {len(r['divisions'])} divisions, {len(r.get('natural', {}))} natural, {len(r.get('neighbours', []))} neighbours", flush=True)


# ── question generation ──────────────────────────────────────────────────────

SUB_WORDS = {
    "region": ["regions", "regions", "states", "provinces", "departments"],
    "county": ["counties", "districts", "districts", "municipalities", "provinces"],
    "country": ["countries", "countries", "nations"],
}
# {S} subtype word, {D} container, {N} natural feature, {E} neighbouring country.
VERBS = {
    "water": ["that touch the {N}", "on the {N}", "that border the {N}", "along the {N}", "on the {N} coast",
              "with a coastline on the {N}", "facing the {N}", "next to the {N}", "that reach the {N}", "bordering the {N}"],
    "river": ["along the {N}", "that the {N} flows through", "crossed by the {N}", "on the {N}", "that the {N} passes through",
              "touching the {N}", "that the {N} runs through", "intersecting the {N}"],
    "lake": ["on {N}", "that touch {N}", "bordering {N}", "on the shore of {N}", "around {N}", "touching {N}"],
    "range": ["in the {N}", "that the {N} cross", "overlapping the {N}", "that touch the {N}", "covered by the {N}",
              "in the {N} mountains"],
    "area": ["in the {N}", "that overlap the {N}", "that touch the {N}", "covering part of the {N}", "inside the {N}"],
}
NEG = {
    "water": ["that don't touch the {N}", "not on the {N}", "without a coast on the {N}", "that do not border the {N}",
              "away from the {N}", "not touching the {N}"],
    "river": ["that the {N} does not cross", "not on the {N}", "away from the {N}", "that the {N} doesn't flow through"],
    "lake": ["not on {N}", "that don't touch {N}", "not bordering {N}"],
    "range": ["outside the {N}", "not in the {N}", "that the {N} does not cross"],
    "area": ["outside the {N}", "not in the {N}", "that don't overlap the {N}"],
}


def _fold(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().strip()


def nat_surface(name, subtype, rnd):
    """How a person writes a Natural Earth feature's name."""
    s = name
    if s.isupper():
        s = s.title() if rnd.random() < 0.8 else s
    if subtype == "ocean" and re.match(r"(?i)(north|south) (pacific|atlantic)", s) and rnd.random() < 0.5:
        s = s.split(" ", 1)[1]  # "Pacific Ocean" means both halves
    if subtype in ("lake", "reservoir") and not re.search(r"(?i)\b(lake|lago|lac|loch|reservoir|sea)\b", s) and rnd.random() < 0.6:
        s = f"Lake {s}"
    if subtype == "river" and not re.search(r"(?i)\briver\b", s) and rnd.random() < 0.4:
        s = f"{s} River" if rnd.random() < 0.7 else f"River {s}"
    return s


def nat_group(nid, rel_nat, surface):
    """IDs a name stands for: features of the same kind and name ("Pacific Ocean" = North + South Pacific)."""
    f = rel_nat[nid]
    key = _fold(re.sub(r"(?i)^(north|south) ", "", f["name"])) if _fold(surface) in ("pacific ocean", "atlantic ocean", "pacific", "atlantic") else _fold(f["name"])
    out = []
    for i, g in rel_nat.items():
        k = _fold(re.sub(r"(?i)^(north|south) ", "", g["name"] or "")) if key in ("pacific ocean", "atlantic ocean") else _fold(g["name"])
        if g["name"] and k == key and KIND_OF.get(g["subtype"]) == KIND_OF[f["subtype"]]:
            out.append(i)
    return out


def div_surface(div, rnd):
    s = div["name"]
    if div["local"] and div["local"] != s and rnd.random() < 0.15:
        s = div["local"]
    return s


def noise(q, rnd):
    """Surface variety: case, articles, punctuation, lead-ins."""
    r = rnd.random()
    if r < 0.12:
        q = q.lower()
    elif r < 0.17:
        q = q[0].upper() + q[1:]
    if rnd.random() < 0.15:
        q = re.sub(r"\bthe ", "", q)
    if rnd.random() < 0.2:
        q = rnd.choice(["show me ", "show ", "find ", "map ", "give me ", "list ", "which are the ", "get "]) + q
    if rnd.random() < 0.1:
        q += "?"
    return re.sub(r"\s+", " ", q).strip()


class Gen:
    def __init__(self, seed):
        self.rnd = random.Random(seed)
        self.rels = {}
        for f in sorted(REL.glob("*.json")):
            r = json.loads(f.read_text())
            if r["divisions"]:
                self.rels[r["country"]] = r
        self.by_id = {}
        self.nat = {}
        for cc, r in self.rels.items():
            for d in r["divisions"]:
                d["country"] = cc
                self.by_id[d["id"]] = d
            self.nat.update(r["natural"])
        # Global relations: every country/region/county -> natural features it intersects (for whole-world queries).
        self.nat_hits = defaultdict(set)
        for d in self.by_id.values():
            for n, x in d["near"].items():
                if x == 0:
                    self.nat_hits[n].add(d["id"])
        import duckdb
        self.con = duckdb.connect()
        self.con.execute("LOAD spatial")

    # candidates exactly as gazet.api builds them
    def candidates(self, places):
        import pandas as pd

        from gazet.api import _per_source_limit
        from gazet.schemas import Place
        from gazet.search import search_candidates
        dfs = []
        for p in places:
            dfs.extend(search_candidates(self.con, Place(place=p), limit=_per_source_limit(len(places))))
        if not dfs:
            return None
        return (pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["source", "id"])
                .sort_values(["is_substring_match", "similarity", "admin_level"], ascending=[False, False, True])
                .reset_index(drop=True))

    def members(self, cc, sub, container):
        """Truth for a container: (country code) all of that subtype; (region id) counties within it."""
        r = self.rels[cc]
        if container is None:
            return {d["id"] for d in r["divisions"] if d["subtype"] == sub}
        return {d["id"] for d in r["divisions"] if d["subtype"] == sub and container in d["within"]}

    def hits(self, ids, nat_ids, m=0):
        return {i for i in ids if any(self.by_id[i]["near"].get(n, 9) <= m / 111320.0 for n in nat_ids)}


def _order(q, surfaces):
    """Places in the order they appear in the question."""
    ql = q.lower()
    pos = [(ql.find(s.lower()), s) for s in surfaces]
    if any(p < 0 for p, _ in pos):
        return None
    return [s for _, s in sorted(pos)]


def make_one(g, tid, cc, split_ok):
    """One example of template `tid` for country `cc`, or None."""
    rnd = g.rnd
    r = g.rels[cc]
    country = next(d for d in r["divisions"] if d["subtype"] == "country")
    regions = [d for d in r["divisions"] if d["subtype"] == "region"]
    counties = [d for d in r["divisions"] if d["subtype"] == "county"]
    nat = r["natural"]

    # container: the country (counties/regions of a country) or one region (its counties)
    if tid in ("seas_both", "countries_both"):
        sub, cont = "country", None
    elif counties and (rnd.random() < 0.45 or not regions):
        sub = "county"
        if len(counties) <= 400 and rnd.random() < 0.4:
            cont = None
        else:
            withreg = [d for d in regions if any(d["id"] in c["within"] for c in counties)]
            if not withreg:
                return None
            cont = rnd.choice(withreg)
    else:
        sub, cont = "region", None
    if sub != "country":
        pool = g.members(cc, sub, cont["id"] if cont else None)
        if len(pool) < 3:
            return None
    words = rnd.choice(SUB_WORDS[sub])
    D_div = cont or country
    D = div_surface(D_div, rnd)
    plan_base = {"find": "divisions", "subtype": [sub]}
    if cont:
        plan_base["within"] = {"ids": [cont["id"]]}
    else:
        plan_base["country"] = [cc]

    def pick_nat(kinds, min_hits=1, exclude=()):
        cands = [n for n in nat if nat[n]["name"] and KIND_OF[nat[n]["subtype"]] in kinds and n not in exclude
                 and len(g.hits(pool, [n])) >= min_hits]
        if not cands:
            return None
        n = rnd.choice(cands)
        return n, nat_surface(nat[n]["name"], nat[n]["subtype"], rnd)

    def phrase(kind, neg=False):
        return rnd.choice((NEG if neg else VERBS)[kind])

    meta = {"template_id": tid, "country": cc}
    if tid in ("nat_filter", "nat_exclude", "nat_rank", "nat_union", "nat_buffer"):
        kinds = ["river", "lake"] if tid == "nat_buffer" else rnd.choices(
            ["water", "river", "lake", "range", "area"], weights=[4, 3, 2, 2, 1.5])
        pk = pick_nat(kinds)
        if not pk:
            return None
        n, N = pk
        kind = KIND_OF[nat[n]["subtype"]]
        verb = phrase(kind, neg=tid == "nat_exclude").replace("{N}", N)
        if tid == "nat_buffer":
            km = rnd.choice([5, 10, 20, 25, 30, 50])
            verb = rnd.choice(["within {k} km of {N}", "less than {k} km from {N}", "{k} km or closer to {N}",
                               "within {k}km of {N}"]).format(k=km, N=N)
        templates = [f"{words} of {D} {verb}", f"{words} in {D} {verb}", f"{D} {words} {verb}",
                     f"which {words} of {D} are {verb}" if not verb.startswith("that") else f"which {words} of {D} {verb[5:]}",
                     f"{D}'s {words} {verb}"]
        if kind == "water" and tid == "nat_filter":
            templates += [f"coastal {words} of {D} on the {N}", f"{N} {words} of {D}", f"{words} of {D} with {N} coastline"]
        if kind == "river" and tid == "nat_filter":
            templates += [f"{words} of {D} the {N} flows through", f"where does the {N} cross {D}'s {words}"]
        q = rnd.choice(templates)
        if tid == "nat_rank":
            k = rnd.choice([1, 1, 2, 3, 3, 5])
            big = rnd.random() < 0.6
            adj = ("largest" if big else "smallest") if rnd.random() < 0.8 else ("biggest" if big else "tiniest")
            q = (f"the {adj} {words[:-1] if words.endswith('s') and k == 1 else words} of {D} {verb}" if k == 1
                 else f"the {k} {adj} {words} of {D} {verb}")
            q = q.replace("countie ", "county ").replace("municipalitie ", "municipality ")
        if tid == "nat_union":
            q = rnd.choice(["merge the ", "union of the ", "combine all ", "one shape of the ", "dissolve the "]) + \
                f"{words} of {D} {verb}"
        q = noise(q, rnd)
        places = _order(q, [D, N])
        if not places:
            return None
        cands = g.candidates(places)
        if cands is None or D_div["id"] not in set(cands["id"]):
            return None
        group = [i for i in nat_group(n, nat, N) if i in set(cands["id"])]
        if not group:
            return None
        m = km * 1000 if tid == "nat_buffer" else 0
        hit = g.hits(pool, group, m)
        truth = (pool - hit) if tid == "nat_exclude" else hit
        if not truth or truth == pool:
            return None  # the second place must matter
        expr = {"ids": group}
        if tid == "nat_buffer":
            expr = {"buffer": expr, "m": m}
        plan = dict(plan_base, **{("not_intersects" if tid == "nat_exclude" else "intersects"): expr})
        if tid == "nat_rank":
            plan["largest" if big else "smallest"] = k
            if k >= len(truth):
                return None
            truth = set(sorted(truth, key=lambda i: g.by_id[i]["area"], reverse=big)[:k])
        if tid == "nat_union":
            plan = {"union": plan}
    elif tid == "two_nat":
        a = pick_nat(["water", "river", "lake", "range", "area"])
        if not a:
            return None
        b = pick_nat(["water", "river", "lake", "range", "area"], exclude={a[0]})
        if not b or _fold(a[1]) == _fold(b[1]):
            return None
        (na, NA), (nb_, NB) = a, b
        conj = rnd.choice(["both {A} and {B}", "{A} and {B}", "{A} as well as {B}"]).format(A=NA, B=NB)
        q = noise(rnd.choice([f"{words} of {D} that touch {conj}", f"{words} of {D} touching {conj}",
                              f"which {words} in {D} touch {conj}", f"{D} {words} on {conj}"]), rnd)
        places = _order(q, [D, NA, NB])
        if not places:
            return None
        cands = g.candidates(places)
        if cands is None or D_div["id"] not in set(cands["id"]):
            return None
        cid = set(cands["id"])
        ga, gb = [i for i in nat_group(na, nat, NA) if i in cid], [i for i in nat_group(nb_, nat, NB) if i in cid]
        if not ga or not gb or set(ga) & set(gb):
            return None
        truth = g.hits(pool, ga) & g.hits(pool, gb)
        if not truth or truth == g.hits(pool, ga) or truth == g.hits(pool, gb):
            return None
        plan = dict(plan_base, intersects=[{"ids": ga}, {"ids": gb}])
    elif tid in ("border_country", "border_country_nat"):
        nbs = [e for e in r["neighbours"] if e in g.by_id and any(e in g.by_id[i]["touches"] for i in pool)]
        if not nbs:
            return None
        e = rnd.choice(nbs)
        E = div_surface(g.by_id[e], rnd)
        bset = {i for i in pool if e in g.by_id[i]["touches"]}
        if bset == pool:
            return None
        if tid == "border_country":
            q = rnd.choice([f"{words} of {D} that border {E}", f"{words} of {D} bordering {E}", f"which {words} of {D} touch {E}", f"{words} in {D} next to {E}", f"{words} of {D} adjacent to {E}",
                            f"{D} {words} that touch {E}"])
            surf = [D, E]
            truth = bset
            plan = dict(plan_base, touches={"ids": [e]})
        else:
            pk = pick_nat(["water", "river", "lake", "range"])
            if not pk:
                return None
            n, N = pk
            q = rnd.choice([f"{words} of {D} that border {E} and touch the {N}", f"{words} of {D} on both {E} and the {N}",
                            f"{words} of {D} bordering {E} {phrase(KIND_OF[nat[n]['subtype']]).replace('{N}', N)}"])
            surf = [D, E, N]
            truth = None
            plan = None
        q = noise(q, rnd)
        places = _order(q, surf)
        if not places:
            return None
        cands = g.candidates(places)
        cid = set(cands["id"]) if cands is not None else set()
        if D_div["id"] not in cid or e not in cid:
            return None
        if tid == "border_country_nat":
            group = [i for i in nat_group(n, nat, N) if i in cid]
            if not group:
                return None
            truth = bset & g.hits(pool, group)
            if not truth or truth == bset:
                return None
            plan = dict(plan_base, touches={"ids": [e]}, intersects={"ids": group})
    elif tid in ("seas_both", "countries_both"):
        if tid == "seas_both":
            # natural features that touch both this country and a neighbour
            nbs = [e for e in r["neighbours"] if e in g.by_id]
            if not nbs:
                return None
            e = rnd.choice(nbs)
            kind = rnd.choices(["water", "river", "lake", "range"], weights=[4, 3, 1, 2])[0]
            words_n = {"water": ["seas", "seas and oceans", "waters"], "river": ["rivers"], "lake": ["lakes"],
                       "range": ["mountain ranges", "mountains"]}[kind]
            subs = {"water": ["ocean", "sea"], "river": ["river"], "lake": ["lake"], "range": ["range/mtn"]}[kind]
            A, E = div_surface(country, rnd), div_surface(g.by_id[e], rnd)
            q = noise(rnd.choice([f"{rnd.choice(words_n)} that touch both {A} and {E}", f"{rnd.choice(words_n)} shared by {A} and {E}",
                                  f"which {rnd.choice(words_n)} do {A} and {E} share", f"{rnd.choice(words_n)} between {A} and {E}",
                                  f"{rnd.choice(words_n)} touching {A} and {E}"]), rnd)
            places = _order(q, [A, E])
            if not places or _fold(A) == _fold(E):
                return None
            cands = g.candidates(places)
            if cands is None or not {country["id"], e} <= set(cands["id"]):
                return None
            ha = {n for n, x in country["near"].items() if x == 0 and g.nat.get(n, {}).get("subtype") in subs}
            hb = {n for n, x in g.by_id[e]["near"].items() if x == 0 and g.nat.get(n, {}).get("subtype") in subs}
            truth = ha & hb
            if not truth:
                return None
            plan = {"find": "natural", "subtype": subs, "intersects": [{"ids": [country["id"]]}, {"ids": [e]}]}
            meta["truth_kind"] = "natural"
        else:
            ints = [n for n, x in country["near"].items() if x == 0 and n in nat and nat[n]["name"]
                    and KIND_OF[nat[n]["subtype"]] in ("water", "range")]
            if len(ints) < 2:
                return None
            na, nb_ = rnd.sample(ints, 2)
            NA, NB = nat_surface(nat[na]["name"], nat[na]["subtype"], rnd), nat_surface(nat[nb_]["name"], nat[nb_]["subtype"], rnd)
            if _fold(NA) == _fold(NB):
                return None
            q = noise(rnd.choice([f"countries that touch both the {NA} and the {NB}", f"countries on the {NA} and the {NB}",
                                  f"which countries border both {NA} and {NB}", f"nations along both the {NA} and the {NB}"]), rnd)
            places = _order(q, [NA, NB])
            if not places:
                return None
            cands = g.candidates(places)
            if cands is None:
                return None
            cid = set(cands["id"])
            ga, gb = [i for i in nat_group(na, nat, NA) if i in cid], [i for i in nat_group(nb_, nat, NB) if i in cid]
            if not ga or not gb or set(ga) & set(gb):
                return None
            ca = set().union(*(g.nat_hits[i] for i in ga))
            cb = set().union(*(g.nat_hits[i] for i in gb))
            truth = {i for i in ca & cb if g.by_id[i]["subtype"] == "country"}
            if not truth:
                return None
            plan = {"find": "divisions", "subtype": ["country"], "intersects": [{"ids": ga}, {"ids": gb}]}
    else:
        raise ValueError(tid)
    from gazet_plan import canonical
    return {"question": q, "places": places, "plan": canonical(plan), "truth": sorted(truth),
            "candidates": cands, **meta}


TEMPLATES = {"nat_filter": 30, "nat_exclude": 10, "nat_rank": 8, "nat_union": 6, "nat_buffer": 8, "two_nat": 10,
             "border_country": 12, "border_country_nat": 5, "seas_both": 6, "countries_both": 5}


def generate(n_total, seed, only=None, tag=""):
    from gazet import lm
    g = Gen(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    ccs = sorted(g.rels)
    tot = sum(TEMPLATES.values())
    files = {s: (OUT / f"{s}{tag}.jsonl").open("w") for s in ("train", "val", "test")}
    counts, seen = Counter(), set()
    for tid, w in TEMPLATES.items():
        if only and tid not in only:
            continue
        target = round(n_total * w / tot)
        made, tries = 0, 0
        while made < target and tries < target * 25:
            tries += 1
            cc = g.rnd.choice(ccs)
            try:
                ex = make_one(g, tid, cc, True)
            except Exception as e:  # noqa: BLE001 - a bad sample is skipped, but counted
                counts[f"error {type(e).__name__}"] += 1
                continue
            if not ex or ex["question"].lower() in seen:
                continue
            seen.add(ex["question"].lower())
            split = "test" if cc in TEST_COUNTRIES else ("val" if g.rnd.random() < 0.03 else "train")
            if split == "test" and counts[f"test {tid}"] >= 40:
                continue
            cands = ex.pop("candidates")
            msgs = lm.plan_messages(ex["question"], cands)
            rec = {**ex, "split": split, "plan_messages": msgs,
                   "places_messages": [{"role": "system", "content": lm._PLACES_SYSTEM_PROMPT},
                                       {"role": "user", "content": ex["question"]}]}
            files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[f"{split} {tid}"] += 1
            if split != "test":
                made += 1
        print(f"{tid}: {made} train/val in {tries} tries; test {counts[f'test {tid}']}", flush=True)
    for f in files.values():
        f.close()
    print(dict(sorted(counts.items())))


# ── paraphrases (OpenRouter) ─────────────────────────────────────────────────

PARA_MODEL = "deepseek/deepseek-v4.1-flash"
PARA_PROMPT = """Rewrite each geographic search query below the way a different person, or an AI agent calling a geocoding
tool, might type it. Vary the wording and structure a lot: questions, commands, terse keyword queries, longer sentences,
casual or formal tone, "X's Y", word order. Rules:
- Keep every place name listed for the query EXACTLY as written (same spelling and letters), and add no other place names.
- Keep the meaning exactly: the same kind of units (regions/provinces/states vs counties/districts/municipalities vs
  countries), the same relation (touch/border/along/in/cross/within N km), any negation, any number, largest/smallest, and
  merging into one shape.
- One rewrite per query, different from the original.
Return JSON only: {"<id>": "<rewrite>", ...}

QUERIES:
"""
NEG_RE = re.compile(r"(?i)\b(not|don't|doesn't|do not|does not|without|outside|away|never|no|excluding|except|lacking|aren't|isn't)\b")
RANK_RE = re.compile(r"(?i)\b(largest|biggest|smallest|tiniest|largest|top|least|most|bigger|smaller)\b")
UNION_RE = re.compile(r"(?i)\b(merge|merged|union|combine|combined|one shape|dissolve|dissolved|single)\b")


def _para_ok(orig, new, places):
    if not new or new.strip().lower() == orig.strip().lower() or len(new) > 3 * len(orig) + 40:
        return None
    for p in places:
        if p not in new:
            return None
    if sorted(re.findall(r"\d+", orig)) != sorted(re.findall(r"\d+", new)):
        return None
    for rx in (NEG_RE, RANK_RE, UNION_RE):
        if bool(rx.search(orig)) != bool(rx.search(new)):
            return None
    return _order(new, places)


def paraphrase(workers):
    from concurrent.futures import ThreadPoolExecutor

    import or_generate
    key = or_generate.api_key()
    cost = 0.0
    for split in ("train", "val", "test"):
        src = OUT / f"{split}.jsonl"
        rows = [json.loads(x) for x in src.read_text().splitlines()]
        base = [r for r in rows if not r.get("para")]
        batches = [base[i:i + 20] for i in range(0, len(base), 20)]

        def run(batch):
            listing = "\n".join(json.dumps({"id": str(k), "query": r["question"], "places": r["places"]}, ensure_ascii=False)
                                for k, r in enumerate(batch))
            resp = or_generate.call(PARA_MODEL, PARA_PROMPT + listing, None, None, key, temperature=0.9)
            try:
                text = resp["choices"][0]["message"]["content"]
                out = json.loads(text[text.find("{"):text.rfind("}") + 1])
            except Exception:  # noqa: BLE001
                out = {}
            return out, (resp.get("usage") or {}).get("cost", 0) or 0
        new, stats = [], Counter()
        with ThreadPoolExecutor(workers) as ex:
            for batch, (out, c) in zip(batches, ex.map(run, batches)):
                cost += c
                for k, r in enumerate(batch):
                    q = out.get(str(k))
                    places = _para_ok(r["question"], q, r["places"]) if isinstance(q, str) else None
                    stats["kept" if places else "dropped"] += 1
                    if not places:
                        continue
                    q = q.strip()
                    pm = json.loads(json.dumps(r["plan_messages"]))
                    u = pm[1]["content"]
                    pm[1]["content"] = u[:u.index("<USER_QUERY>")] + f"<USER_QUERY>\n{q}\n</USER_QUERY>\n"
                    new.append({**r, "question": q, "orig": r["question"], "places": places, "para": True, "plan_messages": pm,
                                "places_messages": [r["places_messages"][0], {"role": "user", "content": q}]})
        with src.open("w") as f:
            for r in base + new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {dict(stats)}; running cost ${cost:.2f}", flush=True)


# ── verification and export ──────────────────────────────────────────────────


def verify(n, split):
    """Compile sampled plans and run them on the full data; the returned features must equal the truth."""
    import duckdb

    from gazet.plan import compile_plan
    from gazet.sql import _normalize_ne_subtypes, _rewrite_data_paths
    rows = [json.loads(line) for line in (OUT / f"{split}.jsonl").read_text().splitlines()]
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["template_id"]].append(r)
    rnd = random.Random(3)
    con = duckdb.connect()
    con.execute("LOAD spatial")
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2500MB'")
    res = Counter()
    for t, rs in sorted(by_t.items()):
        for r in rnd.sample(rs, min(n, len(rs))):
            plan = r["plan"]["union"] if "union" in r["plan"] else r["plan"]  # a union is one merged row
            sql = _normalize_ne_subtypes(_rewrite_data_paths(compile_plan(plan)))
            sql = sql.replace("ST_AsGeoJSON(geometry) AS geometry", "NULL AS geometry")
            try:
                got = {x[0] for x in con.execute(sql).fetchall()}
            except Exception as e:  # noqa: BLE001
                print(t, "ERROR", str(e)[:120])
                res[t, "error"] += 1
                continue
            ok = got == set(r["truth"])
            res[t, "same" if ok else "different"] += 1
            if not ok:
                print(f"{t} DIFF {r['question']!r}: +{len(got - set(r['truth']))} -{len(set(r['truth']) - got)}", flush=True)
    for t in sorted(by_t):
        print(f"{t:20s} same {res[t, 'same']}  different {res[t, 'different']}  error {res[t, 'error']}")


def expand_twins(plan, cands_csv):
    """Natural Earth IDs -> every candidate row with the same name and subtype. The model sees identical rows (a
    river's segments, North/South "Pacific Ocean") and cannot tell them apart, so a name stands for all of them."""
    import csv
    rows = [r for r in csv.DictReader(io.StringIO(cands_csv)) if r["source"] == "natural_earth"]
    key = {r["id"]: (_fold(r["name"]), r["subtype"]) for r in rows}

    def fix(e):
        if isinstance(e, list):
            return [fix(x) for x in e]
        if not isinstance(e, dict):
            return e
        e = {k: fix(v) for k, v in e.items()}
        if "ids" in e and isinstance(e["ids"], list):
            out = []
            for i in e["ids"]:
                for j in [i] + [j for j in key if i in key and j != i and key[j] == key[i]]:
                    if j not in out:
                        out.append(j)
            e["ids"] = sorted(out, key=lambda j: list(key).index(j) if j in key else -1) if out != e["ids"] else out
        return e
    return fix(plan)


def fix_twins():
    """Apply expand_twins to the generated data; recompute the answers that depend on the whole group."""
    from gazet_plan import canonical
    g = Gen(0)
    for split in ("train", "val", "test"):
        p = OUT / f"{split}.jsonl"
        rows, n = [], Counter()
        for line in p.read_text().splitlines():
            r = json.loads(line)
            cands = r["plan_messages"][1]["content"].split("<CANDIDATES>")[1].split("</CANDIDATES>")[0].strip()
            new = canonical(expand_twins(r["plan"], cands))
            if new != r["plan"]:
                n["expanded"] += 1
                r["plan"] = new
                if r["template_id"] == "countries_both":
                    ga, gb = (x["ids"] for x in new["intersects"])
                    ca = set().union(*(g.nat_hits[i] for i in ga))
                    cb = set().union(*(g.nat_hits[i] for i in gb))
                    t = sorted(i for i in ca & cb if g.by_id[i]["subtype"] == "country")
                    n["truth changed"] += t != r["truth"]
                    r["truth"] = t
            rows.append(r)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(split, dict(n))


def relabel_country_within(plan, cands_csv):
    """{"within": {"ids": [<a country>]}} -> {"country": [<its code>]} (ST_Within drops coastal regions)."""
    import csv
    rows = {r["id"]: r for r in csv.DictReader(io.StringIO(cands_csv))}

    def fix(e):
        if isinstance(e, list):
            return [fix(x) for x in e]
        if not isinstance(e, dict):
            return e
        e = {k: fix(v) for k, v in e.items()}
        w = e.get("within")
        if (e.get("find") == "divisions" and isinstance(w, dict) and list(w) == ["ids"] and len(w["ids"]) == 1
                and "country" not in e and rows.get(w["ids"][0], {}).get("subtype") == "country"
                and e.get("subtype") in (["region"], ["county"])):
            e.pop("within")
            e["country"] = [rows[w["ids"][0]]["country"]]
        return e
    return fix(plan)


def export(out_dir, multi_repeat):
    """chatml/gazet_plan_v2: plan v1's records (country containers relabelled) + the multi-place data."""
    from gazet_plan import canonical
    from gazet.plan import dumps
    src = Path.home() / ".cache" / "gazet-exp" / "chatml_plan_v1"
    out = Path(out_dir)
    (out / "sql").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    stats = Counter()
    for split in ("train", "val"):
        recs = []
        for line in (src / "sql" / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["task"] == "plan":
                cands = r["messages"][1]["content"].split("<CANDIDATES>")[1].split("</CANDIDATES>")[0].strip()
                old = json.loads(r["messages"][2]["content"])
                new = canonical(expand_twins(relabel_country_within(old, cands), cands))
                if new != old:
                    stats[f"{split} relabelled {r['family']}"] += 1
                    stats[f"{split} relabelled (twins only)"] += canonical(relabel_country_within(old, cands)) == old
                    r["messages"][2]["content"] = dumps(new)
            recs.append(r)
        multi = [json.loads(x) for x in (OUT / f"{split}.jsonl").read_text().splitlines()]
        n0 = len(multi)
        multi = [m for m in multi if not (m["template_id"].startswith("border_country") and BORDER_RE.search(m["question"]))]
        stats[f"{split} dropped border phrasing"] = n0 - len(multi)
        shorts = [OUT / f for f in ("short_v1.jsonl", "short_v1_multi.jsonl") if (OUT / f).exists()]
        if split == "train" and shorts:
            for m in (json.loads(x) for p in shorts for x in p.read_text().splitlines() for _ in range(3)):
                recs.append({"messages": m["plan_messages"] + [{"role": "assistant", "content": dumps(m["plan"])}],
                             "task": "plan", "family": "short_" + m["family"]})
                recs.append({"messages": m["places_messages"] + [{"role": "assistant", "content": json.dumps(
                    {"places": [{"place": p} for p in m["places"]]}, ensure_ascii=False)}], "task": "places",
                             "family": "short_" + m["family"]})
        for m in multi:
            for _ in range((3 if m.get("short") else multi_repeat) if split == "train" else 1):
                recs.append({"messages": m["plan_messages"] + [{"role": "assistant", "content": dumps(m["plan"])}],
                             "task": "plan", "family": "multi_" + m["template_id"]})
                recs.append({"messages": m["places_messages"] + [{"role": "assistant", "content": json.dumps(
                    {"places": [{"place": p} for p in m["places"]]}, ensure_ascii=False)}],
                             "task": "places", "family": "multi_" + m["template_id"]})
        random.Random(17).shuffle(recs)
        (out / "sql" / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs))
        print(f"{split}: {len(recs):,} records, {Counter(x['task'] for x in recs)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    (out / "eval" / "gazet_compat.jsonl").write_text((src / "eval" / "gazet_compat.jsonl").read_text())
    test = [json.loads(x) for x in (OUT / "test.jsonl").read_text().splitlines()]
    with (out / "eval" / "multi_test.jsonl").open("w") as f:
        for i, m in enumerate(test):
            f.write(json.dumps({"id": f"plan:{i}", "messages": m["plan_messages"]}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"id": f"places:{i}", "messages": m["places_messages"]}, ensure_ascii=False) + "\n")
    print(f"eval: multi_test {2 * len(test)} prompts")


def answer(label, port):
    import os
    import time
    os.environ["LLAMA_SERVER_URL"] = f"http://127.0.0.1:{port}"
    from gazet import lm
    from gazet.plan import plan_schema
    test = [json.loads(x) for x in (OUT / "test.jsonl").read_text().splitlines()]
    path = OUT / f"pred_{label}.jsonl"
    t0 = time.time()
    done = {json.loads(x)["id"] for x in path.read_text().splitlines()} if path.exists() else set()
    with path.open("a") as f:
        for i, m in enumerate(test):
            if f"plan:{i}" in done:
                continue
            f.write(json.dumps({"id": f"places:{i}", "text": lm._llama_chat_complete(m["places_messages"])}, ensure_ascii=False) + "\n")
            ids = re.findall(r"^\w+,([^,]+),", m["plan_messages"][1]["content"].split("<CANDIDATES>")[1], re.M)[1:]
            text = lm._llama_chat_complete(m["plan_messages"], json_schema=plan_schema(ids))
            f.write(json.dumps({"id": f"plan:{i}", "text": text}, ensure_ascii=False) + "\n")
            f.flush()
    print(f"{label}: {len(test)} questions in {(time.time() - t0) / 60:.1f} min -> {path}")


def score(pred_file, label):
    """Held-out multi test: places equal (order-insensitive, case-insensitive), plan exact, same features (plans
    compiled and run on the full data)."""
    import duckdb

    from gazet.plan import compile_plan, parse_plan
    from gazet.sql import _normalize_ne_subtypes, _rewrite_data_paths
    test = [json.loads(x) for x in (OUT / "test.jsonl").read_text().splitlines()]
    preds = {}
    for line in Path(pred_file).read_text().splitlines():
        p = json.loads(line)
        preds[p["id"]] = p["text"]
    con = duckdb.connect()
    con.execute("LOAD spatial")
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2500MB'")
    res = defaultdict(Counter)
    rows = []
    for i, m in enumerate(test):
        t = m["template_id"] + (" (short)" if m.get("short") else " (para)" if m.get("para") else "")
        res[t]["n"] += 1
        try:
            got = [p["place"] for p in json.loads(preds.get(f"places:{i}", ""))["places"]]
        except Exception:  # noqa: BLE001
            got = []
        pl_ok = sorted(_fold(x) for x in got) == sorted(_fold(x) for x in m["places"])
        res[t]["places"] += pl_ok
        status = "invalid"
        try:
            plan = parse_plan(preds.get(f"plan:{i}", ""))
            from gazet_plan import canonical
            if canonical(plan) == m["plan"]:
                status = "exact"
            else:
                if "union" in plan and "union" in m["plan"]:  # a union is one merged row: compare what it merges
                    plan = plan["union"]
                sql = _normalize_ne_subtypes(_rewrite_data_paths(compile_plan(plan)))
                sql = sql.replace("ST_AsGeoJSON(geometry) AS geometry", "NULL AS geometry")
                import threading
                timer = threading.Timer(30, con.interrupt)  # a wrong plan can scan everything: count it as wrong
                timer.start()
                try:
                    got_ids = {x[0] for x in con.execute(sql).fetchall()}
                finally:
                    timer.cancel()
                status = "same" if got_ids == set(m["truth"]) else "different"
        except Exception as e:  # noqa: BLE001
            status = "invalid" if "plan" in type(e).__name__.lower() else "error"
        res[t][status] += 1
        rows.append({"i": i, "template_id": t, "question": m["question"], "places_ok": pl_ok, "places": got,
                     "plan_status": status, "pred": preds.get(f"plan:{i}"), "ref": m["plan"]})
    (OUT / f"score_{label}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tot = Counter()
    for c in res.values():
        tot.update(c)
    def line(name, c):
        n = c["n"]
        ok = c["exact"] + c["same"]
        return (f"{name:20s} n {n:4d}  places {100 * c['places'] / n:5.1f}  plan exact {100 * c['exact'] / n:5.1f}  "
                f"same features {100 * ok / n:5.1f}  invalid {c['invalid'] + c['error']}")
    print(f"== {label}")
    print(line("ALL", tot))
    c = Counter()
    for t, v in res.items():
        if "(short)" in t:
            c.update(v)
    if c["n"]:
        print(line("short names", c))
    for tag in ("templated", "paraphrased"):
        c = Counter()
        for t, v in res.items():
            if ("(short)" not in t) and ("(para)" in t) == (tag == "paraphrased"):
                c.update(v)
        if c["n"]:
            print(line(tag, c))
    for t in sorted(res):
        print(line(t, res[t]))


SUFFIX_RE = re.compile(r"^(.+?)\s+(Province|Region|County|District|Department|State|Municipality|Prefecture|Governorate|Oblast|"
                       r"Canton|Municipio|Provincia|Parish|Division|Raion|Rayon|Okrug|Krai|Voivodeship)$", re.I)
PREFIX_RE = re.compile(r"^(Province|Region|Department|Municipality|Departamento|Provincia|Región|Municipio|District|County|"
                       r"State|Governorate|Prefecture) (of|de|del|du|de la) (.+)$", re.I)
GENERIC = {"north", "south", "east", "west", "northern", "southern", "eastern", "western", "central", "upper", "lower",
           "capital", "north east", "north west", "south east", "south west", "northeast", "northwest", "southeast",
           "southwest", "far north", "far west", "greater", "new", "the"}
# "on the X border" / "border with X" read like gazet's "border between A and B" (a buffered shared border)
BORDER_RE = re.compile(r"(?i)(border with|on the [^,]+? border\b|\bborder (zone|area|region|strip)\b|shared border|along the border)")


def short_name(s):
    """'Loja Province' -> 'Loja', 'Province of Lima' -> 'Lima' (None when nothing specific is left)."""
    m = SUFFIX_RE.match(s)
    t = m.group(1) if m else (PREFIX_RE.match(s).group(3) if PREFIX_RE.match(s) else None)
    if not t or len(t) < 4 or _fold(t) in GENERIC or all(w in GENERIC for w in _fold(t).split()):
        return None
    return t


def _plan_ids(e):
    if isinstance(e, dict):
        return [i for k, v in e.items() for i in (v if k == "ids" else _plan_ids(v))]
    if isinstance(e, list):
        return [i for x in e for i in _plan_ids(x)]
    return []


def variants(max_v1, multi_only=False):
    """Short-name variants: the division named without its 'Province'/'Region' suffix, with fresh candidates from
    gazet's search. Kept only when every plan ID is still a candidate and the plan is unchanged."""
    from gazet import lm
    from gazet.plan import dumps
    from gazet_plan import canonical
    g = Gen(5)
    stats = Counter()

    def remake(q, places, plan, old):
        new_places = [short_name(p) or p for p in places]
        if new_places == places:
            return None
        nq = q
        for p, n in zip(places, new_places):
            if p != n:
                if p not in nq:
                    return None
                nq = nq.replace(p, n, 1)
        cands = g.candidates(new_places)
        if cands is None or not set(_plan_ids(plan)) <= set(cands["id"]):
            stats["anchor lost"] += 1
            return None
        msgs = lm.plan_messages(nq, cands)
        csv_ = msgs[1]["content"].split("<CANDIDATES>")[1].split("</CANDIDATES>")[0].strip()
        if canonical(expand_twins(plan, csv_)) != plan:
            stats["twins differ"] += 1
            return None
        stats["kept"] += 1
        return nq, _order(nq, new_places) or new_places, msgs

    for split in () if multi_only else ("train", "val", "test"):
        rows = [json.loads(x) for x in (OUT / f"{split}.jsonl").read_text().splitlines()]
        out = []
        for r in rows:
            if r.get("short"):
                continue
            res = remake(r["question"], r["places"], r["plan"], r)
            if res:
                nq, pl, msgs = res
                out.append({**r, "question": nq, "places": pl, "short": True, "plan_messages": msgs,
                            "places_messages": [r["places_messages"][0], {"role": "user", "content": nq}]})
        with (OUT / f"{split}.jsonl").open("a") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(out), dict(stats), flush=True)
    # plan v1: single-anchor questions that name a division with a suffix
    src = Path.home() / ".cache" / "gazet-exp" / "chatml_plan_v1" / "sql" / "train.jsonl"
    rnd = random.Random(9)
    lines = src.read_text().splitlines()
    rnd.shuffle(lines)
    n = 0
    with (OUT / ("short_v1_multi.jsonl" if multi_only else "short_v1.jsonl")).open("w") as f:
        for line in lines:
            if n >= max_v1:
                break
            r = json.loads(line)
            if r["task"] != "plan" or r["family"] in ("disambiguation",):
                continue
            plan = json.loads(r["messages"][2]["content"])
            ids = list(dict.fromkeys(_plan_ids(plan)))
            if (len(ids) != 1 or ids[0].startswith("ne_")) if not multi_only else not (2 <= len(ids) <= 3):
                continue
            u = r["messages"][1]["content"]
            q = u.split("<USER_QUERY>")[1].split("</USER_QUERY>")[0].strip()
            csv_ = u.split("<CANDIDATES>")[1].split("</CANDIDATES>")[0].strip()
            import csv as _csv
            byid = {row["id"]: row["name"] for row in _csv.DictReader(io.StringIO(csv_))}
            names = list(dict.fromkeys(byid.get(i) for i in ids))
            if not all(n and n in q for n in names) or not any(short_name(n) for n in names):
                continue
            plan = canonical(expand_twins(relabel_country_within(plan, csv_), csv_))
            res = remake(q, _order(q, names) or names, plan, r)
            if not res:
                continue
            nq, pl, msgs = res
            f.write(json.dumps({"question": nq, "places": pl, "plan": plan, "family": r["family"], "plan_messages": msgs,
                                "places_messages": [{"role": "system", "content": lm._PLACES_SYSTEM_PROMPT},
                                                    {"role": "user", "content": nq}]}, ensure_ascii=False) + "\n")
            n += 1
    print("plan v1 short variants", n, dict(stats))


def compat_corrected(labels):
    """gazet_compat with corrected references: where relabel_country_within / expand_twins change the reference plan,
    the reference is the compiled corrected plan (coastal regions kept, same-name features together)."""
    import gazet_compat as gc
    from gazet_plan import canonical, sql_to_plan
    from gazet.plan import compile_plan
    cands = {}
    for line in (gc.OUT / "prompts_plan.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["id"].startswith("sql:"):
            u = r["messages"][1]["content"]
            cands[r["id"][4:]] = u.split("<CANDIDATES>")[1].split("</CANDIDATES>")[0].strip()
    cache_p = OUT / "compat_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    for label in labels:
        rows = [json.loads(x) for x in (gc.OUT / f"{label}__sql.scored.jsonl").read_text().splitlines()]
        changed, fam = 0, defaultdict(list)
        for r in rows:
            if r["ref_error"]:
                continue
            ok = r["ok"]
            try:
                old = canonical(sql_to_plan(r["ref"]))
                new = canonical(expand_twins(relabel_country_within(old, cands[str(r["idx"])]), cands[str(r["idx"])]))
            except Exception:  # noqa: BLE001 - reference not expressible: keep the original verdict
                new = old = None
            if new != old:
                changed += 1
                ref = gc._run(compile_plan(new), cache)
                pred = gc._run(r["pred"], cache)
                ok = bool("wkt" in ref and "wkt" in pred and gc.same_geom(ref["wkt"], pred["wkt"]))
                cache_p.write_text(json.dumps(cache))
            fam[r["family"]].append(ok)
        allv = [x for v in fam.values() for x in v]
        print(f"{label:28s} corrected refs: same geometry {100 * sum(allv) / len(allv):5.1f} (n={len(allv)}, {changed} references corrected)")
        print("   " + "  ".join(f"{k[:10]} {100 * sum(v) / len(v):.0f}" for k, v in sorted(fam.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("cc")
    rl = sub.add_parser("relations")
    rl.add_argument("only", nargs="*")
    gn = sub.add_parser("generate")
    gn.add_argument("--n", type=int, default=9000)
    gn.add_argument("--seed", type=int, default=21)
    gn.add_argument("--only", nargs="*")
    gn.add_argument("--tag", default="")
    pp = sub.add_parser("paraphrase")
    pp.add_argument("--workers", type=int, default=8)
    vf = sub.add_parser("verify")
    vf.add_argument("--n", type=int, default=6)
    vf.add_argument("--split", default="train")
    sub.add_parser("fix-twins")
    va = sub.add_parser("variants")
    va.add_argument("--max-v1", type=int, default=3000)
    va.add_argument("--multi-only", action="store_true")
    cc = sub.add_parser("compat-corrected")
    cc.add_argument("labels", nargs="+")
    ex = sub.add_parser("export")
    ex.add_argument("--out", default=str(Path.home() / ".cache" / "gazet-exp" / "chatml_plan_v2"))
    ex.add_argument("--multi-repeat", type=int, default=1)
    an = sub.add_parser("answer")
    an.add_argument("label")
    an.add_argument("port")
    sc = sub.add_parser("score")
    sc.add_argument("pred_file")
    sc.add_argument("label")
    a = ap.parse_args()
    if a.cmd == "worker":
        print(json.dumps(worker(a.cc), ensure_ascii=False))
    elif a.cmd == "relations":
        relations(a.only)
    elif a.cmd == "generate":
        generate(a.n, a.seed, a.only, a.tag)
    elif a.cmd == "paraphrase":
        paraphrase(a.workers)
    elif a.cmd == "verify":
        verify(a.n, a.split)
    elif a.cmd == "compat-corrected":
        compat_corrected(a.labels)
    elif a.cmd == "variants":
        variants(a.max_v1, a.multi_only)
    elif a.cmd == "fix-twins":
        fix_twins()
    elif a.cmd == "export":
        export(a.out, a.multi_repeat)
    elif a.cmd == "answer":
        answer(a.label, a.port)
    elif a.cmd == "score":
        score(a.pred_file, a.label)
