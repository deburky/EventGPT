"""Build EventGPT token streams from the Home Credit - Credit Risk Model Stability competition data.

A "client" is one application (case_id), and its stream is every dated record the competition attaches to the
case, in time order, ending at the decision date. The stream opens with a profile event and then merges:

    profile   [EVT] name=profile dt=140 sex inc edu lang fam mar age ctype chan npay loan ann incamt
    app       [EVT] name=app dt=<b> status ctype chan rej canc pos bid tenor pamt dpd dpdtol     one per previous application
    cb_open   [EVT] name=cb_open dt=<b> st purpose cls inst per camt lim ninst                   one per bureau contract start
    cb_close  [EVT] name=cb_close dt=<b> cdpd ovmax novd                                          one per bureau contract closure
    cb_pay    [EVT] name=cb_pay dt=<b> n late pdpd pov                                            one per case and month, over contracts
    tax       [EVT] name=tax dt=<b> src tamt                                                      one per tax registry record
    dep       [EVT] name=dep dt=<b> closed damt                                                   one per deposit account
    card      [EVT] name=card dt=<b>                                                              one per debit card

The bureau's monthly payment table holds one row per contract and month (188M rows); like the geolocation
pings of MBD they are thinned to one event per case and month, which keeps the count of contracts paying,
the number late, the worst days past due and the overdue amount. Months after the decision date are left out
(scheduled instalments, or a spill-over month from the bureau's refresh lag). Every date is a day, so the
time gaps are multiples of 86,400 seconds; month-level records are stamped on the first of the month.

Categorical values are kept as the competition supplies them (masked labels included); the "not available"
token a55475b1 is a missing value. Amounts are binned by quantiles fitted on the pretraining folds, days past
due and counts by fixed edges; a missing number is <key>=na. Rare values of the high-cardinality keys fold into
<key>=other.

Folds: fold 0 is the test fold, every case decided in week 70 or later (about the last five months); folds 1
to 4 split the earlier cases by a hash of case_id. The pretraining streams hold folds 1 to 4, so the model
never sees a test-period case, and a test case's own events all precede its decision date.

The output layout is the one eventgpt.data.mbd writes (fold=<k>/, pretrain/, vocab/, test_fold.txt), plus
labels.parquet (case_id, fold, week, decision date, target, cut) for eventgpt.homecredit.probe, so
eventgpt.training.train, eventgpt.data.check_tokens and scripts/gcp.py work unchanged with MBD_OUT pointed here.

    MBD_OUT=data/home-credit/tokens HC_ROOT=<kaggle parquet_files/train> uv run python -m eventgpt.homecredit.prepare
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from eventgpt.data.mbd import OUT, write_pretrain
from eventgpt.data.vocab import EVT, USR, Vocab

RAW = Path(os.environ.get("HC_ROOT", "data/home-credit/raw"))  # the competition's parquet_files/train folder
MASK = "a55475b1"
N_FOLDS = 5
TEST_WEEK = 70  # cases decided in this week or later form the test fold
EVENT_WIDTH = 16  # [EVT] name dt plus up to 13 fields (the profile event)
N_QUANTILE_BINS = 16
MIN_COUNT = {"inst": 1000, "canc": 1000, "rej": 1000, "cls": 1000, "purpose": 1000, "npay": 1000, "tenor": 1000}
DPD_EDGES = [1, 6, 31, 61, 91, 181]  # days past due: 0, 1-5, 6-30, 31-60, 61-90, 91-180, over 180
COUNT_EDGES = [1, 4, 7, 13, 25, 37, 61]  # instalment counts
KINDS = ["profile", "app", "cb_open", "cb_close", "cb_pay", "tax", "dep", "card"]
PROFILE = KINDS.index("profile")

# field order per kind: categorical keys first, numeric keys after; each gets a slot after [EVT] name dt
FIELDS: dict[str, list[str]] = {
    "profile": ["sex", "inc", "edu", "lang", "fam", "mar", "age", "ctype", "chan", "npay", "loan", "ann", "incamt"],
    "app": ["status", "ctype", "chan", "rej", "canc", "pos", "bid", "tenor", "pamt", "dpd", "dpdtol"],
    "cb_open": ["st", "purpose", "cls", "inst", "per", "camt", "lim", "ninst"],
    "cb_close": ["cdpd", "ovmax", "novd"],
    "cb_pay": ["n", "late", "pdpd", "pov"],
    "tax": ["src", "tamt"],
    "dep": ["closed", "damt"],
    "card": [],
}
CAT_KEYS = ["sex", "inc", "edu", "lang", "fam", "mar", "age", "ctype", "chan", "npay", "status", "rej", "canc", "pos", "bid", "tenor"]
CAT_KEYS += ["st", "purpose", "cls", "inst", "per", "n", "late", "src", "closed"]
NUM_KEYS: dict[str, str] = {  # key -> binning rule
    "loan": "quantile",
    "ann": "quantile",
    "incamt": "quantile",
    "pamt": "quantile",
    "dpd": "dpd",
    "dpdtol": "dpd",
    "camt": "quantile",
    "lim": "quantile",
    "ninst": "count",
    "cdpd": "dpd",
    "ovmax": "quantile",
    "novd": "count",
    "pdpd": "dpd",
    "pov": "quantile",
    "tamt": "quantile",
    "damt": "quantile",
}
VAL_SHARE = 0.02


# -------------------------------------------------------------------------------
# Reading: one wide event table in a file-backed DuckDB
# -------------------------------------------------------------------------------
def src(pattern: str) -> str:
    """Table function of DuckDB for one raw table; shards are unioned by name."""
    return f"read_parquet('{RAW}/{pattern}', union_by_name=true)"


def _select(kind: str, t: str, fields: dict[str, str], from_sql: str) -> str:
    """One SELECT of the event union with every wide column, NULL for the keys this kind does not carry."""
    cols = ["case_id", f"{t} AS t", f"{KINDS.index(kind)} AS kind"]
    for key in CAT_KEYS:
        cols.append(f"({fields[key]})::VARCHAR AS {key}" if key in fields else f"NULL::VARCHAR AS {key}")
    for key in NUM_KEYS:
        cols.append(f"({fields[key]})::DOUBLE AS {key}" if key in fields else f"NULL::DOUBLE AS {key}")
    return f"SELECT {', '.join(cols)} FROM {from_sql}"


def build_events(con: duckdb.DuckDBPyConnection, max_cases: int | None, seed: int) -> None:
    """Create tables cases and events (every dated record of every case, with the profile event) in con."""
    sample = f"USING SAMPLE {max_cases} ROWS (reservoir, {seed})" if max_cases else ""
    con.execute(f"""
        CREATE OR REPLACE TABLE cases AS
        SELECT case_id, WEEK_NUM AS week, TRY_CAST(date_decision AS DATE) AS dd, target,
               CASE WHEN WEEK_NUM >= {TEST_WEEK} THEN 0 ELSE 1 + (hash(case_id) % 4)::INT END AS fold
        FROM {src("train_base.parquet")} {sample}
    """)
    day = "epoch(({d})::TIMESTAMP)"
    dpd_pay = "coalesce(a.pmts_dpd_1073P, a.pmts_dpd_303P)"
    ov_pay = "coalesce(a.pmts_overdue_1140A, a.pmts_overdue_1152A)"
    month_pay = "make_date(coalesce(a.pmts_year_1139T, a.pmts_year_507T)::INT, coalesce(a.pmts_month_158T, a.pmts_month_706T)::INT, 1)"
    parts = [
        _select(
            "app",
            day.format(d="TRY_CAST(a.creationdate_885D AS DATE)"),
            {
                "status": "a.status_219L",
                "ctype": "a.credtype_587L",
                "chan": "a.inittransactioncode_279L",
                "rej": f"NULLIF(a.rejectreason_755M, '{MASK}')",
                "canc": f"NULLIF(a.cancelreason_3545846M, '{MASK}')",
                "pos": f"NULLIF(a.postype_4733339M, '{MASK}')",
                "bid": "a.isbidproduct_390L::INT",
                "tenor": "a.tenor_203L::INT",
                "pamt": "a.credamount_590A",
                "dpd": "a.actualdpd_943P",
                "dpdtol": "a.maxdpdtolerance_577P",
            },
            f"{src('train_applprev_1_*.parquet')} a JOIN cases c USING (case_id) WHERE a.creationdate_885D IS NOT NULL",
        ),
        _select(
            "cb_open",
            day.format(d="TRY_CAST(a.dateofcredstart_739D AS DATE)"),
            {
                "st": "'active'",
                "purpose": f"NULLIF(a.purposeofcred_426M, '{MASK}')",
                "cls": f"NULLIF(a.classificationofcontr_13M, '{MASK}')",
                "inst": f"NULLIF(a.financialinstitution_591M, '{MASK}')",
                "per": "a.periodicityofpmts_837L::INT",
                "camt": "a.totalamount_996A",
                "lim": "a.credlmt_935A",
                "ninst": "a.numberofinstls_320L",
            },
            f"{src('train_credit_bureau_a_1_*.parquet')} a JOIN cases c USING (case_id) WHERE a.dateofcredstart_739D IS NOT NULL",
        ),
        _select(
            "cb_open",
            day.format(d="TRY_CAST(a.dateofcredstart_181D AS DATE)"),
            {
                "st": "'closed'",
                "purpose": f"NULLIF(a.purposeofcred_874M, '{MASK}')",
                "cls": f"NULLIF(a.classificationofcontr_400M, '{MASK}')",
                "inst": f"NULLIF(a.financialinstitution_382M, '{MASK}')",
                "per": "a.periodicityofpmts_1102L::INT",
                "camt": "a.totalamount_6A",
                "lim": "a.credlmt_230A",
                "ninst": "a.numberofinstls_229L",
            },
            f"{src('train_credit_bureau_a_1_*.parquet')} a JOIN cases c USING (case_id) WHERE a.dateofcredstart_181D IS NOT NULL",
        ),
        _select(
            "cb_close",
            day.format(d="TRY_CAST(a.dateofrealrepmt_138D AS DATE)"),
            {"cdpd": "a.dpdmax_757P", "ovmax": "a.overdueamountmax_35A", "novd": "a.numberofoverdueinstlmax_1151L"},
            f"{src('train_credit_bureau_a_1_*.parquet')} a JOIN cases c USING (case_id) "
            "WHERE a.dateofrealrepmt_138D IS NOT NULL AND TRY_CAST(a.dateofrealrepmt_138D AS DATE) <= c.dd",
        ),
        _select(
            "cb_pay",
            day.format(d="m"),
            {
                "n": "CASE WHEN n >= 10 THEN '10+' WHEN n >= 6 THEN '6-9' ELSE n::VARCHAR END",
                "late": "CASE WHEN late >= 3 THEN '3+' ELSE late::VARCHAR END",
                "pdpd": "pdpd",
                "pov": "pov",
            },
            f"""(SELECT a.case_id, {month_pay} AS m, count(*) AS n, count(*) FILTER (WHERE {dpd_pay} > 0) AS late,
                        max({dpd_pay}) AS pdpd, sum({ov_pay}) AS pov
                 FROM {src("train_credit_bureau_a_2_*.parquet")} a JOIN cases c USING (case_id)
                 WHERE coalesce(a.pmts_year_1139T, a.pmts_year_507T) IS NOT NULL
                   AND coalesce(a.pmts_month_158T, a.pmts_month_706T) IS NOT NULL
                   AND {month_pay} <= c.dd
                 GROUP BY 1, 2) p""",
        ),
        _select(
            "tax",
            day.format(d="TRY_CAST(a.recorddate_4527225D AS DATE)"),
            {"src": "'a'", "tamt": "a.amount_4527230A"},
            f"{src('train_tax_registry_a_1.parquet')} a JOIN cases c USING (case_id) WHERE a.recorddate_4527225D IS NOT NULL",
        ),
        _select(
            "tax",
            day.format(d="TRY_CAST(a.deductiondate_4917603D AS DATE)"),
            {"src": "'b'", "tamt": "a.amount_4917619A"},
            f"{src('train_tax_registry_b_1.parquet')} a JOIN cases c USING (case_id) WHERE a.deductiondate_4917603D IS NOT NULL",
        ),
        _select(
            "tax",
            day.format(d="TRY_CAST(a.processingdate_168D AS DATE)"),
            {"src": "'c'", "tamt": "a.pmtamount_36A"},
            f"{src('train_tax_registry_c_1.parquet')} a JOIN cases c USING (case_id) WHERE a.processingdate_168D IS NOT NULL",
        ),
        _select(
            "dep",
            day.format(d="TRY_CAST(a.openingdate_313D AS DATE)"),
            {"closed": "(a.contractenddate_991D IS NOT NULL)::INT", "damt": "a.amount_416A"},
            f"{src('train_deposit_1.parquet')} a JOIN cases c USING (case_id) WHERE a.openingdate_313D IS NOT NULL",
        ),
        _select(
            "card",
            day.format(d="TRY_CAST(a.openingdate_857D AS DATE)"),
            {},
            f"{src('train_debitcard_1.parquet')} a JOIN cases c USING (case_id) WHERE a.openingdate_857D IS NOT NULL",
        ),
    ]
    con.execute("CREATE OR REPLACE TABLE events_raw AS " + "\nUNION ALL\n".join(parts))
    # the profile event is stamped at the case's first event, or the day before the decision for a case without any
    profile = _select(
        "profile",
        f"coalesce(f.t, {day.format(d='c.dd - INTERVAL 1 DAY')})",
        {
            "sex": "p.sex_738L",
            "inc": "p.incometype_1044T",
            "edu": f"NULLIF(p.education_927M, '{MASK}')",
            "lang": f"NULLIF(p.language1_981M, '{MASK}')",
            "fam": "p.familystate_447L",
            "mar": f"NULLIF(cb.maritalst_385M, '{MASK}')",
            "age": "(floor(datediff('day', TRY_CAST(p.birth_259D AS DATE), c.dd) / 365.25 / 5) * 5)::INT",
            "ctype": "s.credtype_322L",
            "chan": "s.inittransactioncode_186L",
            "npay": "s.pmtnum_254L::INT",
            "loan": "s.credamount_770A",
            "ann": "s.annuity_780A",
            "incamt": "p.mainoccupationinc_384A",
        },
        f"""cases c
            LEFT JOIN (SELECT case_id, min(t) AS t FROM events_raw GROUP BY 1) f USING (case_id)
            LEFT JOIN (SELECT * FROM {src("train_person_1.parquet")} WHERE num_group1 = 0) p USING (case_id)
            LEFT JOIN {src("train_static_cb_0.parquet")} cb USING (case_id)
            LEFT JOIN {src("train_static_0_*.parquet")} s USING (case_id)""",
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE events AS
        SELECT e.*, c.fold FROM (SELECT * FROM events_raw UNION ALL {profile}) e JOIN cases c USING (case_id)
    """)
    con.execute("DROP TABLE events_raw")


# -------------------------------------------------------------------------------
# Vocabulary and bins, fitted on the pretraining folds
# -------------------------------------------------------------------------------
def fit(con: duckdb.DuckDBPyConnection, folds: list[int]) -> tuple[Vocab, dict[str, list[float]]]:
    """Categorical value tables with rare values folded into other, and bin edges per numeric key."""
    where = f"fold IN ({', '.join(map(str, folds))})"
    cat: dict[str, list[object]] = {"name": list(KINDS)}
    for key in CAT_KEYS:
        floor = MIN_COUNT.get(key, 1)
        rows = con.execute(f"SELECT {key}, count(*) FROM events WHERE {where} AND {key} IS NOT NULL GROUP BY 1").fetchall()
        vals: list[object] = [v for v, n in rows if n >= floor]
        cat[key] = [*vals, "other"] if len(vals) < len(rows) else vals
    edges: dict[str, list[float]] = {}
    qs = [float(q) for q in np.linspace(0, 1, N_QUANTILE_BINS + 1)[1:-1]]
    for key, rule in NUM_KEYS.items():
        if rule == "quantile":
            got = con.execute(f"SELECT quantile_cont({key}, ?::DOUBLE[]) FROM events WHERE {where} AND {key} > 0", [qs]).fetchone()
            edges[key] = sorted(set(float(x) for x in (got[0] if got and got[0] else [])))
        elif rule == "dpd":
            edges[key] = [float(x) for x in DPD_EDGES]
        else:
            edges[key] = [float(x) for x in COUNT_EDGES]
        cat[key] = (
            ["zero", *[f"b{j}" for j in range(1, len(edges[key]) + 1)], "na"]
            if rule == "quantile"
            else [*[f"b{j}" for j in range(len(edges[key]) + 1)], "na"]
        )
    loans = con.execute(f"SELECT loan FROM events WHERE {where} AND loan > 0").fetchnumpy()["loan"]
    return Vocab.fit(cat, np.asarray(loans, dtype=np.float64)), edges


# -------------------------------------------------------------------------------
# Tokenizing
# -------------------------------------------------------------------------------
def _cat_codes(vocab: Vocab, key: str, col: pa.ChunkedArray) -> np.ndarray:
    """Token id per row of a string column; NULL stays [PAD] (dropped), unknown values become <key>=other."""
    arr = pc.dictionary_encode(col.combine_chunks()) if col.num_chunks != 1 else pc.dictionary_encode(col.chunk(0))
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    words = arr.dictionary.to_pylist()
    other = vocab.structured.get(f"{key}=other", vocab.id("[PAD]"))
    lut = np.array([vocab.structured.get(f"{key}={w}", other) for w in words] + [0], dtype=np.int64)
    idx = arr.indices.fill_null(len(words)).to_numpy(zero_copy_only=False).astype(np.int64)
    return lut[idx]


def _num_codes(vocab: Vocab, key: str, col: pa.ChunkedArray, edges: list[float], rule: str) -> np.ndarray:
    """Token id per row of a numeric column, binned by its edges; NULL and NaN become <key>=na."""
    x = col.to_numpy(zero_copy_only=False).astype(np.float64)
    missing = ~np.isfinite(x)
    b = np.digitize(np.nan_to_num(x, nan=0.0), edges)  # 0 .. len(edges)
    if rule == "quantile":
        labels = ["zero"] + [f"b{j}" for j in range(1, len(edges) + 1)]
        tok = np.array([vocab.id(f"{key}={lab}") for lab in labels])[np.where(x <= 0, 0, np.maximum(b, 1))]
    else:
        tok = np.array([vocab.id(f"{key}=b{j}") for j in range(len(edges) + 1)])[b]
    return np.where(missing, vocab.id(f"{key}=na"), tok)


def tokenize(tab: pa.Table, vocab: Vocab, edges: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Fill the (events, EVENT_WIDTH) grid of a fold's events and drop its padding.

    Returns:
        (flat token ids, token offset of each event's [EVT] within the flat array)
    """
    n = tab.num_rows
    pad = vocab.id("[PAD]")
    mat = np.full((n, EVENT_WIDTH), pad, dtype=np.int32)
    kind = tab["kind"].to_numpy()
    case = tab["case_id"].to_numpy()
    t = tab["t"].to_numpy()
    first = np.ones(n, dtype=bool)
    first[1:] = (case[1:] != case[:-1]) | (kind[:-1] == PROFILE)
    gap = np.diff(t, prepend=0.0)
    gap[first] = 1e12  # a case's first event, and the profile, land in the top bucket
    mat[:, 0] = vocab.id(EVT)
    name_lut = np.array([vocab.id(f"name={k}") for k in KINDS])
    mat[:, 1] = name_lut[kind]
    mat[:, 2] = vocab.dt_tokens(gap)
    codes: dict[str, np.ndarray] = {}
    for key in CAT_KEYS:
        codes[key] = _cat_codes(vocab, key, tab[key])
    for key, rule in NUM_KEYS.items():
        codes[key] = _num_codes(vocab, key, tab[key], edges[key], rule)
    for k, fields in FIELDS.items():
        rows = kind == KINDS.index(k)
        for j, key in enumerate(fields):
            mat[rows, 3 + j] = codes[key][rows]
    keep = mat != pad
    counts = keep.sum(axis=1)
    flat = mat[keep].astype(np.int16)
    starts = np.cumsum(counts) - counts
    return flat, starts


def build_stream(tab: pa.Table, vocab: Vocab, edges: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Tokenize a fold's events and insert [USR] at each case boundary: the arrays a fold directory holds."""
    flat, starts = tokenize(tab, vocab, edges)
    case = tab["case_id"].to_numpy()
    change = np.ones(len(case), dtype=bool)
    change[1:] = case[1:] != case[:-1]
    first_event = np.flatnonzero(change)
    stream = np.insert(flat, starts[first_event], np.int16(vocab.id(USR)))
    shift = np.cumsum(change)
    return {
        "tokens": stream,
        "client_ids": case[first_event].astype(str),
        "client_events": np.append(first_event, len(case)).astype(np.int64),
        "event_start": (starts + shift).astype(np.int64),
        "event_time": tab["t"].to_numpy().astype(np.int64),
    }


def load_fold(con: duckdb.DuckDBPyConnection, k: int) -> pa.Table:
    """Every event of fold k, sorted by case then time, the profile event first within a case."""
    cols = ", ".join(["case_id", "t", "kind", *CAT_KEYS, *NUM_KEYS])
    res = con.execute(f"SELECT {cols} FROM events WHERE fold = {k} ORDER BY case_id, kind != {PROFILE}, t, kind")
    return getattr(res, "to_arrow_table", res.fetch_arrow_table)()  # the method was renamed in DuckDB 2


def write_fold(con: duckdb.DuckDBPyConnection, k: int, vocab: Vocab, edges: dict[str, list[float]]) -> dict[str, int]:
    """Tokenize one fold and write its stream and indexes."""
    tab = load_fold(con, k)
    arrays = build_stream(tab, vocab, edges)
    d = OUT / f"fold={k}"
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(d / f"{name}.npy", arr)
    return {"events": tab.num_rows, "cases": len(arrays["client_ids"]), "tokens": len(arrays["tokens"])}


def main() -> None:
    """Read the raw tables, fit the vocabulary, write every fold, the labels and the pretraining streams."""
    p = argparse.ArgumentParser()
    p.add_argument("--max-cases", type=int, default=0, help="random sample of cases, for a smoke test; 0 is every case")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory-limit", default="20GB")
    a = p.parse_args()
    if not RAW.exists():
        raise SystemExit(f"raw data not found at {RAW}; set HC_ROOT to the competition's parquet_files/train folder")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    db = OUT / "build.duckdb"
    if db.exists():
        db.unlink()
    con = duckdb.connect(str(db))
    con.execute(f"SET threads TO {a.threads}; SET memory_limit = '{a.memory_limit}'; SET preserve_insertion_order = false;")
    build_events(con, a.max_cases or None, a.seed)
    got = con.execute("SELECT (SELECT count(*) FROM cases), count(*) FROM events").fetchone()
    assert got is not None
    n_cases, n_events = got
    print(f"events table: {n_events:,} events for {n_cases:,} cases ({time.time() - t0:.0f}s)", flush=True)
    print(
        con.execute("SELECT fold, count(DISTINCT case_id) cases, count(*) events FROM events GROUP BY 1 ORDER BY 1")
        .df()
        .to_string(index=False)
    )
    pre = list(range(1, N_FOLDS))
    vocab, edges = fit(con, pre)
    vocab.save(OUT / "vocab")
    (OUT / "bins.json").write_text(json.dumps(edges, indent=1))
    print(f"vocab {vocab.size} tokens, fingerprint {vocab.fingerprint()[:12]} ({time.time() - t0:.0f}s)", flush=True)
    labels = con.execute(
        "SELECT case_id, fold, week, dd AS date_decision, target, epoch(dd::TIMESTAMP)::BIGINT AS cut FROM cases ORDER BY case_id"
    ).df()
    labels.to_parquet(OUT / "labels.parquet", index=False)
    for k in range(N_FOLDS):
        stats = write_fold(con, k, vocab, edges)
        print(f"fold {k}: {stats} ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "test_fold.txt").write_text("0\n")
    con.close()
    db.unlink()
    print(f"pretrain folds {pre}: {write_pretrain(pre)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
