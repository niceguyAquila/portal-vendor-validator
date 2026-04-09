"""
Local Streamlit dashboard: normalize portal CSV and vendor Excel to a shared schema, then reconcile.
"""

from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import streamlit as st

# --- Canonical schema ---
CANONICAL_SLOTS = (
    "transaction_id",
    "occurred_at",
    "amount",
    "currency",
    "txn_type",
)

PORTAL_DEFAULT_COLS: dict[str, str] = {
    "transaction_id": "Ticket #",
    "occurred_at": "Date",
    "amount": "Amount",
    "currency": "Currency",
    "txn_type": "Type",
}


def _none_option() -> str:
    return "(none)"


def slot_choices(columns: list[str]) -> list[str]:
    return [_none_option()] + list(columns)


def col_for_slot(choice: str) -> str | None:
    return None if choice == _none_option() else choice


def parse_local_path(raw: str) -> Path:
    """Strip quotes/whitespace; expand ~ for a path the user pasted in the sidebar."""
    t = raw.strip().strip('"').strip("'")
    return Path(t).expanduser()


def drop_separator_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def empty_cell(v: Any) -> bool:
        if pd.isna(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    mask = df.apply(
        lambda row: not all(empty_cell(v) for v in row),
        axis=1,
    )
    return df.loc[mask].reset_index(drop=True)


def load_portal_csv(source: Any) -> pd.DataFrame:
    """Load portal export: semicolon-separated, all strings first."""
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw), sep=";", dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(source, sep=";", dtype=str, keep_default_na=False)
    df = drop_separator_only_rows(df)
    return df


def _normalize_one_transaction_id(v: Any) -> Any:
    if pd.isna(v):
        return np.nan
    if isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v).is_integer():
        t = str(int(v))
    else:
        t = str(v).strip()
    if t == "" or t.lower() in ("nan", "none"):
        return np.nan
    t = re.sub(r"\s+", " ", t)
    return t


def normalize_transaction_id(series: pd.Series) -> pd.Series:
    # Do not chain .str after .replace(np.nan): dtype becomes object and .str breaks.
    return series.map(_normalize_one_transaction_id).astype("string")


def parse_amount(
    series: pd.Series,
    *,
    integer_round: bool,
    scale_multiplier: float = 1.0,
) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")

    def one(v: Any) -> float:
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.floating, np.integer)):
            x = float(v) * scale_multiplier
            return round(x) if integer_round else x
        t = str(v).strip()
        if t == "" or t.lower() in ("nan", "none"):
            return np.nan
        t = re.sub(r"[,\s]", "", t)
        try:
            x = float(t) * scale_multiplier
            return round(x) if integer_round else x
        except ValueError:
            return np.nan

    out = series.map(one)
    return pd.to_numeric(out, errors="coerce")


def parse_occurred_at(series: pd.Series, *, dayfirst: bool) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="datetime64[ns]")

    parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    return parsed


def raw_to_canonical(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    *,
    source_label: str,
    portal_date_dayfirst: bool,
    integer_amount_round: bool,
    amount_scale_multiplier: float = 1.0,
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    for slot in CANONICAL_SLOTS:
        col = mapping.get(slot)
        if not col or col not in df.columns:
            rows[slot] = pd.Series([np.nan] * len(df), index=df.index)
        else:
            rows[slot] = df[col]

    out = pd.DataFrame(
        {
            "transaction_id": normalize_transaction_id(rows["transaction_id"]),
            "occurred_at": parse_occurred_at(
                rows["occurred_at"], dayfirst=portal_date_dayfirst
            ),
            "amount": parse_amount(
                rows["amount"],
                integer_round=integer_amount_round,
                scale_multiplier=amount_scale_multiplier,
            ),
            "currency": rows["currency"]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
            "txn_type": rows["txn_type"]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
        }
    )
    out["source"] = source_label
    valid_id = out["transaction_id"].notna() & (out["transaction_id"].astype(str) != "")
    valid_dt = out["occurred_at"].notna()
    valid_amt = np.isfinite(out["amount"])
    out = out.loc[valid_id & valid_dt & valid_amt].reset_index(drop=True)
    return out


def guess_vendor_mapping(columns: list[str]) -> dict[str, str | None]:
    lower = [str(c).strip().lower() for c in columns]
    mapping: dict[str, str | None] = {s: None for s in CANONICAL_SLOTS}

    def pick(pred) -> str | None:
        for c, l in zip(columns, lower):
            if pred(l, c):
                return c
        return None

    mapping["transaction_id"] = pick(
        lambda l, c: (
            any(
                k in l
                for k in (
                    "ticket",
                    "reference",
                    "ref no",
                    "ref_no",
                    "order id",
                    "orderid",
                    "trans id",
                    "transaction id",
                    "trx id",
                )
            )
            or l in ("id", "ref")
        )
        and "fee" not in l
    )
    if mapping["transaction_id"] is None:
        mapping["transaction_id"] = pick(
            lambda l, c: l.endswith(" id") and "member" not in l and "login" not in l
        )

    mapping["occurred_at"] = pick(
        lambda l, c: any(
            k in l for k in ("date", "time", "created", "timestamp", "datetime")
        )
    )

    mapping["amount"] = pick(
        lambda l, c: ("amount" in l or l in ("nominal", "value", "total"))
        and "fee" not in l
    )

    mapping["currency"] = pick(lambda l, c: "currency" in l or l == "ccy")

    mapping["txn_type"] = pick(
        lambda l, c: l in ("type", "txn type", "transaction type", "direction")
    )

    return mapping


class ReconcileStats(TypedDict):
    portal_rows: int
    vendor_rows: int
    dup_portal_ids: int
    dup_vendor_ids: int
    matched_pairs: int
    both_ok: int
    date_mismatch: int
    amount_mismatch: int
    both_mismatch: int
    portal_only: int
    vendor_only: int


def merge_reconcile(
    portal_norm: pd.DataFrame,
    vendor_norm: pd.DataFrame,
    *,
    date_tolerance_seconds: float,
    amount_epsilon: float,
    same_minute: bool,
    vendor_shift_hours: float,
) -> tuple[pd.DataFrame, ReconcileStats, dict[str, pd.DataFrame]]:
    p = portal_norm.drop(columns=["source"], errors="ignore").copy()
    v = vendor_norm.drop(columns=["source"], errors="ignore").copy()

    dup_p = int(p["transaction_id"].duplicated().sum())
    dup_v = int(v["transaction_id"].duplicated().sum())

    merged = p.merge(v, on="transaction_id", how="outer", suffixes=("_portal", "_vendor"))

    has_p = merged["occurred_at_portal"].notna()
    has_v = merged["occurred_at_vendor"].notna()
    matched = has_p & has_v

    # Vendor clock 1h behind portal → add hours to vendor time before comparing to portal.
    v_adj = merged["occurred_at_vendor"] + pd.to_timedelta(vendor_shift_hours, unit="h")
    merged["occurred_at_vendor_adj"] = v_adj

    if same_minute:
        dt_delta = (
            merged["occurred_at_portal"].dt.floor("min")
            - v_adj.dt.floor("min")
        ).abs()
    else:
        dt_delta = (merged["occurred_at_portal"] - v_adj).abs().dt.total_seconds()

    date_ok = matched & (
        (dt_delta <= date_tolerance_seconds)
        if not same_minute
        else (dt_delta == pd.Timedelta(0))
    )

    ap = merged["amount_portal"]
    av = merged["amount_vendor"]
    amount_ok = matched & np.isfinite(ap) & np.isfinite(av) & (np.abs(ap - av) <= amount_epsilon)

    portal_only = has_p & ~has_v
    vendor_only = ~has_p & has_v

    both_ok = matched & date_ok & amount_ok
    date_only_bad = matched & ~date_ok & amount_ok
    amount_only_bad = matched & date_ok & ~amount_ok
    both_bad = matched & ~date_ok & ~amount_ok

    merged["_date_delta_seconds"] = np.where(
        matched & has_p & has_v,
        (merged["occurred_at_portal"] - v_adj).abs().dt.total_seconds(),
        np.nan,
    )
    merged["_amount_delta"] = np.where(
        matched & np.isfinite(ap) & np.isfinite(av),
        np.abs(ap - av),
        np.nan,
    )

    stats: ReconcileStats = {
        "portal_rows": len(p),
        "vendor_rows": len(v),
        "dup_portal_ids": dup_p,
        "dup_vendor_ids": dup_v,
        "matched_pairs": int(matched.sum()),
        "both_ok": int(both_ok.sum()),
        "date_mismatch": int((matched & ~date_ok).sum()),
        "amount_mismatch": int((matched & ~amount_ok).sum()),
        "both_mismatch": int(both_bad.sum()),
        "portal_only": int(portal_only.sum()),
        "vendor_only": int(vendor_only.sum()),
    }

    views = {
        "merged": merged,
        "both_ok": merged.loc[both_ok],
        "date_mismatch": merged.loc[date_only_bad],
        "amount_mismatch": merged.loc[amount_only_bad],
        "both_mismatch": merged.loc[both_bad],
        "portal_only": merged.loc[portal_only],
        "vendor_only": merged.loc[vendor_only],
    }
    return merged, stats, views


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def amount_column_config(df: pd.DataFrame) -> dict[str, Any]:
    """Thousand separators in the UI for amount-like numeric columns (downloads stay raw)."""
    cfg: dict[str, Any] = {}
    if df is None or df.empty:
        return cfg
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        name = str(col).lower()
        if (
            "amount" in name
            or name == "_amount_delta"
            or name.endswith("amount_delta")
        ):
            cfg[str(col)] = st.column_config.NumberColumn(
                str(col),
                format="localized",
            )
    return cfg


def show_dataframe_table(df: pd.DataFrame) -> None:
    st.dataframe(
        df,
        use_container_width=True,
        column_config=amount_column_config(df),
    )


def _fmt_amount(x: float) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "—"
    return f"{x:,.0f}"


def _sum_amount(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nansum(s)) if len(s) else 0.0


def render_transaction_metrics_tab(
    portal_norm: pd.DataFrame,
    vendor_norm: pd.DataFrame,
    merged: pd.DataFrame,
    stats: ReconcileStats,
    views: dict[str, pd.DataFrame],
) -> None:
    """High-level counts and amounts for portal, vendor, and reconcile buckets."""
    st.subheader("Transaction metrics")
    st.caption(
        "Based on normalized rows (after column mapping, portal ×1000 if enabled, and row validation)."
    )

    pu = int(portal_norm["transaction_id"].nunique()) if not portal_norm.empty else 0
    vu = int(vendor_norm["transaction_id"].nunique()) if not vendor_norm.empty else 0
    pr = len(portal_norm)
    vr = len(vendor_norm)

    p_total = _sum_amount(portal_norm["amount"]) if not portal_norm.empty else 0.0
    v_total = _sum_amount(vendor_norm["amount"]) if not vendor_norm.empty else 0.0

    st.markdown("### Portal vs vendor (all normalized rows)")
    c = st.columns(3)
    with c[0]:
        st.metric("Portal — transactions (rows)", f"{pr:,}")
        st.metric("Portal — unique transaction IDs", f"{pu:,}")
        st.metric("Portal — total amount", _fmt_amount(p_total))
    with c[1]:
        st.metric("Vendor — transactions (rows)", f"{vr:,}")
        st.metric("Vendor — unique transaction IDs", f"{vu:,}")
        st.metric("Vendor — total amount", _fmt_amount(v_total))
    with c[2]:
        st.metric("Row count delta (portal − vendor)", f"{pr - vr:,}")
        st.metric("Unique ID delta (portal − vendor)", f"{pu - vu:,}")
        st.metric("Total amount delta (portal − vendor)", _fmt_amount(p_total - v_total))

    if not portal_norm.empty:
        pa = portal_norm["amount"]
        st.markdown("### Portal — amount distribution")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Minimum", _fmt_amount(float(pa.min())))
        d2.metric("Maximum", _fmt_amount(float(pa.max())))
        d3.metric("Mean", _fmt_amount(float(pa.mean())))
        d4.metric("Median", _fmt_amount(float(pa.median())))

    if not vendor_norm.empty:
        va = vendor_norm["amount"]
        st.markdown("### Vendor — amount distribution")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Minimum", _fmt_amount(float(va.min())))
        d2.metric("Maximum", _fmt_amount(float(va.max())))
        d3.metric("Mean", _fmt_amount(float(va.mean())))
        d4.metric("Median", _fmt_amount(float(va.median())))

    st.markdown("### Reconciliation buckets (matched on transaction ID)")
    bucket_rows: list[dict[str, Any]] = []
    for key, title in [
        ("both_ok", "Matched — date & amount OK"),
        ("date_mismatch", "Matched — date mismatch only"),
        ("amount_mismatch", "Matched — amount mismatch only"),
        ("both_mismatch", "Matched — date & amount mismatch"),
        ("portal_only", "Portal only (no vendor ID)"),
        ("vendor_only", "Vendor only (no portal ID)"),
    ]:
        df = views.get(key, pd.DataFrame())
        n = len(df)
        sp = _sum_amount(df["amount_portal"]) if n and "amount_portal" in df.columns else 0.0
        sv = _sum_amount(df["amount_vendor"]) if n and "amount_vendor" in df.columns else 0.0
        bucket_rows.append(
            {
                "Bucket": title,
                "Rows": n,
                "Sum amount (portal)": sp,
                "Sum amount (vendor)": sv,
                "Amount delta (P−V)": sp - sv,
            }
        )
    bucket_df = pd.DataFrame(bucket_rows)
    show_dataframe_table(bucket_df)

    st.markdown("### Match quality (ID-level)")
    mcols = st.columns(4)
    mcols[0].metric("Matched ID pairs", f"{stats['matched_pairs']:,}")
    mcols[1].metric("Both OK", f"{stats['both_ok']:,}")
    mcols[2].metric("Any date issue (matched)", f"{stats['date_mismatch']:,}")
    mcols[3].metric("Any amount issue (matched)", f"{stats['amount_mismatch']:,}")
    st.caption(
        "Duplicate IDs on one side can inflate row counts; unique ID counts above reflect normalized tables."
    )

    if not portal_norm.empty and portal_norm["txn_type"].notna().any():
        st.markdown("### Portal — by transaction type")
        g = (
            portal_norm.groupby("txn_type", dropna=False)
            .agg(rows=("transaction_id", "count"), total_amount=("amount", "sum"))
            .reset_index()
        )
        show_dataframe_table(g)

    if not vendor_norm.empty and vendor_norm["txn_type"].notna().any():
        st.markdown("### Vendor — by transaction type")
        g = (
            vendor_norm.groupby("txn_type", dropna=False)
            .agg(rows=("transaction_id", "count"), total_amount=("amount", "sum"))
            .reset_index()
        )
        show_dataframe_table(g)

    if not portal_norm.empty and portal_norm["currency"].notna().any():
        st.markdown("### Portal — by currency")
        g = (
            portal_norm.groupby("currency", dropna=False)
            .agg(rows=("transaction_id", "count"), total_amount=("amount", "sum"))
            .reset_index()
        )
        show_dataframe_table(g)

    if not vendor_norm.empty and vendor_norm["currency"].notna().any():
        st.markdown("### Vendor — by currency")
        g = (
            vendor_norm.groupby("currency", dropna=False)
            .agg(rows=("transaction_id", "count"), total_amount=("amount", "sum"))
            .reset_index()
        )
        show_dataframe_table(g)


def _filter_by_calendar_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    ts_col: str = "occurred_at",
) -> pd.DataFrame:
    if df.empty or ts_col not in df.columns:
        return df.iloc[0:0].copy()
    od = df[ts_col].dt.date
    return df.loc[(od >= start_d) & (od <= end_d)].copy()


def period_key_and_sort(
    ts: pd.Series, granularity: str
) -> tuple[pd.Series, pd.Series]:
    """Return (display key, sort key) for each timestamp."""
    if granularity == "Daily":
        key = ts.dt.strftime("%Y-%m-%d")
        sort_key = pd.to_datetime(ts.dt.date)
        return key, sort_key
    if granularity == "Weekly":
        iso = ts.dt.isocalendar()
        wk = iso.week.astype(int)
        yr = iso.year.astype(int)
        key = yr.astype(str) + "-W" + wk.astype(str).str.zfill(2)
        sort_key = pd.Series(yr * 100 + wk, index=ts.index, dtype="int64")
        return key, sort_key
    if granularity == "Monthly":
        key = ts.dt.strftime("%Y-%m")
        sort_key = ts.dt.year * 100 + ts.dt.month
        return key, sort_key.astype("int64")
    raise ValueError(granularity)


def _aggregate_by_period(
    df: pd.DataFrame, granularity: str, ts_col: str = "occurred_at"
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["period_key", "_sort", "transaction_count", "total_amount"]
        )
    ts = df[ts_col]
    pk, sk = period_key_and_sort(ts, granularity)
    tmp = df.assign(_period_key=pk, _sort=sk)
    g = (
        tmp.groupby("_period_key", as_index=False)
        .agg(
            _sort=("_sort", "min"),
            transaction_count=("transaction_id", "count"),
            total_amount=("amount", "sum"),
        )
        .rename(columns={"_period_key": "period_key"})
    )
    return g.sort_values("_sort", kind="mergesort").reset_index(drop=True)


def render_period_breakdown_tab(
    portal_norm: pd.DataFrame,
    vendor_norm: pd.DataFrame,
    *,
    vendor_shift_hours: float,
) -> None:
    st.subheader("Period breakdown")
    st.caption(
        "Portal uses each row’s `occurred_at`. Vendor uses the same **+hour shift** as reconciliation "
        f"({vendor_shift_hours:g} h) when assigning day/week/month so buckets match “matched OK” logic. "
        "You can bucket on raw vendor time instead using the checkbox below."
    )

    v_ts_for_range = (
        vendor_norm["occurred_at"] + pd.to_timedelta(vendor_shift_hours, unit="h")
        if not vendor_norm.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    all_ts = pd.concat(
        [
            portal_norm["occurred_at"] if not portal_norm.empty else pd.Series(dtype="datetime64[ns]"),
            v_ts_for_range,
        ],
        ignore_index=True,
    )
    if all_ts.empty:
        st.info("No transaction timestamps to summarize.")
        return

    dmin = all_ts.min().date()
    dmax = all_ts.max().date()

    gran = st.radio(
        "Breakdown",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="pb_granularity",
    )

    c1, c2 = st.columns(2)
    with c1:
        start_d = st.date_input("From date", value=dmin, key="pb_start")
    with c2:
        end_d = st.date_input("To date", value=dmax, key="pb_end")

    if start_d > end_d:
        st.error("From date must be on or before To date.")
        return

    p_f = _filter_by_calendar_range(portal_norm, start_d, end_d, "occurred_at")

    raw_vendor_bucket = st.checkbox(
        "Bucket vendor by raw `occurred_at` (ignore reconciliation hour shift)",
        value=False,
        key="pb_raw_vendor_bucket",
        help="If checked, daily/weekly/monthly counts use vendor timestamps as exported. "
        "That can disagree with “matched OK” when portal and vendor time zones differ.",
    )

    v_work = vendor_norm.copy()
    if not v_work.empty:
        if raw_vendor_bucket:
            v_work["_bucket_ts"] = v_work["occurred_at"]
        else:
            v_work["_bucket_ts"] = v_work["occurred_at"] + pd.to_timedelta(
                vendor_shift_hours, unit="h"
            )

    v_f = _filter_by_calendar_range(v_work, start_d, end_d, "_bucket_ts")

    if gran == "Weekly":
        pk_p = period_key_and_sort(p_f["occurred_at"], "Weekly")[0] if not p_f.empty else pd.Series(dtype=object)
        pk_v = period_key_and_sort(v_f["_bucket_ts"], "Weekly")[0] if not v_f.empty else pd.Series(dtype=object)
        week_opts = sorted(set(pk_p.dropna().astype(str).tolist()) | set(pk_v.dropna().astype(str).tolist()))
        if week_opts:
            pick = st.multiselect(
                "ISO weeks to include (leave empty to include every week in the date range)",
                options=week_opts,
                default=[],
                key="pb_weeks",
                help="Week labels use ISO-8601 (week starts Monday). Select specific weeks to narrow the table.",
            )
            if pick:
                if not p_f.empty:
                    p_f = p_f.loc[pk_p.astype(str).isin(pick)].copy()
                if not v_f.empty:
                    v_f = v_f.loc[pk_v.astype(str).isin(pick)].copy()
        else:
            st.caption("No ISO weeks in the selected date range.")

    if gran == "Monthly":
        pk_p = period_key_and_sort(p_f["occurred_at"], "Monthly")[0] if not p_f.empty else pd.Series(dtype=object)
        pk_v = period_key_and_sort(v_f["_bucket_ts"], "Monthly")[0] if not v_f.empty else pd.Series(dtype=object)
        month_opts = sorted(
            set(pk_p.dropna().astype(str).tolist()) | set(pk_v.dropna().astype(str).tolist())
        )
        if month_opts:
            pick_m = st.multiselect(
                "Months to include (leave empty to include every month in the date range)",
                options=month_opts,
                default=[],
                key="pb_months",
            )
            if pick_m:
                if not p_f.empty:
                    p_f = p_f.loc[pk_p.astype(str).isin(pick_m)].copy()
                if not v_f.empty:
                    v_f = v_f.loc[pk_v.astype(str).isin(pick_m)].copy()
        else:
            st.caption("No calendar months in the selected date range.")

    pa = _aggregate_by_period(p_f, gran, "occurred_at").rename(
        columns={
            "transaction_count": "portal_transactions",
            "total_amount": "portal_total_amount",
            "_sort": "_sort_p",
        }
    )
    va = _aggregate_by_period(v_f, gran, "_bucket_ts").rename(
        columns={
            "transaction_count": "vendor_transactions",
            "total_amount": "vendor_total_amount",
            "_sort": "_sort_v",
        }
    )

    out = pa.merge(va, on="period_key", how="outer")
    out["_sort"] = out["_sort_p"].combine_first(out["_sort_v"])
    out = out.drop(columns=["_sort_p", "_sort_v"], errors="ignore")
    out = out.sort_values("_sort", kind="mergesort")

    out["portal_transactions"] = out["portal_transactions"].fillna(0).astype(int)
    out["vendor_transactions"] = out["vendor_transactions"].fillna(0).astype(int)
    out["portal_total_amount"] = out["portal_total_amount"].fillna(0.0)
    out["vendor_total_amount"] = out["vendor_total_amount"].fillna(0.0)
    out["transaction_count_delta_P_minus_V"] = (
        out["portal_transactions"] - out["vendor_transactions"]
    )
    out["total_amount_delta_P_minus_V"] = (
        out["portal_total_amount"] - out["vendor_total_amount"]
    )

    out = out.drop(columns=["_sort"], errors="ignore")

    st.markdown(f"### {gran} totals (portal vs vendor)")
    if out.empty:
        st.warning("No rows in the selected period / filters.")
        return

    show_dataframe_table(out)
    st.download_button(
        "Download period breakdown CSV",
        data=df_to_csv_bytes(out),
        file_name=f"period_breakdown_{gran.lower()}.csv",
        mime="text/csv",
        key="dl_period_breakdown",
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Periods shown", f"{len(out):,}")
    m2.metric(
        "Portal total amount (sum of rows in table)",
        _fmt_amount(float(out["portal_total_amount"].sum())),
    )
    m3.metric(
        "Vendor total amount (sum of rows in table)",
        _fmt_amount(float(out["vendor_total_amount"].sum())),
    )


def main() -> None:
    st.set_page_config(page_title="Transaction reconciliation", layout="wide")
    st.title("Transaction reconciliation")
    st.caption("Normalize portal and vendor exports, then compare dates and amounts by transaction ID.")

    # --- Session defaults for vendor guess ---
    if "vendor_guess_cols" not in st.session_state:
        st.session_state["vendor_guess_cols"] = None
    if "vendor_mapping_guess" not in st.session_state:
        st.session_state["vendor_mapping_guess"] = None

    with st.sidebar:
        st.header("Data sources")
        portal_path = st.text_input(
            "Portal CSV path (optional if uploading)",
            value="",
            help="Leave empty and use upload below, or paste the full path to a .csv on disk.",
        )
        portal_upload = st.file_uploader("Upload portal CSV", type=["csv"])

        vendor_path = st.text_input(
            "Vendor Excel path (full path to the file on disk — not for uploads)",
            value="",
            help="Use this only when the .xlsx is saved on your PC. "
            "If you use “upload” below, leave this blank. "
            "Include the full path, e.g. D:\\Downloads\\file.xlsx",
        )
        vendor_upload = st.file_uploader(
            "Upload vendor .xlsx",
            type=["xlsx"],
            help="Preferred: pick the file here. The path field above is only for files already on disk.",
        )

        st.header("Comparison")
        same_minute = st.checkbox("Date match: same calendar minute (ignore seconds)", value=False)
        vendor_shift_hours = st.number_input(
            "Hours to add to vendor datetime before comparing to portal",
            min_value=-24.0,
            max_value=24.0,
            value=1.0,
            step=0.5,
            help="If vendor export is one time zone behind portal, use +1 so the same instant lines up.",
        )
        date_tol = st.number_input(
            "Date tolerance (seconds, ignored if same-minute mode)",
            min_value=0.0,
            value=60.0,
            step=1.0,
        )
        amount_eps = st.number_input("Amount epsilon", min_value=0.0, value=0.01, step=0.01, format="%.4f")
        int_round = st.checkbox("Round amounts to whole numbers before compare", value=True)
        portal_amount_k = st.checkbox(
            "Portal amount is in thousands (multiply portal amount by 1000)",
            value=True,
            help="Enable when portal value like 300 should be treated as 300,000.",
        )
        portal_dayfirst = st.checkbox("Portal/vendor date: day-first parsing (DD/MM/…)", value=False)

    # --- Load portal ---
    portal_raw: pd.DataFrame | None = None
    try:
        if portal_upload is not None:
            portal_raw = load_portal_csv(portal_upload)
        elif portal_path.strip() and Path(portal_path.strip()).is_file():
            portal_raw = load_portal_csv(Path(portal_path.strip()))
    except Exception as e:
        st.error(f"Failed to load portal CSV: {e}")
        portal_raw = None

    # --- Load vendor ---
    vendor_raw: pd.DataFrame | None = None
    vendor_sheet: str | int = 0
    xls_names: list[str] = []

    vendor_src = None
    vendor_load_label = ""
    path_attempted: Path | None = None
    if vendor_upload is not None:
        vendor_src = vendor_upload
        vendor_load_label = getattr(vendor_upload, "name", "uploaded file")
    elif vendor_path.strip():
        path_attempted = parse_local_path(vendor_path)
        if path_attempted.is_file():
            vendor_src = path_attempted
            vendor_load_label = str(path_attempted)
        else:
            vendor_src = None

    if vendor_src is not None:
        try:
            if vendor_upload is not None:
                vbytes = vendor_upload.getvalue()
                vbio = io.BytesIO(vbytes)
                xls = pd.ExcelFile(vbio)
                xls_names = list(xls.sheet_names)
                if len(xls_names) > 1:
                    vendor_sheet = st.sidebar.selectbox(
                        "Vendor sheet", xls_names, index=0, key="vendor_sheet_pick"
                    )
                else:
                    vendor_sheet = xls_names[0] if xls_names else 0
                vbio.seek(0)
                vendor_raw = pd.read_excel(vbio, sheet_name=vendor_sheet, dtype=None)
            else:
                xls = pd.ExcelFile(vendor_src)
                xls_names = list(xls.sheet_names)
                if len(xls_names) > 1:
                    vendor_sheet = st.sidebar.selectbox(
                        "Vendor sheet", xls_names, index=0, key="vendor_sheet_pick"
                    )
                else:
                    vendor_sheet = xls_names[0] if xls_names else 0
                vendor_raw = pd.read_excel(
                    vendor_src, sheet_name=vendor_sheet, dtype=None
                )
        except Exception as e:
            st.error(f"Failed to load vendor Excel: {e}")
            vendor_raw = None

    if vendor_path.strip() and vendor_upload is None and path_attempted is not None:
        if vendor_raw is None and not path_attempted.is_file():
            st.warning(
                f"**Vendor file not found** at `{path_attempted}`. "
                "Copy the full path from File Explorer (address bar), paste it here, or use **Upload vendor .xlsx** instead."
            )

    if vendor_src is not None and vendor_raw is not None:
        st.success(
            f"Vendor loaded: **{vendor_load_label}** — {len(vendor_raw)} raw rows, "
            f"{len(vendor_raw.columns)} columns."
        )

    if portal_raw is None:
        st.info("Load a portal CSV via path or upload to begin.")
        return

    portal_cols = [str(c) for c in portal_raw.columns]

    st.sidebar.header("Column mapping — portal")
    portal_map: dict[str, str | None] = {}
    for slot in CANONICAL_SLOTS:
        default = PORTAL_DEFAULT_COLS.get(slot, "")
        idx = 0
        choices = slot_choices(portal_cols)
        if default in portal_cols:
            idx = choices.index(default)
        label = f"Portal → {slot}"
        portal_map[slot] = col_for_slot(
            st.sidebar.selectbox(label, choices, index=idx, key=f"pm_{slot}")
        )

    st.sidebar.header("Column mapping — vendor")
    vendor_map: dict[str, str | None] = {s: None for s in CANONICAL_SLOTS}
    if vendor_raw is not None:
        vcols = [str(c) for c in vendor_raw.columns]
        if st.session_state["vendor_guess_cols"] != vcols:
            st.session_state["vendor_guess_cols"] = vcols
            st.session_state["vendor_mapping_guess"] = guess_vendor_mapping(vcols)
        guess = st.session_state["vendor_mapping_guess"] or {}

        for slot in CANONICAL_SLOTS:
            choices = slot_choices(vcols)
            g = guess.get(slot)
            idx = 0
            if g and g in vcols:
                idx = choices.index(g)
            vendor_map[slot] = col_for_slot(
                st.sidebar.selectbox(
                    f"Vendor → {slot}",
                    choices,
                    index=idx,
                    key=f"vm_{slot}",
                )
            )
    else:
        st.sidebar.caption("Load vendor Excel to map its columns.")

    portal_norm = raw_to_canonical(
        portal_raw,
        portal_map,
        source_label="portal",
        portal_date_dayfirst=portal_dayfirst,
        integer_amount_round=int_round,
        amount_scale_multiplier=1000.0 if portal_amount_k else 1.0,
    )

    vendor_norm = (
        raw_to_canonical(
            vendor_raw,
            vendor_map,
            source_label="vendor",
            portal_date_dayfirst=portal_dayfirst,
            integer_amount_round=int_round,
            amount_scale_multiplier=1.0,
        )
        if vendor_raw is not None
        else pd.DataFrame(columns=list(CANONICAL_SLOTS) + ["source"])
    )

    tabs = st.tabs(
        [
            "Normalized preview",
            "Matched OK",
            "Date mismatch",
            "Amount mismatch",
            "Both mismatched",
            "Portal only",
            "Vendor only",
            "Transaction metrics",
            "Period breakdown",
            "Summary",
        ]
    )

    with tabs[0]:
        st.subheader("Normalized portal")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", len(portal_norm))
        with c2:
            if not portal_norm.empty:
                st.metric(
                    "Date range",
                    f"{portal_norm['occurred_at'].min()} → {portal_norm['occurred_at'].max()}",
                )
        show_dataframe_table(portal_norm.head(200))

        st.subheader("Normalized vendor")
        if vendor_raw is None:
            st.warning("Vendor file not loaded.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rows (after mapping)", len(vendor_norm))
            with c2:
                if not vendor_norm.empty:
                    st.metric(
                        "Date range",
                        f"{vendor_norm['occurred_at'].min()} → {vendor_norm['occurred_at'].max()}",
                    )
            show_dataframe_table(vendor_norm.head(200))
            if vendor_raw is not None and not vendor_raw.empty and vendor_norm.empty:
                st.error(
                    "**Normalized vendor has 0 rows.** The file loaded, but no rows passed validation. "
                    "In the sidebar under **Column mapping — vendor**, set **transaction_id**, "
                    "**occurred_at**, and **amount** to the correct columns (not “(none)”). "
                    "Rows are dropped if ID, date, or amount is missing after parsing."
                )
                with st.expander("Raw vendor preview (first rows)"):
                    show_dataframe_table(vendor_raw.head(15))
                    st.caption(f"Columns detected: {list(vendor_raw.columns)}")

    if vendor_raw is None or vendor_norm.empty:
        for i in range(1, len(tabs)):
            with tabs[i]:
                st.info("Load and map vendor data to run reconciliation.")
        return

    merged, stats, views = merge_reconcile(
        portal_norm,
        vendor_norm,
        date_tolerance_seconds=float(date_tol),
        amount_epsilon=float(amount_eps),
        same_minute=same_minute,
        vendor_shift_hours=float(vendor_shift_hours),
    )

    if stats["dup_portal_ids"] or stats["dup_vendor_ids"]:
        st.warning(
            f"Duplicate transaction_id values: portal={stats['dup_portal_ids']}, "
            f"vendor={stats['dup_vendor_ids']}. Merge rows may be duplicated."
        )

    def show_tab(key: str, title: str, tab_index: int) -> None:
        with tabs[tab_index]:
            st.subheader(title)
            dfv = views[key]
            st.metric("Rows", len(dfv))
            show_dataframe_table(dfv)
            if not dfv.empty:
                st.download_button(
                    f"Download {title} CSV",
                    data=df_to_csv_bytes(dfv),
                    file_name=f"reconcile_{key}.csv",
                    mime="text/csv",
                    key=f"dl_{key}",
                )

    show_tab("both_ok", "Matched: date and amount OK", 1)
    show_tab("date_mismatch", "Matched IDs with date mismatch", 2)
    show_tab("amount_mismatch", "Matched IDs with amount mismatch", 3)
    show_tab("both_mismatch", "Matched IDs with both date and amount mismatch", 4)
    show_tab("portal_only", "Portal only (no vendor row for ID)", 5)
    show_tab("vendor_only", "Vendor only (no portal row for ID)", 6)

    with tabs[7]:
        render_transaction_metrics_tab(
            portal_norm, vendor_norm, merged, stats, views
        )

    with tabs[8]:
        render_period_breakdown_tab(
            portal_norm,
            vendor_norm,
            vendor_shift_hours=float(vendor_shift_hours),
        )

    with tabs[9]:
        st.subheader("Summary")
        mcols = st.columns(4)
        mcols[0].metric("Portal rows", stats["portal_rows"])
        mcols[1].metric("Vendor rows", stats["vendor_rows"])
        mcols[2].metric("Matched IDs", stats["matched_pairs"])
        mcols[3].metric("Both OK", stats["both_ok"])
        mcols = st.columns(4)
        mcols[0].metric("Portal only", stats["portal_only"])
        mcols[1].metric("Vendor only", stats["vendor_only"])
        mcols[2].metric("Date mismatch (matched)", stats["date_mismatch"])
        mcols[3].metric("Amount mismatch (matched)", stats["amount_mismatch"])
        st.metric("Both mismatched (matched)", stats["both_mismatch"])
        show_dataframe_table(merged.head(500))
        st.caption("Showing up to 500 rows above; CSV download contains the full join.")
        st.download_button(
            "Download full outer-join CSV",
            data=df_to_csv_bytes(merged),
            file_name="reconcile_full_outer.csv",
            mime="text/csv",
            key="dl_full",
        )


if __name__ == "__main__":
    main()
