"""Core numbers: weekly cost per tonne-km and the two comparison baselines."""
import pandas as pd

from .config import Config

REQUIRED = ["shipment_id", "origin", "destination", "route_type", "quantity_tonnes",
            "distance_km", "freight_cost_inr", "shipment_date"]


def load_shipments(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"shipment file is missing columns: {missing}")
    bad = df[(df.quantity_tonnes <= 0) | (df.distance_km <= 0) | (df.freight_cost_inr <= 0)
             | df[REQUIRED].isna().any(axis=1)]
    if len(bad):  # fail loudly rather than silently skew a ratio
        raise ValueError(f"{len(bad)} rows have missing or non-positive values, e.g. {bad.iloc[0].shipment_id}")
    df["route"] = df.origin.str.strip() + "-" + df.destination.str.strip()
    # Monday of the shipment's Monday-Sunday week
    df["week_of"] = df.shipment_date - pd.to_timedelta(df.shipment_date.dt.weekday, unit="D")
    return df


def weekly_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Cost per tonne-km per (route, week) = sum(cost) / sum(tonnes * km)  (volume-weighted)."""
    d = df.assign(tonne_km=df.quantity_tonnes * df.distance_km)
    out = (d.groupby(["route", "route_type", "week_of"], as_index=False)
             .agg(shipments=("shipment_id", "count"),
                  total_cost_inr=("freight_cost_inr", "sum"),
                  tonne_km=("tonne_km", "sum")))
    out["cost_per_tonne_km"] = out.total_cost_inr / out.tonne_km
    return out


def add_baselines(weekly: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Adds own_avg / vs_own_pct (trailing window, strictly before the week) and
    peer_avg / vs_peer_pct (same week, same route_type, excluding the route itself)."""
    w = weekly.sort_values(["route", "week_of"]).reset_index(drop=True)

    def trailing(g: pd.DataFrame) -> pd.DataFrame:
        # Reindex to calendar weeks so a missing week is a gap, not a silent shift.
        s = g.set_index("week_of").cost_per_tonne_km
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="7D"))
        prior = s.shift(1).rolling(cfg.history_weeks, min_periods=1)
        res = pd.DataFrame({"own_avg": prior.mean(), "n_prior_weeks": prior.count()})
        res = res.reindex(g.week_of)
        res.index = g.index
        return res

    parts = [trailing(g) for _, g in w.groupby("route", sort=True)]
    w = w.join(pd.concat(parts))

    grp = w.groupby(["route_type", "week_of"]).cost_per_tonne_km
    total, n = grp.transform("sum"), grp.transform("count")
    w["peer_avg"] = ((total - w.cost_per_tonne_km) / (n - 1)).where(n > 1)

    w["vs_own_pct"] = (w.cost_per_tonne_km / w.own_avg - 1) * 100
    w["vs_peer_pct"] = (w.cost_per_tonne_km / w.peer_avg - 1) * 100
    return w


def is_candidate(w: pd.DataFrame, cfg: Config = Config()) -> pd.Series:
    """Rising AND out of the ordinary vs own history AND vs similar routes."""
    return ((w.n_prior_weeks >= cfg.min_prior_weeks)
            & (w.vs_own_pct >= cfg.min_vs_own_pct)
            & (w.vs_peer_pct >= cfg.min_vs_peer_pct))
