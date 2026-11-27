"""Operations command centre: Requirements 1-5 as a Streamlit tab.
One REGION = one CITY. Each city has its own stock level and its own spray-crew pool."""
import pandas as pd
import plotly.express as px
import streamlit as st
from src.capacity import load_capacity
from src.config import CREW_EFFECT, MAX_CREWS_PER_WARD
from src.dispatch import brief_html, export_all, sms, whatsapp
from src.overflow import route_overflow
from src.resources import allocate_regions
from src.simulator import case_curve, optimal_deployment, roi_ranking, simulate


def render_ops(fc: pd.DataFrame, model, meta: dict, week_label: str):
    """fc: one row per ward for the chosen date (features + pred_cases + tier + city)."""
    F, cut, cap = meta["features"], meta["tier_cutoffs"], load_capacity()
    st.success("Running 100% locally: forecasting, allocation, routing and dispatch need "
               "no internet, no external API and no LLM.")
    cities = sorted(fc.city.unique())

    # ---- regional scarcity controls: every city has its own stock and crew pool ----
    st.subheader("Regional scarcity controls (one region = one city)")
    settings = {}
    for col, c in zip(st.columns(len(cities)), cities):
        with col:
            st.markdown(f"**{c}**")
            s = st.slider("Available stock (%)", 30, 100, 70, key=f"ops_stock_{c}")
            k = st.slider("Spraying crews", 0, 60, 8, key=f"ops_crews_{c}")
        settings[c] = (s, k)

    alloc_all = allocate_regions(fc, cap, settings)
    wards_all, transfers_all, city_all = route_overflow(alloc_all)

    glance = alloc_all.groupby("city").agg(
        saline=("saline_allocated", "sum"), saline_req=("saline_bags_needed", "sum"),
        kits=("kits_allocated", "sum"), kits_req=("diagnostic_kits_needed", "sum"),
        crews=("crews_assigned", "sum"), crews_req=("crews_needed", "sum"),
        deficit_wards=("alert", lambda s: int((s == "CRITICAL SUPPLY DEFICIT").sum())),
    ).reset_index()
    glance.insert(1, "stock_%", glance.city.map(lambda c: settings[c][0]))
    glance.insert(2, "crew_pool", glance.city.map(lambda c: settings[c][1]))
    glance = glance.merge(city_all[["city", "utilization_pct", "unplaced"]], on="city")
    st.caption("All regions at a glance")
    st.dataframe(glance.rename(columns={"utilization_pct": "bed_util_%",
                                        "unplaced": "beds_unplaced"}), hide_index=True)

    region = st.selectbox("View region", cities, key="ops_region")
    stock, crews_total = settings[region]
    alloc = alloc_all[alloc_all.city == region]
    fc_r = fc[fc.city == region].reset_index(drop=True)

    t1, t2, t3, t4 = st.tabs(["1 Allocation", "2 Overflow routing",
                              "3 What-if simulator", "4 Dispatch"])

    with t1:
        n_def = int((alloc.alert == "CRITICAL SUPPLY DEFICIT").sum())
        if n_def:
            st.error(f"CRITICAL SUPPLY DEFICIT in {n_def} of {len(alloc)} {region} wards: demand "
                     f"cannot be fully met at {stock}% stock and {crews_total} crews.")
        else:
            st.success(f"{region}: all supply and crew demand is fully met.")
        m = st.columns(4)
        for col, (lab, a, n) in zip(m, [("Saline", "saline_allocated", "saline_bags_needed"),
                                        ("Kits", "kits_allocated", "diagnostic_kits_needed"),
                                        ("Beds", "beds_allocated", "hospital_beds_required"),
                                        ("Crews", "crews_assigned", "crews_needed")]):
            col.metric(f"{lab} allocated / required", f"{int(alloc[a].sum())} / {int(alloc[n].sum())}")
        st.dataframe(alloc[["ward_id", "tier", "pred_cases", "population_density_sqkm",
                            "saline_allocated", "saline_bags_needed", "kits_allocated",
                            "diagnostic_kits_needed", "beds_allocated", "hospital_beds_required",
                            "crews_assigned", "crews_needed", "alert", "deficit_items"]]
                     .round(1), hide_index=True)
        st.caption(f"{region} wards are served from {region}'s own stock, in priority order: "
                   "risk tier x predicted cases x crowding.")

    with t2:
        st.subheader(f"{region}: bed utilization")
        st.dataframe(city_all[city_all.city == region], hide_index=True)
        u = int(city_all.loc[city_all.city == region, "unplaced"].iloc[0])
        if u > 0:
            st.error(f"{u} patients cannot be placed: every {region} ward is at capacity. "
                     "Routing never crosses city borders. Escalate for emergency beds.")
        st.subheader("Wards over local capacity (net deficit)")
        w = wards_all[wards_all.city == region]
        st.dataframe(w[w.net_deficit > 0][["ward_id", "hospital_beds_required", "bed_capacity_eff",
                     "net_deficit", "sent", "unplaced"]], hide_index=True)
        st.subheader("Patient transfers (within this city)")
        st.dataframe(transfers_all[transfers_all.city == region], hide_index=True)
        st.subheader("Receiving wards")
        st.dataframe(w[w.received > 0][["ward_id", "received", "headroom_left"]], hide_index=True)
        with st.expander("All regions: bed utilization"):
            st.dataframe(city_all, hide_index=True)

    with t3:
        st.write(f"Deploy {region}'s spray crews. Crews shrink the breeding inputs and the trained "
                 "model is re-run to get the new forecast. Reductions never exceed the baseline.")
        effect = st.slider("Crew effectiveness (breeding cut per crew)", 0.05, 0.40,
                           CREW_EFFECT, 0.01,
                           help="Scenario parameter, not estimable from observational data.")
        curve = case_curve(model, F, fc_r, effect=effect)
        mode = st.radio("Deployment", ["Auto-optimise", "Manual"], horizontal=True)
        ids = fc_r.ward_id.tolist()
        if mode == "Auto-optimise":
            n = st.slider("Crews to deploy", 0, max(crews_total, 1), crews_total)
            plan = optimal_deployment(curve, ids, n)
        else:
            ed = st.data_editor(pd.DataFrame({"ward_id": ids, "crews": 0}), hide_index=True,
                                disabled=["ward_id"], column_config={"crews": st.column_config
                                .NumberColumn(min_value=0, max_value=MAX_CREWS_PER_WARD, step=1)})
            plan = dict(zip(ed.ward_id, ed.crews))
            if sum(plan.values()) > crews_total:
                st.warning(f"You assigned {sum(plan.values())} crews but {region} has "
                           f"only {crews_total}.")
        res = simulate(curve, fc_r, plan, cut)
        k = st.columns(4)
        k[0].metric("Crews deployed", int(res.crews.sum()))
        k[1].metric("Cases prevented (2-4 wk)", f"{res.cases_prevented.sum():.0f}")
        k[2].metric("Beds saved", int(res.beds_saved.sum()))
        k[3].metric("Saline saved", int(res.saline_saved.sum()))
        wk = pd.DataFrame({"week": ["Week +2", "Week +3", "Week +4"],
                           "cases prevented": [res.prevented_w2.sum(), res.prevented_w3.sum(),
                                               res.prevented_w4.sum()]})
        st.plotly_chart(px.bar(wk, x="week", y="cases prevented", height=250))
        st.subheader("Deployed wards ranked by ROI (cases prevented per crew)")
        st.dataframe(res[res.crews > 0][["ward_id", "crews", "base_cases", "sim_cases",
                     "cases_prevented", "roi", "beds_saved", "saline_saved", "tier_before",
                     "tier_after"]].round(1), hide_index=True)
        st.subheader(f"All {region} wards: value of the first crew")
        st.dataframe(roi_ranking(curve, fc_r).round(1), hide_index=True)

    with t4:
        ward = st.selectbox("Ward", alloc.ward_id)
        r = next(alloc[alloc.ward_id == ward].itertuples())
        st.markdown("**SMS**")
        st.code(sms(r, week_label, transfers_all), language=None)
        st.markdown("**WhatsApp**")
        st.code(whatsapp(r, week_label, transfers_all), language=None)
        st.download_button(f"Download {region} printable brief (HTML)",
                           brief_html(alloc_all, region, week_label, transfers_all),
                           f"brief_{region.replace(' ', '_')}.html", "text/html")
        if st.button("Write notices for ALL regions to /exports"):
            export_all(alloc_all, week_label, transfers_all)
            st.success("Written to exports/")
