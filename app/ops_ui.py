"""Operations command centre: Requirements 1-5 as a Streamlit tab."""
import pandas as pd
import plotly.express as px
import streamlit as st
from src.capacity import load_capacity
from src.config import CREW_EFFECT, MAX_CREWS_PER_WARD
from src.dispatch import brief_html, export_all, sms, whatsapp
from src.overflow import route_overflow
from src.resources import allocate
from src.simulator import case_curve, optimal_deployment, roi_ranking, simulate


def render_ops(fc: pd.DataFrame, model, meta: dict, week_label: str):
    """fc: one row per ward for the chosen date (features + pred_cases + tier + city)."""
    F, cut, cap = meta["features"], meta["tier_cutoffs"], load_capacity()
    st.success("Running 100% locally: forecasting, allocation, routing and dispatch need "
               "no internet, no external API and no LLM.")

    c1, c2 = st.columns(2)
    stock = c1.slider("Available regional stock (%)", 30, 100, 70, key="ops_stock")
    crews_total = c2.slider("Total spraying crews available", 0, 150, 40, key="ops_crews")
    alloc = allocate(fc, cap, stock, crews_total)
    wards, transfers, city = route_overflow(alloc)

    t1, t2, t3, t4 = st.tabs(["1 Allocation", "2 Overflow routing",
                              "3 What-if simulator", "4 Dispatch"])

    with t1:
        n_def = int((alloc.alert == "CRITICAL SUPPLY DEFICIT").sum())
        if n_def:
            st.error(f"CRITICAL SUPPLY DEFICIT in {n_def} wards: demand cannot be fully met "
                     f"at {stock}% stock and {crews_total} crews.")
        else:
            st.success("All supply and crew demand is fully met.")
        m = st.columns(4)
        for col, (lab, a, n) in zip(m, [("Saline", "saline_allocated", "saline_bags_needed"),
                                        ("Kits", "kits_allocated", "diagnostic_kits_needed"),
                                        ("Beds", "beds_allocated", "hospital_beds_required"),
                                        ("Crews", "crews_assigned", "crews_needed")]):
            col.metric(lab + " allocated / required", f"{alloc[a].sum()} / {alloc[n].sum()}")
        st.dataframe(alloc[["ward_id", "city", "tier", "pred_cases", "population_density_sqkm",
                            "saline_allocated", "saline_bags_needed", "kits_allocated",
                            "diagnostic_kits_needed", "beds_allocated", "hospital_beds_required",
                            "crews_assigned", "crews_needed", "alert", "deficit_items"]]
                     .round(1), hide_index=True)
        st.caption("Wards are served in priority order: risk tier x predicted cases x crowding.")

    with t2:
        st.subheader("City-wide bed utilization")
        st.dataframe(city, hide_index=True)
        if (city.unplaced > 0).any():
            st.error("Some overflow cannot be placed: the whole city pool is full. Routing "
                     "never crosses city borders. Escalate for emergency beds in: "
                     + ", ".join(city[city.unplaced > 0].city))
        st.subheader("Wards over local capacity (net deficit)")
        st.dataframe(wards[wards.net_deficit > 0][["city", "ward_id", "hospital_beds_required",
                     "bed_capacity_eff", "net_deficit", "sent", "unplaced"]], hide_index=True)
        st.subheader("Patient transfers (same city only)")
        st.dataframe(transfers, hide_index=True)
        st.subheader("Receiving wards")
        st.dataframe(wards[wards.received > 0][["city", "ward_id", "received", "headroom_left"]],
                     hide_index=True)

    with t3:
        st.write("Crews shrink the breeding inputs; the trained model is re-run to get the new "
                 "forecast. Reductions are never allowed to exceed the baseline.")
        effect = st.slider("Crew effectiveness (breeding cut per crew)", 0.05, 0.40,
                           CREW_EFFECT, 0.01,
                           help="Scenario parameter, not estimable from observational data.")
        curve = case_curve(model, F, fc, effect=effect)
        mode = st.radio("Deployment", ["Auto-optimise", "Manual"], horizontal=True)
        ids = fc.ward_id.tolist()
        if mode == "Auto-optimise":
            n = st.slider("Crews to deploy", 0, 150, min(15, crews_total))
            plan = optimal_deployment(curve, ids, n)
        else:
            ed = st.data_editor(pd.DataFrame({"ward_id": ids, "crews": 0}), hide_index=True,
                                disabled=["ward_id"], column_config={"crews": st.column_config
                                .NumberColumn(min_value=0, max_value=MAX_CREWS_PER_WARD, step=1)})
            plan = dict(zip(ed.ward_id, ed.crews))
        res = simulate(curve, fc, plan, cut)
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
        st.dataframe(res[res.crews > 0][["ward_id", "city", "crews", "base_cases", "sim_cases",
                     "cases_prevented", "roi", "beds_saved", "saline_saved", "tier_before",
                     "tier_after"]].round(1), hide_index=True)
        st.subheader("All wards: value of the first crew")
        st.dataframe(roi_ranking(curve, fc).round(1), hide_index=True)

    with t4:
        ward = st.selectbox("Ward", alloc.ward_id)
        r = next(alloc[alloc.ward_id == ward].itertuples())
        st.markdown("**SMS**")
        st.code(sms(r, week_label, transfers), language=None)
        st.markdown("**WhatsApp**")
        st.code(whatsapp(r, week_label, transfers), language=None)
        c = st.selectbox("City brief", sorted(alloc.city.unique()))
        st.download_button("Download printable brief (HTML)",
                           brief_html(alloc, c, week_label, transfers),
                           f"brief_{c.replace(' ', '_')}.html", "text/html")
        if st.button("Write all notices to /exports"):
            export_all(alloc, week_label, transfers)
            st.success("Written to exports/")