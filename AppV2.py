# ============================================================
# PART 1 — APP SETUP + SECTOR PRESETS + AI CALIBRATION
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from groq import Groq
import rasterio
from rasterio.transform import rowcol
from math import radians, sin, cos, asin, sqrt
import folium
from streamlit_folium import st_folium
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    palette="Set2",
    font_scale=1.05
)

plt.rcParams.update({
    "figure.figsize": (6.5, 4),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False
})

plt.style.use("ggplot")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Integrated Climate & Credit Risk Engine",
    layout="wide"
)

st.title("Integrated Climate & Credit Risk Engine")
st.caption(
    "ISSB S2 / IFRS S2 / TCFD / NGFS aligned — "
    "Integrated Transition, Physical & Credit Risk Assessment"
)

# ------------------------------------------------------------
# GLOBAL SESSION STATE — AUTHORITATIVE & CLEAN
# ------------------------------------------------------------
STATE_DEFAULTS = {
    "ai_outputs": {},
    # ========== INTEGRATED RESULTS ==========
    "integrated_ran": False,
    "df_integrated_summary": None,
    "df_integrated_assets": None,

    # ========== USER ENABLE FLAGS (INTENT) ==========
    "enable_transition": True,
    "enable_physical": False,
    "enable_targets": False,
    "enable_brsr": False,

    # ========== EXECUTION FLAGS ==========
    "transition_ran": False,
    "physical_ran": False,
    "targets_ran": False,
    "brsr_ran": False,
    "results_ready": False,

    # ========== EMISSIONS ==========
    "scope1": 0.0,
    "scope2": 0.0,
    "scope3": 0.0,

    # ========== DATA STORAGE ==========
    "df_transition": None,
    "df_target": None,
    "phys_summary": None,
    "phys_assets": None,
    "brsr_summary": None,
    "brsr_flags": None,
    "df_results": None
}

for k, v in STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------
# SECTOR PRESETS (UNCHANGED LOGIC)
# ------------------------------------------------------------
sector_ranges = {
    "carbon_pass_through": (0.1, 0.9),
    "demand_elasticity": (-1.5, 0.2),
    "price_elasticity": (-0.5, 0.5),
    "abatement_potential": (0.05, 0.80)
}

# ------------------------------------------------------------
# AI CALIBRATION (PREVIEW ONLY — NO AUTO-APPLICATION)
# ------------------------------------------------------------
def ai_calibrate_parameters(
    sector,
    revenue_0,
    ebitda_margin_0,
    interest_payment,
    TOTAL_EMISSIONS,
    total_assets,
    high_carbon_assets
):
    return {
        "carbon_pass_through": 0.45,
        "demand_elasticity": -0.30,
        "price_elasticity": -0.05,
        "beta_trans": 1.05,
        "abatement_potential": 0.35
    }

# ============================================================
# PART 2 — SIDEBAR INPUTS + MODULE CONTROL + NGFS DATA
# ============================================================

# ------------------------------------------------------------
# SIDEBAR — COMPANY BASELINE INPUTS (COMMON TO ALL MODULES)
# ------------------------------------------------------------
st.sidebar.header("🏢 Company Baseline Inputs")

company_name = st.sidebar.text_input("Company Name", "Sample Company")

sector = st.sidebar.selectbox(
    "Sector",
    ["Steel", "Power", "Cement", "Oil & Gas", "Manufacturing"]
)

REPORTING_YEAR = st.sidebar.selectbox(
    "Reporting Year",
    [2025, 2030, 2035, 2040]
)

# ---------------- Financials ----------------
revenue_0 = st.sidebar.number_input("Base Revenue (₹ Cr)", value=10_000.0)
ebitda_margin_0 = st.sidebar.slider("Base EBITDA Margin", 0.05, 0.40, 0.22)
interest_payment = st.sidebar.number_input("Annual Interest Expense (₹ Cr)", value=600.0)
total_assets = st.sidebar.number_input("Total Assets (₹ Cr)", value=25_000.0)
exposure_at_default = st.sidebar.number_input("Exposure at Default (₹ Cr)", value=8_000.0)

# ---------------- Emissions (Scope-wise) ----------------
st.sidebar.subheader("GHG Emissions (tCO₂e)")

scope1 = st.sidebar.number_input("Scope 1 Emissions", value=2_000_000.0)
scope2 = st.sidebar.number_input("Scope 2 Emissions", value=1_500_000.0)
scope3 = st.sidebar.number_input("Scope 3 Emissions", value=1_500_000.0)

TOTAL_EMISSIONS = scope1 + scope2 + scope3

st.session_state["scope1"] = scope1
st.session_state["scope2"] = scope2
st.session_state["scope3"] = scope3

high_carbon_assets = st.sidebar.number_input(
    "High-carbon Assets (₹ Cr)", value=6_000.0
)

# ------------------------------------------------------------
# MODULE SELECTION — SINGLE CONTROL TOWER (INTENT ONLY)
# ------------------------------------------------------------
st.sidebar.header("⚙️ Modules to Run")

st.session_state["enable_transition"] = st.sidebar.checkbox(
    "Transition Risk", value=True
)
st.session_state["enable_physical"] = st.sidebar.checkbox("Physical Risk")
st.session_state["enable_targets"] = st.sidebar.checkbox("Transition Targets")
st.session_state["enable_brsr"] = st.sidebar.checkbox("BRSR Diagnostics")

# ------------------------------------------------------------
# TRANSITION PARAMETERS — SHOWN ONLY IF ENABLED
# ------------------------------------------------------------
if st.session_state["enable_transition"]:

    st.sidebar.divider()
    st.sidebar.header("Transition Risk Parameters")

    USD_INR = st.sidebar.number_input("USD → INR Exchange Rate", value=83.0)

    carbon_pass_through = st.sidebar.slider(
        "Carbon Pass-through",
        *sector_ranges["carbon_pass_through"],
        np.mean(sector_ranges["carbon_pass_through"])
    )

    demand_elasticity = st.sidebar.slider(
        "Demand Elasticity",
        *sector_ranges["demand_elasticity"],
        np.mean(sector_ranges["demand_elasticity"])
    )

    price_elasticity = st.sidebar.slider(
        "Price Elasticity",
        *sector_ranges["price_elasticity"],
        np.mean(sector_ranges["price_elasticity"])
    )

    beta_trans = st.sidebar.slider(
        "Carbon Transition Sensitivity (β)",
        0.6, 1.6, 1.0, 0.05
    )

    planned_capex = st.sidebar.number_input(
        "Planned CAPEX (₹ Cr)", value=1_200.0
    )

    abatement_cost = st.sidebar.number_input(
        "Abatement Cost (₹ / tCO₂)", value=4_500.0
    )

    abatement_potential = st.sidebar.slider(
        "Abatement Potential",
        *sector_ranges["abatement_potential"],
        np.mean(sector_ranges["abatement_potential"])
    )

    # Credit & governance
    base_pd = st.sidebar.slider("Base PD", 0.0, 0.10, 0.015)
    LGD_0 = st.sidebar.slider("Base LGD", 0.0, 1.0, 0.45)
    tax_rate = st.sidebar.slider("Tax Rate", 0.0, 0.50, 0.25)

    g1 = st.sidebar.slider("Board Oversight", 0.0, 1.0, 0.7)
    g2 = st.sidebar.slider("CAPEX Alignment", 0.0, 1.0, 0.5)
    g3 = st.sidebar.slider("Incentives", 0.0, 1.0, 0.4)
    g4 = st.sidebar.slider("Internal Carbon Price", 0.0, 1.0, 0.6)

    G = np.mean([g1, g2, g3, g4])

# ------------------------------------------------------------
# MODEL CONSTANTS (GLOBAL)
# ------------------------------------------------------------
P_START = 50
P_FULL = 250
MAX_STRANDING = 0.80

margin_erosion_rate = 0.005
margin_floor = 0.10

alpha_dscr = 0.5
gamma_phys = 0.5

# ------------------------------------------------------------
# NGFS SCENARIO DATA LOAD (REQUIRED BEFORE ENGINE)
# ------------------------------------------------------------
@st.cache_data
def load_ngfs():
    df = pd.read_csv(r"Data/ngfs_scenarios.csv")
    df.columns = df.columns.str.strip().str.replace("﻿", "", regex=False)
    return df

df_ngfs = load_ngfs()

for c in df_ngfs.select_dtypes(include="object"):
    df_ngfs[c] = df_ngfs[c].astype(str).str.strip()

keywords = [
    "Price|Carbon",
    "GDP|MER|Counterfactual without damage",
    "AR6 climate diagnostics|Surface Temperature"
]

df_ngfs = df_ngfs[
    df_ngfs["Variable"].apply(
        lambda x: any(k.lower() in x.lower() for k in keywords)
    )
]

year_cols = [c for c in df_ngfs.columns if c.isdigit()]

df_long = df_ngfs.melt(
    id_vars=[c for c in df_ngfs.columns if c not in year_cols],
    value_vars=year_cols,
    var_name="Year",
    value_name="Value"
)

df_long["Year"] = df_long["Year"].astype(int)

# ------------------------------------------------------------
# 🔑 TRANSITION RISK ENGINE — DEFINED HERE (FIXES NameError)
# ------------------------------------------------------------
def run_transition_engine(
    df_long,
    selected_scenarios,
    revenue_0,
    ebitda_margin_0,
    interest_payment,
    TOTAL_EMISSIONS,
    high_carbon_assets,
    exposure_at_default,
    carbon_pass_through,
    demand_elasticity,
    price_elasticity,
    beta_trans,
    USD_INR,
    G,
    base_pd,
    LGD_0,
    abatement_cost,
    abatement_potential,
    planned_capex,
    P_START,
    P_FULL,
    MAX_STRANDING,
    margin_erosion_rate,
    margin_floor,
    alpha_dscr,
    tax_rate,
    baseline_temp=1.5,
    temp_sensitivity=0.02,
    gamma_phys=0.5
):
    """
    Transition Risk Engine
    ----------------------
    Authoritative NGFS-driven transition risk model.
    """

    results = []

    for scen in selected_scenarios:
        d = df_long[df_long["Scenario"] == scen]

        for y in sorted(d["Year"].unique()):

            try:
                carbon_price = d.loc[
                    (d["Year"] == y) &
                    d["Variable"].str.contains("Carbon", case=False),
                    "Value"
                ].iloc[0]

                temp = d.loc[
                    (d["Year"] == y) &
                    d["Variable"].str.contains("Temperature", case=False),
                    "Value"
                ].iloc[0]

                gdp = d.loc[
                    (d["Year"] == y) &
                    d["Variable"].str.contains("GDP", case=False),
                    "Value"
                ].iloc[0]

            except Exception:
                continue

            try:
                gdp_prev = d.loc[
                    (d["Year"] == y - 5) &
                    d["Variable"].str.contains("GDP", case=False),
                    "Value"
                ].iloc[0]
                gdp_growth = (gdp - gdp_prev) / gdp_prev
            except Exception:
                gdp_growth = 0.0

            gross_carbon_cost = (
                TOTAL_EMISSIONS * carbon_price * USD_INR / 1e7
            )

            net_carbon_cost = gross_carbon_cost * (1 - carbon_pass_through)

            revenue = revenue_0 * (1 + demand_elasticity * gdp_growth)
            revenue *= (1 + price_elasticity * (net_carbon_cost / max(revenue, 1)))

            physical_loss = temp_sensitivity * max(0, temp - baseline_temp)
            revenue *= (1 - physical_loss)

            t_index = max(0, y - 2025)
            margin_t = max(
                margin_floor,
                ebitda_margin_0 * (1 - margin_erosion_rate * t_index)
            )

            ebitda = revenue * margin_t - net_carbon_cost
            ebitda_adj = ebitda * (1 - (1 - G) * 0.15)

            dscr = ebitda_adj / max(interest_payment, 1e-6)
            carbon_burden = net_carbon_cost / max(revenue, 1)

            pd_t = (
                base_pd
                * (1 + alpha_dscr * max(0, 1.5 - dscr))
                * (1 + beta_trans * carbon_burden)
                * (1 + gamma_phys * physical_loss)
            )
            pd_t = np.clip(pd_t, 0, 1)

            lgd_t = np.clip(LGD_0 * (1 + 0.2 * carbon_burden), 0, 1)
            ecl = pd_t * lgd_t * exposure_at_default / 1e3

            ratio = (
                0 if carbon_price < P_START
                else min((carbon_price - P_START) / (P_FULL - P_START), 1)
            )

            stranded_assets = high_carbon_assets * min(ratio, MAX_STRANDING)

            required_capex = (
                TOTAL_EMISSIONS * abatement_potential * abatement_cost / 1e7
            )

            capex_gap = required_capex - planned_capex

            results.append({
                "Scenario": scen,
                "Year": y,
                "Revenue": revenue,
                "EBITDA": ebitda_adj,
                "EBITDA_Margin": ebitda_adj / max(revenue, 1),
                "Carbon_Burden": carbon_burden,
                "DSCR": dscr,
                "PD_Transition": pd_t,
                "LGD": lgd_t,
                "ECL_Transition": ecl,
                "Stranded_Assets": stranded_assets,
                "CAPEX_Gap": capex_gap
            })

    df_transition = pd.DataFrame(results)

    if df_transition.empty:
        raise ValueError("Transition Risk Engine produced no outputs.")

    return df_transition.sort_values(["Scenario", "Year"]).reset_index(drop=True)
# ============================================================
# PART 3 — TRANSITION RISK ENGINE (TAB-DRIVEN)
# ============================================================

transition_tab, physical_tab, targets_tab, integrated_tab, plots_tab, brsr_tab, ai_tab = st.tabs([
    "⚡ Transition Risk",
    "🌍 Physical Risk",
    "🎯 Transition Targets",
    "🧩 Integrated Risk",
    "📈 Plots",
    "📘 BRSR Core",
    "🤖 AI Narrative"
])


with transition_tab:

    st.header("⚡ Transition Risk Engine")
    st.caption(
        "Baseline transition risk assessment using NGFS scenarios. "
        "This step must be completed before physical integration or targets."
    )

    # --------------------------------------------------------
    # HARD GUARD — ENABLE FLAG
    # --------------------------------------------------------
    if not st.session_state.get("enable_transition", False):
        st.info("ℹ️ Transition Risk module is not enabled in the sidebar.")
    
    else:
        all_scenarios = sorted(df_long["Scenario"].unique())

        selected_scenarios = st.multiselect(
            "Select NGFS Scenarios",
            options=all_scenarios,
            default=all_scenarios
        )

        run_transition = st.button("▶ Run Transition Risk Engine")

        if not run_transition:
            st.info("Select scenarios and click **Run Transition Risk Engine**.")
        
        elif not selected_scenarios:
            st.warning("Please select at least one NGFS scenario.")
        
        else:
            # ------------------------------------------------
            # ENGINE EXECUTION
            # ------------------------------------------------
            df_transition = run_transition_engine(
                df_long=df_long,
                selected_scenarios=selected_scenarios,
                revenue_0=revenue_0,
                ebitda_margin_0=ebitda_margin_0,
                interest_payment=interest_payment,
                TOTAL_EMISSIONS=TOTAL_EMISSIONS,
                high_carbon_assets=high_carbon_assets,
                exposure_at_default=exposure_at_default,
                carbon_pass_through=carbon_pass_through,
                demand_elasticity=demand_elasticity,
                price_elasticity=price_elasticity,
                beta_trans=beta_trans,
                USD_INR=USD_INR,
                G=G,
                base_pd=base_pd,
                LGD_0=LGD_0,
                abatement_cost=abatement_cost,
                abatement_potential=abatement_potential,
                planned_capex=planned_capex,
                P_START=P_START,
                P_FULL=P_FULL,
                MAX_STRANDING=MAX_STRANDING,
                margin_erosion_rate=margin_erosion_rate,
                margin_floor=margin_floor,
                alpha_dscr=alpha_dscr,
                tax_rate=tax_rate
            )

            # ------------------------------------------------
            # AUTHORITATIVE STORAGE (UNCHANGED)
            # ------------------------------------------------
            st.session_state["df_transition"] = df_transition
            st.session_state["transition_ran"] = True

            st.success("✅ Transition Risk Engine executed successfully")

            st.subheader("🔍 Sample Engine Output")
            st.dataframe(df_transition.head(10), use_container_width=True)

            # =================================================
            # 🔹 NEW — SCENARIO-LEVEL TRANSITION RISK SUMMARY
            # (ISSB S2 / RBI CORE DISCLOSURE)
            # =================================================
            df_transition_summary = (
                df_transition
                .groupby("Scenario")
                .agg({
                    "Carbon_Burden": "max",
                    "EBITDA_Margin": "min",
                    "DSCR": "min",
                    "PD_Transition": "max",
                    "ECL_Transition": "max",
                    "Stranded_Assets": "max",
                    "CAPEX_Gap": "max"
                })
                .round(3)
                .reset_index()
            )

            st.subheader("📊 Scenario-Level Transition Risk Summary")
            st.caption(
                "Worst-case financial and credit risk metrics across the projection horizon "
                "(ISSB S2 / RBI-aligned)."
            )
            st.dataframe(df_transition_summary, use_container_width=True)

            # =================================================
            # 🔹 NEW — WORST-CASE YEAR PER SCENARIO
            # (SUPERVISORY STRESS TEST VIEW)
            # =================================================
            df_transition_worst = (
                df_transition
                .sort_values("PD_Transition", ascending=False)
                .groupby("Scenario")
                .head(1)
                .reset_index(drop=True)
            )

            st.subheader("⚠️ Worst-Case Year by Scenario")
            st.caption(
                "Year in which transition risk impact peaks for each scenario "
                "(used for stress testing and capital assessment)."
            )

            st.dataframe(
                df_transition_worst[
                    [
                        "Scenario",
                        "Year",
                        "PD_Transition",
                        "ECL_Transition",
                        "EBITDA_Margin",
                        "DSCR",
                        "Stranded_Assets",
                        "CAPEX_Gap"
                    ]
                ],
                use_container_width=True
            )

            # =================================================
            # 🔹 NEW — HEADLINE TRANSITION RISK INDICATORS
            # (BOARD / CXO VIEW)
            # =================================================
            st.subheader("📌 Headline Transition Risk Indicators")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Max PD (Transition)",
                f"{df_transition['PD_Transition'].max():.2%}"
            )

            c2.metric(
                "Max ECL (₹ Cr)",
                f"{df_transition['ECL_Transition'].max():,.1f}"
            )

            c3.metric(
                "Min DSCR",
                f"{df_transition['DSCR'].min():.2f}"
            )

            c4.metric(
                "Max Stranded Assets (₹ Cr)",
                f"{df_transition['Stranded_Assets'].max():,.0f}"
            )

            # =================================================
            # 🔹 NEW — STORE AGGREGATES FOR INTEGRATION & AI
            # =================================================
            st.session_state["df_transition_summary"] = df_transition_summary
            st.session_state["df_transition_worst"] = df_transition_worst

# ============================================================
# PART 4 — 📈 RISK VISUALISATIONS (PRECOMPUTED & INTERACTIVE)
# ============================================================

with plots_tab:

    st.header("📈 Risk Visual Diagnostics")
    st.caption(
        "Scenario-selectable, regulator-grade visual diagnostics derived strictly "
        "from executed risk engines (no recomputation)."
    )

    transition_ran = bool(st.session_state.get("transition_ran", False))
    physical_ran   = bool(st.session_state.get("physical_ran", False))
    targets_ran    = bool(st.session_state.get("targets_ran", False))

    if not (transition_ran or physical_ran or targets_ran):
        st.info(
            "ℹ️ No plots available yet.\n\n"
            "Run at least one engine (Transition, Physical, or Targets)."
        )
        st.stop()

    # ========================================================
    # ⚙️ COMMON SCENARIO SELECTION (TRANSITION / TARGETS ONLY)
    # ========================================================
    scenario_options = []

    if transition_ran:
        df_transition = st.session_state.get("df_transition")
        if isinstance(df_transition, pd.DataFrame) and not df_transition.empty:
            scenario_options = sorted(df_transition["Scenario"].unique())

    selected_scenarios = st.multiselect(
        "Select scenarios for visual analysis (Transition / Targets)",
        options=scenario_options,
        default=scenario_options
    )

    # ========================================================
    # ⚡ TRANSITION RISK — CORE TRANSMISSION CHANNELS
    # ========================================================
    if transition_ran and selected_scenarios:

        st.subheader("⚡ Transition Risk — Financial & Credit Transmission")
        st.caption("**ISSB S2 §§14–16 · IFRS S2 · RBI Climate Stress Testing**")

        df_t = st.session_state.get("df_transition")
        if not isinstance(df_t, pd.DataFrame) or df_t.empty:
            st.warning("Transition risk results unavailable.")
        else:
            df_p = df_t[df_t["Scenario"].isin(selected_scenarios)]

            # ---------------- PD Trajectory ----------------
            st.markdown("### 📉 Probability of Default (PD) Trajectory")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            sns.lineplot(
                data=df_p,
                x="Year",
                y="PD_Transition",
                hue="Scenario",
                marker="o",
                linewidth=2,
                ax=ax
            )
            ax.set_ylabel("Probability of Default")
            ax.legend(fontsize=8, frameon=False, loc="best")
            st.pyplot(fig)

            # ---------------- ECL Trajectory ----------------
            st.markdown("### 💰 Expected Credit Loss (ECL) Trajectory")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            sns.lineplot(
                data=df_p,
                x="Year",
                y="ECL_Transition",
                hue="Scenario",
                marker="o",
                linewidth=2,
                ax=ax
            )
            ax.set_ylabel("ECL (₹ Cr)")
            ax.legend(fontsize=8, frameon=False, loc="best")
            st.pyplot(fig)

            # ---------------- Carbon vs EBITDA ----------------
            st.markdown("### 🧮 Carbon Burden vs EBITDA Margin")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            sns.scatterplot(
                data=df_p,
                x="Carbon_Burden",
                y="EBITDA_Margin",
                hue="Scenario",
                s=60,
                alpha=0.8,
                ax=ax
            )
            ax.legend(fontsize=8, frameon=False, loc="best")
            st.pyplot(fig)

            # ---------------- DSCR Stress ----------------
            st.markdown("### 🏦 DSCR Stress Trajectory")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            sns.lineplot(
                data=df_p,
                x="Year",
                y="DSCR",
                hue="Scenario",
                marker="o",
                linewidth=2,
                ax=ax
            )
            ax.axhline(1.2, color="red", linestyle="--", linewidth=1)
            ax.legend(fontsize=8, frameon=False, loc="best")
            st.pyplot(fig)

            # ---------------- CAPEX Gap ----------------
            st.markdown("### 🏗️ CAPEX Adequacy Gap")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            sns.lineplot(
                data=df_p,
                x="Year",
                y="CAPEX_Gap",
                hue="Scenario",
                marker="o",
                linewidth=2,
                ax=ax
            )
            ax.legend(fontsize=8, frameon=False, loc="best")
            st.pyplot(fig)

    # ========================================================
    # 🎯 TRANSITION TARGETS — EFFECTIVENESS
    # ========================================================
    if targets_ran and selected_scenarios:

        st.subheader("🎯 Transition Targets — Effectiveness")
        st.caption("**ISSB S2 §§18–20 · GFANZ · TPT**")

        df_b = st.session_state.get("df_transition")
        df_tg = st.session_state.get("df_target")

        if not isinstance(df_b, pd.DataFrame) or not isinstance(df_tg, pd.DataFrame):
            st.warning("Target or baseline results unavailable.")
        else:
            df_b = df_b[df_b["Scenario"].isin(selected_scenarios)]
            df_tg = df_tg[df_tg["Scenario"].isin(selected_scenarios)]

            # ---------------- PD Baseline vs Target ----------------
            st.markdown("### 📉 PD — Baseline vs Target")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            for s in selected_scenarios:
                ax.plot(
                    df_b[df_b["Scenario"] == s].groupby("Year")["PD_Transition"].mean(),
                    linestyle="--",
                    linewidth=2,
                    label=f"{s} Baseline"
                )
                ax.plot(
                    df_tg[df_tg["Scenario"] == s].groupby("Year")["PD_Target"].mean(),
                    linewidth=2,
                    label=f"{s} Target"
                )

            ax.legend(fontsize=8, frameon=False, loc="best")
            st.pyplot(fig)

            # ---------------- ECL Baseline vs Target ----------------
            st.markdown("### 💰 ECL — Baseline vs Target")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            for s in selected_scenarios:
                ax.plot(
                    df_b[df_b["Scenario"] == s].groupby("Year")["ECL_Transition"].mean(),
                    linestyle="--",
                    linewidth=2,
                    label=f"{s} Baseline"
                )
                ax.plot(
                    df_tg[df_tg["Scenario"] == s].groupby("Year")["ECL_Target"].mean(),
                    linewidth=2,
                    label=f"{s} Target"
                )

            ax.legend(fontsize=8, frameon=False, loc="best")
            st.pyplot(fig)

            # ---------------- Ambition vs EBITDA ----------------
            st.markdown("### 🎯 Target Ambition vs EBITDA Impact")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            sns.scatterplot(
                data=df_tg,
                x="PD_Target",
                y="EBITDA_Target",
                hue="Scenario",
                s=60,
                alpha=0.8,
                ax=ax
            )
            ax.legend(fontsize=8, frameon=False, loc="best")
            st.pyplot(fig)

    # ========================================================
    # 🌍 PHYSICAL RISK — REPORTING-YEAR-LOCKED VIEW
    # ========================================================
    if physical_ran:

        st.subheader("🌍 Physical Risk — Reporting Year View")
        st.caption(
            "**ISSB S2 §§22–25 · RBI Physical Risk Stress Testing**  \n"
            "Physical risk results shown strictly for the reporting / projection year "
            "selected in the Physical Risk module."
        )

        phys = st.session_state.get("phys_assets")

        if not isinstance(phys, pd.DataFrame) or phys.empty:
            st.warning("Physical asset results unavailable.")
        else:
            hazard_choice = st.selectbox(
                "Select physical hazard map",
                ["Flood", "Heat", "Cyclone", "Integrated Physical Risk"],
                key="phys_hazard_map"
            )

            m = folium.Map(
                location=[phys["latitude"].mean(), phys["longitude"].mean()],
                zoom_start=5,
                tiles="cartodbpositron"
            )

            for _, r in phys.iterrows():

                val = (
                    r["H_flood"] if hazard_choice == "Flood"
                    else r["H_heat"] if hazard_choice == "Heat"
                    else r["H_cyclone"] if hazard_choice == "Cyclone"
                    else r["D_total"]
                )

                color = "green" if val < 0.2 else "orange" if val < 0.4 else "red"

                folium.CircleMarker(
                    location=[r["latitude"], r["longitude"]],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"{r['asset_id']} | Risk Index: {val:.2f}"
                ).add_to(m)

            st_folium(m, width=650, height=380)

            # ---------------- Baseline Revenue Loss ----------------
            st.markdown("### 💸 Baseline Revenue Loss by Asset")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            sns.barplot(data=phys, x="asset_id", y="revenue_loss", ax=ax)
            ax.set_ylabel("Revenue Loss (₹ Cr)")
            st.pyplot(fig)

            # ---------------- Damage vs Loss ----------------
            st.markdown("### ⚠️ Damage Severity vs Financial Loss")

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            sns.scatterplot(data=phys, x="D_total", y="revenue_loss", s=60, ax=ax)
            st.pyplot(fig)

            # ----------------------------------------------------
            # 🌡️ NGFS PHYSICAL PROJECTION — YEAR FILTER ONLY
            # ----------------------------------------------------
            df_phys_proj = st.session_state.get("df_physical_projection")

            if isinstance(df_phys_proj, pd.DataFrame) and not df_phys_proj.empty:

                available_years = sorted(df_phys_proj["Projection_Year"].unique())

                selected_year = st.selectbox(
                    "Select reporting / projection year",
                    options=available_years,
                    key="phys_proj_year"
                )

                df_py = df_phys_proj[
                    df_phys_proj["Projection_Year"] == selected_year
                ]

                st.subheader(f"🌡️ NGFS Physical Risk Outcomes — {selected_year}")

                # ---------------- Revenue Loss ----------------
                st.markdown("### 📉 Projected Revenue Loss")

                fig, ax = plt.subplots(figsize=(5.2, 3.2))
                sns.barplot(
                    data=df_py,
                    x="Scenario",
                    y="Revenue_Loss (₹ Cr)",
                    ax=ax
                )
                st.pyplot(fig)

                # ---------------- PD ----------------
                st.markdown("### 📉 Physical Risk — Probability of Default")

                fig, ax = plt.subplots(figsize=(5.2, 3.2))
                sns.barplot(
                    data=df_py,
                    x="Scenario",
                    y="PD_Physical",
                    ax=ax
                )
                st.pyplot(fig)

                # ---------------- ΔECL ----------------
                st.markdown("### 💰 Incremental ECL from Physical Risk")

                fig, ax = plt.subplots(figsize=(5.2, 3.2))
                sns.barplot(
                    data=df_py,
                    x="Scenario",
                    y="ΔECL (₹ Cr)",
                    ax=ax
                )
                st.pyplot(fig)

# ============================================================
# PART 5 — 🌍 PHYSICAL RISK ENGINE (NON-BLOCKING & SAFE)
# ============================================================

with physical_tab:

    st.header("🌍 Physical Risk Assessment (Asset-Level)")
    st.caption(
        "Asset-level physical climate risk assessment covering floods, heat stress, "
        "and cyclones. This module is optional and does not block other engines."
    )

    # --------------------------------------------------------
    # ENABLE FLAG — NON-BLOCKING
    # --------------------------------------------------------
    if not st.session_state.get("enable_physical", False):
        st.info("ℹ️ Physical Risk module is disabled. Enable it from the sidebar to run.")

    else:
        # ----------------------------------------------------
        # ASSET REGISTER
        # ----------------------------------------------------
        st.subheader("🏭 Asset Register")

        asset_df = st.data_editor(
            pd.DataFrame({
                "asset_id": ["A1", "A2", "A3"],
                "asset_type": ["Steel Plant", "Rolling Mill", "Port Logistics Yard"],
                "latitude": [22.80, 23.55, 20.32],
                "longitude": [86.20, 87.32, 86.61],
                "revenue_inr_cr": [3500.0, 2000.0, 1200.0]
            }),
            num_rows="dynamic",
            use_container_width=True
        )

        # ----------------------------------------------------
        # FINANCIAL ASSUMPTIONS
        # ----------------------------------------------------
        st.subheader("💰 Financial Assumptions")

        EBITDA_MARGIN_PHYS = float(
            st.number_input(
                "EBITDA Margin",
                value=float(ebitda_margin_0),
                step=0.01
            )
        )

        INTEREST_PHYS = float(
            st.number_input(
                "Annual Interest Expense (₹ Cr)",
                value=float(interest_payment),
                step=1.0,
                min_value=1.0
            )
        )

        # ----------------------------------------------------
        # BASE PD — SAFE RESOLUTION
        # ----------------------------------------------------
        if st.session_state.get("transition_ran", False):
            df_t = st.session_state.get("df_transition")
            if isinstance(df_t, pd.DataFrame) and not df_t.empty:
                base_pd_scalar = float(df_t["PD_Transition"].iloc[0])
            else:
                base_pd_scalar = 0.015
        else:
            base_pd_scalar = 0.015

        PD_BASE_PHYS = float(
            st.number_input(
                "Base PD (Physical)",
                value=base_pd_scalar,
                step=0.001,
                min_value=0.0,
                max_value=1.0
            )
        )

        LGD_PHYS   = float(LGD_0)
        EAD_PHYS   = float(exposure_at_default)
        GAMMA_PHYS = float(gamma_phys)

        # ----------------------------------------------------
        # 🌡️ NGFS SCENARIO PROJECTION (ADDITIVE)
        # ----------------------------------------------------
        st.subheader("🌡️ NGFS Scenario-Based Physical Risk Projection")

        use_ngfs_physical = st.checkbox(
            "Project physical risk across NGFS scenarios & years",
            value=False
        )

        phys_scenario = None
        phys_years = []

        if use_ngfs_physical:
            phys_scenario = st.selectbox(
                "NGFS Scenario",
                options=sorted(df_long["Scenario"].unique())
            )

            phys_years = st.multiselect(
                "Projection Years",
                options=sorted(df_long["Year"].unique()),
                default=[
                    y for y in sorted(df_long["Year"].unique())
                    if y >= REPORTING_YEAR
                ]
            )

        # ----------------------------------------------------
        # EXECUTION BUTTON
        # ----------------------------------------------------
        run_phys = st.button("▶ Run Physical Risk Engine")

        if run_phys:

            # ====================================================
            # BASELINE PHYSICAL RISK ENGINE
            # ====================================================
            df = asset_df.copy()

            V = {"flood": 0.6, "heat": 0.7, "cyclone": 0.6}
            KAPPA = {"flood": 0.4, "heat": 0.2, "cyclone": 0.5}

            MAX_DOWNTIME = {
                "Steel Plant": 90,
                "Rolling Mill": 60,
                "Port Logistics Yard": 45
            }

            # ---------------- FLOOD ----------------
            FLOOD_RASTER = r"Data/floodMapGL_rp100y.tif"
            BUFFER_KM = 5
            flood_vals = []

            with rasterio.open(FLOOD_RASTER) as src:
                arr = src.read(1)
                pixel_km = abs(src.res[0]) * 111
                buffer_px = int(BUFFER_KM / pixel_km)

                for _, a in df.iterrows():
                    try:
                        r, c = rowcol(src.transform, a["longitude"], a["latitude"])
                        window = arr[
                            max(0, r-buffer_px):min(arr.shape[0], r+buffer_px),
                            max(0, c-buffer_px):min(arr.shape[1], c+buffer_px)
                        ]
                        valid = window[(window > 0) & (window < 100)]
                        flood_vals.append(float(valid.max()) if valid.size else 0.0)
                    except Exception:
                        flood_vals.append(0.0)

            df["flood_depth_m"] = flood_vals

            # ---------------- HEAT ----------------
            heat = pd.read_csv(
                r"Data/era5_test_day_grid.csv"
            )
            heat["exceed"] = (heat["temp_c"] >= 35).astype(int)

            heat_grid = heat.groupby(["lat", "lon"], as_index=False)["exceed"].sum()

            df["heat_days"] = df.apply(
                lambda r: int(
                    heat_grid.iloc[
                        ((heat_grid["lat"] - r["latitude"])**2 +
                         (heat_grid["lon"] - r["longitude"])**2).idxmin()
                    ]["exceed"]
                ),
                axis=1
            )

            # ---------------- CYCLONE ----------------
            cyc = pd.read_csv(
                r"Data/ibtracs.NI.list.v04r01.csv"
            )[["LAT", "LON", "USA_WIND"]]

            cyc = cyc.apply(pd.to_numeric, errors="coerce").dropna()
            cyc["wind_kmh"] = cyc["USA_WIND"] * 1.852

            def hav(lat1, lon1, lat2, lon2):
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                a = (
                    sin((lat2 - lat1) / 2)**2
                    + cos(lat1) * cos(lat2)
                    * sin((lon2 - lon1) / 2)**2
                )
                return 6371 * 2 * asin(sqrt(a))

            df["max_wind_kmh"] = df.apply(
                lambda a: cyc.loc[
                    cyc.apply(
                        lambda r: hav(
                            a["latitude"], a["longitude"],
                            r["LAT"], r["LON"]
                        ),
                        axis=1
                    ) <= 100,
                    "wind_kmh"
                ].max() if not cyc.empty else 0.0,
                axis=1
            )

            # ---------------- DAMAGE ----------------
            df["H_flood"]   = (df["flood_depth_m"] / 3).clip(0, 1)
            df["H_heat"]    = (df["heat_days"] / 30).clip(0, 1)
            df["H_cyclone"] = (df["max_wind_kmh"] / 200).clip(0, 1)

            df["D_total"] = (
                df["H_flood"]   * V["flood"]   * KAPPA["flood"]
                + df["H_heat"]  * V["heat"]    * KAPPA["heat"]
                + df["H_cyclone"] * V["cyclone"] * KAPPA["cyclone"]
            ).clip(0, 1)

            df["downtime_days"] = df.apply(
                lambda r: r["D_total"]
                * MAX_DOWNTIME.get(r["asset_type"], 60),
                axis=1
            )

            # ---------------- FINANCIALS ----------------
            df["revenue_loss"] = (
                df["downtime_days"] / 365 * df["revenue_inr_cr"]
            )

            TOTAL_REV_LOSS = df["revenue_loss"].sum()
            DELTA_EBITDA   = TOTAL_REV_LOSS * EBITDA_MARGIN_PHYS
            TOTAL_EBITDA   = df["revenue_inr_cr"].sum() * EBITDA_MARGIN_PHYS

            DSCR_PHYS = (
                (TOTAL_EBITDA - DELTA_EBITDA)
                / max(INTEREST_PHYS, 1e-6)
            )

            PD_PHYS = PD_BASE_PHYS * (
                1 + GAMMA_PHYS * max(0, 1.5 - DSCR_PHYS)
            )

            DELTA_ECL = (
                (PD_PHYS - PD_BASE_PHYS)
                * LGD_PHYS
                * EAD_PHYS
            )

            # ----------------------------------------------------
            # STORE BASELINE RESULTS
            # ----------------------------------------------------
            st.session_state["phys_assets"] = df
            st.session_state["phys_summary"] = {
                "Total Revenue Loss (₹ Cr)": TOTAL_REV_LOSS,
                "EBITDA Loss (₹ Cr)": DELTA_EBITDA,
                "Post-Risk DSCR": DSCR_PHYS,
                "Base PD": PD_BASE_PHYS,
                "Physical Risk PD": PD_PHYS,
                "ΔECL (₹ Cr)": DELTA_ECL
            }
            st.session_state["physical_ran"] = True

            # ----------------------------------------------------
            # 🌡️ NGFS SCENARIO × YEAR PROJECTION (SAFE)
            # ----------------------------------------------------
            df_phys_proj = []

            if use_ngfs_physical and phys_years and phys_scenario:

                base_temp_rows = df_long.loc[
                    (df_long["Scenario"] == phys_scenario)
                    & (df_long["Year"] == REPORTING_YEAR)
                    & df_long["Variable"].str.contains(
                        "Temperature", case=False
                    ),
                    "Value"
                ]

                if not base_temp_rows.empty:
                    BASE_TEMP = float(base_temp_rows.iloc[0])

                    for y in phys_years:
                        temp_rows = df_long.loc[
                            (df_long["Scenario"] == phys_scenario)
                            & (df_long["Year"] == y)
                            & df_long["Variable"].str.contains(
                                "Temperature", case=False
                            ),
                            "Value"
                        ]

                        if temp_rows.empty:
                            continue

                        temp_y = float(temp_rows.iloc[0])
                        delta_T = max(0, temp_y - BASE_TEMP)

                        loss_multiplier = 1 + 0.25 * delta_T

                        proj_loss = TOTAL_REV_LOSS * loss_multiplier
                        proj_ebitda_loss = proj_loss * EBITDA_MARGIN_PHYS

                        proj_dscr = (
                            (TOTAL_EBITDA - proj_ebitda_loss)
                            / max(INTEREST_PHYS, 1e-6)
                        )

                        proj_pd = PD_BASE_PHYS * (
                            1 + GAMMA_PHYS * max(0, 1.5 - proj_dscr)
                        )

                        proj_delta_ecl = (
                            (proj_pd - PD_BASE_PHYS)
                            * LGD_PHYS
                            * EAD_PHYS
                        )

                        df_phys_proj.append({
                            "Scenario": phys_scenario,
                            "Reporting_Year": REPORTING_YEAR,
                            "Projection_Year": y,
                            "ΔTemperature (°C)": round(delta_T, 3),
                            "Revenue_Loss (₹ Cr)": round(proj_loss, 2),
                            "DSCR": round(proj_dscr, 3),
                            "PD_Physical": round(proj_pd, 4),
                            "ΔECL (₹ Cr)": round(proj_delta_ecl, 2)
                        })

            df_phys_proj = pd.DataFrame(df_phys_proj)
            st.session_state["df_physical_projection"] = df_phys_proj

            # ----------------------------------------------------
            # OUTPUTS
            # ----------------------------------------------------
            st.success("✅ Physical Risk Engine executed successfully")

            st.subheader("📊 Physical Risk — Portfolio Summary")
            st.json(st.session_state["phys_summary"])

            st.subheader("🏭 Asset-Level Physical Risk Metrics")
            st.dataframe(
                df[
                    [
                        "asset_id",
                        "asset_type",
                        "flood_depth_m",
                        "heat_days",
                        "max_wind_kmh",
                        "D_total",
                        "downtime_days",
                        "revenue_loss"
                    ]
                ].round(3),
                use_container_width=True
            )

            if use_ngfs_physical and not df_phys_proj.empty:
                st.subheader("📈 NGFS Scenario-Based Physical Risk Projection")
                st.dataframe(df_phys_proj, use_container_width=True)

        else:
            st.info("Configure assets and click **Run Physical Risk Engine**.")

# ============================================================
# PART 6 — 🎯 TRANSITION TARGETS & FINANCIAL IMPACT (NON-BLOCKING)
# ============================================================

with targets_tab:

    st.header("🎯 Transition Targets & Financial Impact")
    st.caption(
        "Management-defined transition targets applied to baseline "
        "transition risk outputs."
    )

    # --------------------------------------------------------
    # ENABLE FLAG (NON-BLOCKING)
    # --------------------------------------------------------
    if not st.session_state.get("enable_targets", False):
        st.info("ℹ️ Transition Targets module not enabled.")
    
    elif not st.session_state.get("transition_ran", False):
        st.warning("⚠️ Run Transition Risk before applying targets.")
    
    else:
        df_base = st.session_state.get("df_transition")

        if not isinstance(df_base, pd.DataFrame) or df_base.empty:
            st.warning("Baseline transition results unavailable.")

        else:
            # ------------------------------------------------
            # TARGET INPUTS
            # ------------------------------------------------
            st.subheader("📌 Target Assumptions")

            c1, c2, c3 = st.columns(3)

            with c1:
                s_em = st.slider("Short-term Emissions Reduction (%)", 0, 100, 0)
                s_rev = st.slider("Short-term Revenue Adjustment (%)", -30, 30, 0)
                s_mar = st.slider("Short-term EBITDA Margin Adjustment (%)", -30, 30, 0)

            with c2:
                m_em = st.slider("Medium-term Emissions Reduction (%)", 0, 100, 0)
                m_rev = st.slider("Medium-term Revenue Adjustment (%)", -30, 30, 0)
                m_mar = st.slider("Medium-term EBITDA Margin Adjustment (%)", -30, 30, 0)

            with c3:
                l_em = st.slider("Long-term Emissions Reduction (%)", 0, 100, 0)
                l_rev = st.slider("Long-term Revenue Adjustment (%)", -30, 30, 0)
                l_mar = st.slider("Long-term EBITDA Margin Adjustment (%)", -30, 30, 0)

            run_targets = st.button("▶ Run Target Scenario")

            if run_targets:

                # ------------------------------------------------
                # APPLY TARGETS (UNCHANGED CORE LOGIC)
                # ------------------------------------------------
                df_target = df_base.copy().reset_index(drop=True)

                def apply_targets(row):

                    if row["Year"] <= 2030:
                        em, rev, mar = s_em, s_rev, s_mar
                    elif row["Year"] <= 2040:
                        em, rev, mar = m_em, m_rev, m_mar
                    else:
                        em, rev, mar = l_em, l_rev, l_mar

                    revenue_t = row["Revenue"] * (1 + rev / 100)
                    margin_t = row["EBITDA_Margin"] * (1 + mar / 100)

                    pd_t = np.clip(
                        row["PD_Transition"] * (1 - em / 100),
                        0, 1
                    )

                    ecl_t = pd_t * row["LGD"] * exposure_at_default / 1e3

                    return pd.Series({
                        "Revenue_Target": revenue_t,
                        "EBITDA_Target": revenue_t * margin_t,
                        "PD_Target": pd_t,
                        "ECL_Target": ecl_t
                    })

                targets_applied = df_target.apply(apply_targets, axis=1)

                df_targets = pd.concat(
                    [
                        df_target[["Scenario", "Year"]],
                        targets_applied
                    ],
                    axis=1
                ).reset_index(drop=True)

                # --------------------------------------------
                # AUTHORITATIVE STORAGE (EXISTING)
                # --------------------------------------------
                st.session_state["df_target"] = df_targets
                st.session_state["targets_ran"] = True

                # ------------------------------------------------
                # 🔹 NEW — TARGET EFFECTIVENESS SUMMARY (SAFE ADD)
                # ------------------------------------------------
                df_target_effect = (
                    df_targets
                    .groupby("Scenario")
                    .agg({
                        "PD_Target": "max"
                    })
                    .rename(columns={"PD_Target": "PD_Target_Max"})
                    .reset_index()
                )

                df_base_effect = (
                    df_base
                    .groupby("Scenario")
                    .agg({
                        "PD_Transition": "max"
                    })
                    .rename(columns={"PD_Transition": "PD_Base_Max"})
                    .reset_index()
                )

                df_target_effect = df_target_effect.merge(
                    df_base_effect,
                    on="Scenario",
                    how="left"
                )

                df_target_effect["PD_Reduction_%"] = (
                    (df_target_effect["PD_Base_Max"]
                     - df_target_effect["PD_Target_Max"])
                    / df_target_effect["PD_Base_Max"]
                    * 100
                ).round(2)

                # Store ONLY for plots & AI (no logic dependency)
                st.session_state["df_target_effect"] = df_target_effect

                # --------------------------------------------
                # OUTPUTS
                # --------------------------------------------
                st.success("✅ Transition Targets applied successfully")

                st.subheader("📊 Target-Adjusted Outputs (Sample)")
                st.dataframe(df_targets.head(10), use_container_width=True)

                st.subheader("🎯 Target Effectiveness Summary")
                st.caption(
                    "Comparison of baseline vs target-adjusted peak credit risk "
                    "(ISSB S2 §18–20 / GFANZ-aligned)."
                )
                st.dataframe(df_target_effect, use_container_width=True)

            else:
                st.info("Set targets and click **Run Target Scenario**.")

# ============================================================
# PART 7 — 📘 BRSR CORE DIAGNOSTICS (ENHANCED, ADDITIVE ONLY)
# ============================================================

with brsr_tab:

    st.header("📘 BRSR Core Climate Diagnostics")
    st.caption(
        "SEBI BRSR Core–aligned operational climate diagnostics. "
        "This module is independent of transition, physical, and target engines."
    )

    # --------------------------------------------------------
    # ENABLE FLAG — NON-BLOCKING
    # --------------------------------------------------------
    if not st.session_state.get("enable_brsr", False):
        st.info("ℹ️ Enable **BRSR Diagnostics** from the sidebar to proceed.")
    else:

        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------
        st.subheader("🔢 Operational Climate Inputs")

        c1, c2, c3 = st.columns(3)

        with c1:
            total_energy_kwh = st.number_input(
                "Total Energy Consumption (kWh)",
                value=1_000_000_000,
                step=10_000_000
            )
            renewable_share_pct = st.slider(
                "Renewable Energy Share (%)", 0, 100, 15
            )

        with c2:
            total_water_m3 = st.number_input(
                "Total Water Withdrawal (m³)",
                value=50_000_000,
                step=1_000_000
            )
            water_stress_region = st.selectbox(
                "Primary Water Stress Region",
                ["Low", "Medium", "High"]
            )
            recycled_water_pct = st.slider(
                "Recycled Water Share (%)", 0, 100, 10
            )

        with c3:
            total_waste_mt = st.number_input(
                "Total Waste Generated (MT)",
                value=100_000,
                step=1_000
            )
            hazardous_waste_pct = st.slider(
                "Hazardous Waste Share (%)", 0, 100, 20
            )
            target_coverage_pct = st.slider(
                "Emissions Covered by Reduction Targets (%)", 0, 100, 50
            )

        # ----------------------------------------------------
        # EXECUTION BUTTON
        # ----------------------------------------------------
        run_brsr = st.button("▶ Run BRSR Diagnostics")

        if run_brsr:

            # ================================================
            # KPI COMPUTATION — PURE & STANDALONE
            # ================================================
            base_revenue = revenue_0

            # --- Scope-wise intensity (BRSR mandatory) ---
            scope1_intensity = scope1 / max(base_revenue, 1e6)
            scope2_intensity = scope2 / max(base_revenue, 1e6)
            scope3_intensity = scope3 / max(base_revenue, 1e6)

            brsr_summary = pd.DataFrame([{
                "GHG_Intensity_tCO2e_per_RsCr":
                    TOTAL_EMISSIONS / max(base_revenue, 1e6),

                "Scope1_Intensity_tCO2e_per_RsCr":
                    scope1_intensity,

                "Scope2_Intensity_tCO2e_per_RsCr":
                    scope2_intensity,

                "Scope3_Intensity_tCO2e_per_RsCr":
                    scope3_intensity,

                "Energy_Intensity_kWh_per_RsCr":
                    total_energy_kwh / max(base_revenue, 1e6),

                "Renewable_Energy_%":
                    renewable_share_pct,

                "Water_Intensity_m3_per_RsCr":
                    total_water_m3 / max(base_revenue, 1e6),

                "Recycled_Water_%":
                    recycled_water_pct,

                "Waste_Intensity_MT_per_RsCr":
                    total_waste_mt / max(base_revenue, 1e6),

                "Hazardous_Waste_%":
                    hazardous_waste_pct,

                "Target_Coverage_%":
                    target_coverage_pct,

                "Water_Stress_Level":
                    water_stress_region
            }]).round(3)

            # ------------------------------------------------
            # DIAGNOSTIC FLAGS (RULE-BASED — EXISTING)
            # ------------------------------------------------
            flags = []

            if brsr_summary.loc[0, "GHG_Intensity_tCO2e_per_RsCr"] > 4.0:
                flags.append("High emissions intensity")

            if renewable_share_pct < 20:
                flags.append("Low renewable energy share")

            if brsr_summary.loc[0, "Energy_Intensity_kWh_per_RsCr"] > 150:
                flags.append("High energy intensity")

            if water_stress_region == "High":
                flags.append("Operations in high water-stress regions")

            if target_coverage_pct < 50:
                flags.append("Low emissions target coverage")

            if hazardous_waste_pct > 30:
                flags.append("High hazardous waste share")

            flag_df = pd.DataFrame({"Observation": flags})

            # ------------------------------------------------
            # 🟢🔴 BRSR CORE COMPLIANCE STATUS (NEW)
            # ------------------------------------------------
            compliance_rows = [
                {
                    "Indicator": "Renewable Energy Share",
                    "Threshold": "≥ 20%",
                    "Value": renewable_share_pct,
                    "Status": "Green" if renewable_share_pct >= 20 else "Amber"
                },
                {
                    "Indicator": "Emissions Target Coverage",
                    "Threshold": "≥ 50%",
                    "Value": target_coverage_pct,
                    "Status": "Green" if target_coverage_pct >= 50 else "Red"
                },
                {
                    "Indicator": "Hazardous Waste Share",
                    "Threshold": "≤ 30%",
                    "Value": hazardous_waste_pct,
                    "Status": "Green" if hazardous_waste_pct <= 30 else "Red"
                },
                {
                    "Indicator": "Water Stress Exposure",
                    "Threshold": "Low / Medium",
                    "Value": water_stress_region,
                    "Status": "Red" if water_stress_region == "High" else "Green"
                }
            ]

            df_brsr_compliance = pd.DataFrame(compliance_rows)

            # ------------------------------------------------
            # 🏭 OPERATIONAL CLIMATE RISK EXPOSURE (NEW)
            # ------------------------------------------------
            df_operational_risk = pd.DataFrame([{
                "Energy Transition Risk":
                    "High" if renewable_share_pct < 20 else "Moderate",

                "Water Risk Exposure":
                    water_stress_region,

                "Waste Management Risk":
                    "High" if hazardous_waste_pct > 30 else "Low",

                "Target Coverage Risk":
                    "High" if target_coverage_pct < 50 else "Low"
            }])

            # ------------------------------------------------
            # AUTHORITATIVE STORAGE
            # ------------------------------------------------
            st.session_state["brsr_summary"] = brsr_summary
            st.session_state["brsr_flags"] = flag_df
            st.session_state["df_brsr_compliance"] = df_brsr_compliance
            st.session_state["df_brsr_operational_risk"] = df_operational_risk
            st.session_state["brsr_ran"] = True

            # ------------------------------------------------
            # OUTPUTS
            # ------------------------------------------------
            st.subheader("📊 BRSR Core — Environmental Performance Snapshot")
            st.dataframe(brsr_summary.T, use_container_width=True)

            st.subheader("🟢🔴 BRSR Core Compliance Status")
            st.dataframe(df_brsr_compliance, use_container_width=True)

            st.subheader("🏭 Operational Climate Risk Exposure")
            st.dataframe(df_operational_risk, use_container_width=True)

            st.subheader("🚩 Diagnostic Flags")
            if flag_df.empty:
                st.success("No material BRSR red flags identified.")
            else:
                st.dataframe(flag_df, use_container_width=True)

            st.success("✅ BRSR diagnostics completed successfully")

        else:
            st.info("Click **Run BRSR Diagnostics** to compute BRSR Core indicators.")
# ============================================================
# PART 8 — 🧩 INTEGRATED TRANSITION + PHYSICAL RISK
# (DERIVED ONLY | ISSB S2 / RBI / ICAAP ALIGNED)
# ============================================================

with integrated_tab:

    st.header("🧩 Integrated Climate & Credit Risk (Transition + Physical)")
    st.caption(
        "Integrated assessment derived strictly from executed Transition and/or "
        "Physical Risk modules. No recomputation, no implicit assumptions."
    )

    # --------------------------------------------------------
    # HARD GATING — EXECUTION STATUS
    # --------------------------------------------------------
    transition_ran = st.session_state.get("transition_ran", False)
    physical_ran   = st.session_state.get("physical_ran", False)

    if not (transition_ran or physical_ran):
        st.info(
            "ℹ️ Integrated results are unavailable.\n\n"
            "Run at least one module: Transition Risk or Physical Risk."
        )
        st.stop()

    # --------------------------------------------------------
    # DISCLOSURE — INCLUDED COMPONENTS
    # --------------------------------------------------------
    included = []
    if transition_ran:
        included.append("Transition Risk")
    if physical_ran:
        included.append("Physical Risk")

    st.success(
        "📌 Integrated results currently include:\n\n"
        + " • " + "\n • ".join(included)
    )

    if transition_ran and not physical_ran:
        st.warning(
            "⚠️ Physical Risk has not been executed.\n"
            "Results below reflect **transition risk only**."
        )

    if physical_ran and not transition_ran:
        st.warning(
            "⚠️ Transition Risk has not been executed.\n"
            "Results below reflect **physical risk only**."
        )

    # ========================================================
    # READ TRANSITION OUTPUTS (AS-IS)
    # ========================================================
    pd_trans = None
    ecl_trans = None
    dscr_trans = None
    stranded_assets = None
    capex_gap = None

    if transition_ran:
        df_t = st.session_state.get("df_transition")

        if isinstance(df_t, pd.DataFrame) and not df_t.empty:
            pd_trans = float(df_t["PD_Transition"].max())
            ecl_trans = float(df_t["ECL_Transition"].max())
            dscr_trans = float(df_t["DSCR"].min())
            stranded_assets = float(df_t["Stranded_Assets"].max())
            capex_gap = float(df_t["CAPEX_Gap"].max())

    # ========================================================
    # READ PHYSICAL OUTPUTS (AS-IS)
    # ========================================================
    pd_phys = None
    ecl_phys = None
    dscr_phys = None
    revenue_loss_phys = None

    if physical_ran:
        phys = st.session_state.get("phys_summary")

        if isinstance(phys, dict):
            pd_phys = float(phys.get("Physical Risk PD", 0.0))
            ecl_phys = float(phys.get("ΔECL (₹ Cr)", 0.0))
            dscr_phys = float(phys.get("Post-Risk DSCR", np.nan))
            revenue_loss_phys = float(phys.get("Total Revenue Loss (₹ Cr)", 0.0))

    # ========================================================
    # INTEGRATION EQUATIONS (RBI / ISSB SAFE)
    # ========================================================
    pd_integrated = None
    ecl_integrated = None
    dscr_integrated = None

    if transition_ran and physical_ran:
        # Joint default probability
        pd_integrated = 1 - (1 - pd_trans) * (1 - pd_phys)

        # Additive ECL
        ecl_integrated = ecl_trans + ecl_phys

        # Conservative DSCR (lower bound)
        dscr_integrated = min(dscr_trans, dscr_phys)

    elif transition_ran:
        pd_integrated = pd_trans
        ecl_integrated = ecl_trans
        dscr_integrated = dscr_trans

    elif physical_ran:
        pd_integrated = pd_phys
        ecl_integrated = ecl_phys
        dscr_integrated = dscr_phys

    # ========================================================
    # ICAAP / CET-1 STYLE CAPITAL STRESS
    # ========================================================
    EAD = float(exposure_at_default)

    climate_ecl_ratio = (
        ecl_integrated / EAD if EAD > 0 else np.nan
    )

    capital_signal = (
        "High" if climate_ecl_ratio >= 0.03 else
        "Moderate" if climate_ecl_ratio >= 0.015 else
        "Low"
    )

    # ========================================================
    # INTEGRATED RISK SCORE (0–100)
    # ========================================================
    # Components: PD stress, ECL stress, DSCR stress
    pd_score = min(100, (pd_integrated / 0.20) * 100)          # 20% PD = max stress
    ecl_score = min(100, (climate_ecl_ratio / 0.05) * 100)    # 5% of EAD
    dscr_score = (
        100 if dscr_integrated < 1.0 else
        60 if dscr_integrated < 1.2 else
        30 if dscr_integrated < 1.5 else
        10
    )

    integrated_risk_score = round(
        0.4 * pd_score +
        0.4 * ecl_score +
        0.2 * dscr_score,
        1
    )

    # ========================================================
    # INTEGRATED SUMMARY TABLE (ISSB-TAGGED)
    # ========================================================
    st.subheader("📊 Integrated Climate Risk Summary (ISSB S2 Aligned)")

    summary_rows = [
        {
            "Metric": "Integrated Probability of Default",
            "Value": round(pd_integrated, 4),
            "ISSB S2 Reference": "§15(a) – Financial effects of climate risk"
        },
        {
            "Metric": "Integrated Expected Credit Loss (₹ Cr)",
            "Value": round(ecl_integrated, 2),
            "ISSB S2 Reference": "§15(b) – Credit & impairment impacts"
        },
        {
            "Metric": "Post-Stress DSCR",
            "Value": round(dscr_integrated, 2),
            "ISSB S2 Reference": "§15(c) – Cash-flow resilience"
        },
        {
            "Metric": "Climate ECL / EAD",
            "Value": f"{climate_ecl_ratio:.2%}",
            "ISSB S2 Reference": "§16 – Capital allocation & resilience"
        },
        {
            "Metric": "Capital Stress Signal (ICAAP)",
            "Value": capital_signal,
            "ISSB S2 Reference": "§16 – Financial resilience"
        },
        {
            "Metric": "Integrated Climate Risk Score (0–100)",
            "Value": integrated_risk_score,
            "ISSB S2 Reference": "§14–16 (Composite indicator)"
        }
    ]

    df_integrated_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_integrated_summary, use_container_width=True)

    # ========================================================
    # TRANSITION-SPECIFIC CONTEXT (IF AVAILABLE)
    # ========================================================
    if transition_ran:
        st.subheader("⚡ Transition Risk Context")

        st.dataframe(
            pd.DataFrame([{
                "Max PD (Transition)": round(pd_trans, 4),
                "Max ECL (₹ Cr)": round(ecl_trans, 2),
                "Min DSCR": round(dscr_trans, 2),
                "Stranded Assets (₹ Cr)": round(stranded_assets, 0),
                "CAPEX Gap (₹ Cr)": round(capex_gap, 0)
            }]),
            use_container_width=True
        )

    # ========================================================
    # PHYSICAL-SPECIFIC CONTEXT (IF AVAILABLE)
    # ========================================================
    if physical_ran:
        st.subheader("🌍 Physical Risk Context")

        st.dataframe(
            pd.DataFrame([{
                "Physical PD": round(pd_phys, 4),
                "Physical ΔECL (₹ Cr)": round(ecl_phys, 2),
                "Revenue Loss (₹ Cr)": round(revenue_loss_phys, 2),
                "Post-Risk DSCR": round(dscr_phys, 2)
            }]),
            use_container_width=True
        )

    # ========================================================
    # STORE FOR AI / EXPORT
    # ========================================================
    st.session_state["df_integrated_summary"] = df_integrated_summary
    st.session_state["integrated_ran"] = True

    st.success("✅ Integrated Transition + Physical Risk assessment completed.")

# ============================================================
# PART 9 — 🤖 AI RISK ADVISOR & REGULATORY NARRATIVE ENGINE
# ============================================================

with ai_tab:

    st.header("🤖 AI Climate Risk Advisor & Reporting Assistant")
    st.caption(
        "Plain-language, regulator-ready interpretation of results. "
        "Explains impacts, identifies weaknesses, and provides actions. "
        "Outputs can be directly used in ISSB / IFRS S2, RBI, ICAAP, and BRSR reports."
    )

    # --------------------------------------------------------
    # HARD GUARD — AT LEAST ONE MODULE MUST HAVE RUN
    # --------------------------------------------------------
    transition_ran = st.session_state.get("transition_ran", False)
    physical_ran   = st.session_state.get("physical_ran", False)
    targets_ran    = st.session_state.get("targets_ran", False)
    brsr_ran       = st.session_state.get("brsr_ran", False)
    integrated_ran = st.session_state.get("integrated_ran", False)

    if not (transition_ran or physical_ran or targets_ran or brsr_ran):
        st.info(
            "ℹ️ Run at least one module (Transition, Physical, Targets, or BRSR) "
            "before generating the AI analysis."
        )
        st.stop()

    # --------------------------------------------------------
    # USER INTENT — HOW WILL THIS BE USED
    # --------------------------------------------------------
    st.subheader("📌 How do you plan to use this output?")

    usage_context = st.selectbox(
        "Primary use",
        [
            "ISSB / IFRS S2 Climate Disclosure",
            "RBI / Bank Credit Risk & Stress Testing",
            "ICAAP / Capital Adequacy Assessment",
            "Board / Senior Management Briefing",
            "Investor / Lender Disclosure",
            "Internal Risk Management & Action Plan"
        ]
    )

    st.subheader("📄 Report Sections to Generate")

    report_sections = st.multiselect(
        "Select sections you want ready-to-use text for",
        [
            "Executive Summary",
            "Business Model & Strategy (ISSB S2 §14)",
            "Risk Identification & Assessment (ISSB S2 §15)",
            "Financial Impacts & Resilience (ISSB S2 §15–16)",
            "Physical Risk Exposure (ISSB S2 §22–25)",
            "Capital Adequacy & ICAAP Implications",
            "Metrics, Targets & Performance",
            "Key Risk Findings & Red Flags",
            "Management Actions & Recommendations",
            "Limitations & Assumptions (Disclosure Note)"
        ],
        default=[
            "Executive Summary",
            "Risk Identification & Assessment (ISSB S2 §15)",
            "Financial Impacts & Resilience (ISSB S2 §15–16)",
            "Key Risk Findings & Red Flags",
            "Management Actions & Recommendations"
        ]
    )

    tone = st.selectbox(
        "Language style",
        [
            "Simple & explanatory (non-technical)",
            "Balanced (management & regulator friendly)",
            "Strictly regulatory & supervisory"
        ],
        index=1
    )

    # --------------------------------------------------------
    # BUILD DATA PAYLOAD (STRICT, TRACEABLE)
    # --------------------------------------------------------
    def build_advisor_payload():

        payload = {
            "company_context": {
                "Company": company_name,
                "Sector": sector,
                "Reporting_Year": REPORTING_YEAR,
                "Usage_Context": usage_context,
                "Requested_Sections": report_sections,
                "Tone": tone
            }
        }

        if transition_ran:
            df_t = st.session_state.get("df_transition")
            if isinstance(df_t, pd.DataFrame) and not df_t.empty:
                payload["transition_risk"] = (
                    df_t.groupby("Scenario")
                    .agg({
                        "PD_Transition": "max",
                        "ECL_Transition": "max",
                        "DSCR": "min",
                        "EBITDA_Margin": "min",
                        "Carbon_Burden": "max",
                        "Stranded_Assets": "max",
                        "CAPEX_Gap": "max"
                    })
                    .round(3)
                    .to_dict()
                )

        if physical_ran:
            payload["physical_risk"] = st.session_state.get("phys_summary")

        if targets_ran:
            df_tg = st.session_state.get("df_target_effect")
            if isinstance(df_tg, pd.DataFrame):
                payload["transition_targets"] = df_tg.to_dict("records")

        if integrated_ran:
            payload["integrated_risk"] = (
                st.session_state.get("df_integrated_summary")
                .to_dict("records")
            )

        if brsr_ran:
            payload["brsr"] = {
                "kpis": st.session_state.get("brsr_summary").to_dict(),
                "flags": (
                    st.session_state.get("brsr_flags").to_dict("records")
                    if not st.session_state.get("brsr_flags").empty
                    else []
                )
            }

        return payload

    # --------------------------------------------------------
    # AI SYSTEM INSTRUCTION (CRITICAL)
    # --------------------------------------------------------
    SYSTEM_PROMPT = """
You are a senior climate risk, credit risk, and regulatory reporting expert.
Your role is to help a company understand its climate-related risks without
needing an external consultant.

Rules:
1. Use only the data provided. Do NOT assume missing results.
2. Clearly explain what each result means in simple language.
3. Identify material risks, weaknesses, and supervisory concerns.
4. Clearly state which disclosures the results belong to:
   - ISSB / IFRS S2 sections
   - RBI / ICAAP credit risk implications
   - BRSR where relevant
5. Provide practical, realistic management actions.
6. Use a professional, regulator-friendly tone.
7. Structure the response with clear headings matching report sections.
8. If a module was not run, explicitly say so and limit conclusions accordingly.
"""

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------
    run_ai = st.button("🧠 Generate Risk Analysis & Report Text")

    if run_ai:
        with st.spinner("AI is analysing results like a senior risk professional..."):
            try:
                payload = build_advisor_payload()

                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(payload, indent=2)
                        }
                    ],
                    temperature=0.25
                )

                st.session_state["ai_outputs"]["advisor"] = (
                    response.choices[0].message.content
                )

            except Exception as e:
                st.error(f"❌ AI analysis failed: {e}")

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
    if "advisor" in st.session_state.get("ai_outputs", {}):
        st.subheader("📘 AI Risk Analysis & Reporting Text")
        st.markdown(st.session_state["ai_outputs"]["advisor"])
        st.success(
            "✅ You can directly use this text in disclosures, board notes, "
            "ICAAP documents, or regulatory submissions."
        )
    else:
        st.info("Click **Generate Risk Analysis & Report Text** to proceed.")
