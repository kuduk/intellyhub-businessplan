import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from collections import deque

# ==============================================================================
# --- SCRIPT CONFIGURATION ---
# This script simulates a financial projection for a business plan, focusing on
# the break-even point for a PLG (Product-Led Growth) and SLG (Sales-Led Growth)
# model. It includes cost structures, revenue streams, customer acquisition,
# and runway management.
#
# Author: Pasetto Francesco
# Date: 2025-08-08
# ==============================================================================

# --- 1a. Initial capital (€) ---
INITIAL_CAPITAL = 7_150_000

# --- 1b. Base annual costs (€) ---
# These are planned "net" costs. The prudential uplift is applied separately.

# Management & Leadership
MGMT_COSTS = {1: 420_000, 2: 453_600, 3: 547_200, 4: 547_200, 5: 547_200}

# R&D (Product & Engineering)
RND_COSTS = {1: 351_200, 2: 359_200, 3: 453_000, 4: 453_000, 5: 453_000}

# PLG Team (Marketing & Community)
PLG_TEAM_COSTS = {1: 46_800, 2: 85_800, 3: 117_000, 4: 117_000, 5: 117_000}

# SLG Team (Sales, Success & Partners)
SLG_TEAM_COSTS = {1: 45_000, 2: 127_440, 3: 232_440, 4: 314_880, 5: 404_880}

# Partner Enablement
PARTNER_TEAM_COSTS = {1: 0, 2: 46_800, 3: 145_600, 4: 145_600, 5: 145_600}

# Infrastructure & Platform
INFRA_COSTS = {1: 100_000, 2: 200_000, 3: 250_000, 4: 250_000, 5: 250_000}

# General & Administrative (G&A)
GA_COSTS = {1: 300_000, 2: 300_000, 3: 450_000, 4: 450_000, 5: 450_000}

# --- 1c. Prudential uplift by category (instead of a single buffer) ---
UPLIFT = {
    'MGMT': 1.10,
    'RND': 1.10,
    'PLG': 1.20,
    'SLG': 1.15,
    'PARTNER': 1.30,
    'INFRA': 1.35,
    'GA': 1.25
}

# --- 2. Monthly prices (€) ---
PRICES = {
    'standard': 49,
    'business': 299,
    'scale': 999,
    'enterprise_mrr': 40_000 / 12,  # ACV €40k
    'add_on_pod': 25
}

# --- 3. Churn rates ---
CHURN_PLG_MONTHLY = 0.05
CHURN_SLG_ANNUAL = 0.10

# --- 4. Add-on adoption (monthly active share) ---
ADD_ON_ADOPTION = {
    'standard': {'adoption_rate': 0.15, 'avg_pods': 2},
    'business': {'adoption_rate': 0.25, 'avg_pods': 4},
    'scale':    {'adoption_rate': 0.30, 'avg_pods': 8}
}

# --- 5. Store (one-off; NOT in MRR) ---
STORE_REVENUE = {
    'avg_sale_price': 50,
    'purchase_rate_monthly': 0.05,
    'avg_purchases_per_customer': 1.2
}

# --- 6. Acquisitions ---
ACQUISITION_PLG = {
    1: (3, 1, 0),
    2: (7, 2, 1),
    3: (12, 4, 2),
    4: (12, 4, 2),
    5: (12, 4, 2)
}
ACQUISITION_SLG = {1: 2, 2: 8, 3: 18, 4: 24, 5: 36}

# --- 7. Spend Gates ---
ENABLE_SPEND_GATES = True
RUNWAY_MA_WINDOW = 3          # moving average window (months)
RUNWAY_GUARDRAIL = 12         # if runway < this, cut spend for next month
SPEND_CUT_PERCENTAGE = 0.10   # next-month cost multiplier reduction
RUNWAY_SOFT_FREEZE = 9        # if runway < this, dampen PLG acquisitions
ACQ_DAMPING_WHEN_SOFT_FREEZE = 0.7

# ==============================================================================
# --- COST MODEL HELPERS ---
# ==============================================================================

def annual_cost_with_uplift(year: int) -> float:
    return (
        MGMT_COSTS[year]    * UPLIFT['MGMT']  +
        RND_COSTS[year]     * UPLIFT['RND']   +
        PLG_TEAM_COSTS[year]* UPLIFT['PLG']   +
        SLG_TEAM_COSTS[year]* UPLIFT['SLG']   +
        PARTNER_TEAM_COSTS[year]*UPLIFT['PARTNER'] +
        INFRA_COSTS[year]   * UPLIFT['INFRA'] +
        GA_COSTS[year]      * UPLIFT['GA']
    )

def monthly_cost_with_uplift(year: int, spend_cut_multiplier: float = 1.0) -> float:
    base = annual_cost_with_uplift(year) / 12.0
    return base * spend_cut_multiplier

# ==============================================================================
# --- SIMULATION LOGIC ---
# ==============================================================================

def run_simulation(simulation_years=5, initial_capital=0):
    churn_slg_monthly = 1 - (1 - CHURN_SLG_ANNUAL) ** (1/12)

    simulation_months = simulation_years * 12
    simulation_data = []

    customers = {'standard': 0.0, 'business': 0.0, 'scale': 0.0, 'enterprise': 0.0}
    cash_balance = float(initial_capital)

    spend_cut_multiplier = 1.0
    acq_plg_multiplier = 1.0

    recent_burns = deque(maxlen=RUNWAY_MA_WINDOW)

    print("Running simulation...")

    for month in range(1, simulation_months + 1):
        year = ((month - 1) // 12) + 1

        # Churn
        customers['standard']   *= (1 - CHURN_PLG_MONTHLY)
        customers['business']   *= (1 - CHURN_PLG_MONTHLY)
        customers['scale']      *= (1 - CHURN_PLG_MONTHLY)
        customers['enterprise'] *= (1 - churn_slg_monthly)

        # PLG acquisitions
        new_std, new_biz, new_scl = ACQUISITION_PLG[year]
        customers['standard'] += new_std * acq_plg_multiplier
        customers['business'] += new_biz * acq_plg_multiplier
        customers['scale']    += new_scl * acq_plg_multiplier

        # SLG quarterly
        if month % 3 == 0:
            customers['enterprise'] += ACQUISITION_SLG[year] / 4.0

        # Plan revenues
        mrr_base_plg = (customers['standard'] * PRICES['standard'] +
                        customers['business'] * PRICES['business'] +
                        customers['scale']    * PRICES['scale'])
        mrr_base_slg = customers['enterprise'] * PRICES['enterprise_mrr']

        # Add-ons
        mrr_addons = 0.0
        for plan in ['standard', 'business', 'scale']:
            adopters = customers[plan] * ADD_ON_ADOPTION[plan]['adoption_rate']
            pods_sold = adopters * ADD_ON_ADOPTION[plan]['avg_pods']
            mrr_addons += pods_sold * PRICES['add_on_pod']

        # Recurring totals and store
        mrr_recurring = mrr_base_plg + mrr_base_slg + mrr_addons
        total_customers = customers['standard'] + customers['business'] + customers['scale'] + customers['enterprise']
        purchasing_customers = total_customers * STORE_REVENUE['purchase_rate_monthly']
        items_sold = purchasing_customers * STORE_REVENUE['avg_purchases_per_customer']
        revenue_store = items_sold * STORE_REVENUE['avg_sale_price']

        monthly_cost = monthly_cost_with_uplift(year, spend_cut_multiplier)

        revenue_total = mrr_recurring + revenue_store
        net_monthly = revenue_total - monthly_cost

        cash_balance += net_monthly

        burn_this_month = -net_monthly if net_monthly < 0 else 0.0
        recent_burns.append(burn_this_month)
        positive_burns = [b for b in recent_burns if b > 0]
        avg_burn = np.mean(positive_burns) if positive_burns else 0.0
        runway_months = (cash_balance / avg_burn) if avg_burn > 0 else float('inf')

        # --- Save ---
        simulation_data.append({
            'Month': month,
            'Year': year,
            'Monthly Cost (€)': monthly_cost,
            'Self-Service Customers': int(customers['standard'] + customers['business'] + customers['scale']),
            'Enterprise Customers': int(customers['enterprise']),
            'MRR PLG Plans (€)': mrr_base_plg,
            'MRR SLG (€)': mrr_base_slg,
            'MRR from Add-ons (€)': mrr_addons,
            'Total PLG MRR (€)': mrr_base_plg + mrr_addons,
            'Recurring MRR (€)': mrr_recurring,
            'Store Revenue (€)': revenue_store,
            'Total Revenue (€)': revenue_total,
            'Monthly Profit/Loss (€)': net_monthly,
            'Cash Balance (€)': cash_balance,
            'Runway (months)': runway_months,
            'Cost Multiplier': spend_cut_multiplier,
            'PLG Acquisition Multiplier': acq_plg_multiplier
        })

        # Policy (next month)
        if ENABLE_SPEND_GATES and np.isfinite(runway_months):
            spend_cut_multiplier = (1.0 - SPEND_CUT_PERCENTAGE) if runway_months < RUNWAY_GUARDRAIL else 1.0
            acq_plg_multiplier = (ACQ_DAMPING_WHEN_SOFT_FREEZE if runway_months < RUNWAY_SOFT_FREEZE else 1.0)
        else:
            spend_cut_multiplier = 1.0
            acq_plg_multiplier = 1.0

    print("Simulation completed.")
    return pd.DataFrame(simulation_data)

# ==============================================================================
# --- OUTPUT & METRICS (TEXT) ---
# ==============================================================================

def _fmt_euro(x):
    # English thousands separator (comma)
    return f"{int(round(x, 0)):,}"


def display_results(df: pd.DataFrame):
    """Prints annual summary and key metrics: operating/cash BE, capital burned, peak burn, min cash, min runway, remaining reserve."""

    def _fmt_month(m: int) -> str:
        y = ((int(m) - 1) // 12) + 1
        return f"Month {int(m)} (Year {y})"

    # Year-end summary
    summary_rows = []
    for year in sorted(df['Year'].unique()):
        year_df = df[df['Year'] == year]
        summary_rows.append(year_df.iloc[-1])
    summary_df = pd.DataFrame(summary_rows).copy()
    summary_df.index = [f'End of Year {i}' for i in summary_df['Year']]

    cols = [
        'Monthly Cost (€)', 'Recurring MRR (€)', 'Store Revenue (€)',
        'Total Revenue (€)', 'Monthly Profit/Loss (€)', 'Cash Balance (€)', 'Runway (months)'
    ]
    print("\n--- FINANCIAL PROJECTION SUMMARY (YEAR-END) ---")
    disp = summary_df.copy()
    disp['Runway (months)'] = disp['Runway (months)'].apply(lambda x: f"{x:.1f}" if np.isfinite(x) else "Profitable")
    for c in cols:
        if c != 'Runway (months)':
            disp[c] = disp[c].apply(_fmt_euro)
    print(disp[cols].to_string())

    # Key metrics
    print("\n--- KEY FINANCIAL METRICS ---")

    # Operating BE
    be_oper_series = df[df['Recurring MRR (€)'] >= df['Monthly Cost (€)']]
    if not be_oper_series.empty:
        be_oper_row = be_oper_series.iloc[0]
        be_oper_month = int(be_oper_row['Month'])
        print(f"OPERATING break-even: {_fmt_month(be_oper_month)} → Recurring MRR €{_fmt_euro(be_oper_row['Recurring MRR (€)'])} ≥ Costs €{_fmt_euro(be_oper_row['Monthly Cost (€)'])}.")
    else:
        be_oper_month = None
        print("OPERATING break-even not reached in the simulated period.")

    # Cash BE
    be_cash_series = df[df['Monthly Profit/Loss (€)'] >= 0]
    if not be_cash_series.empty:
        be_cash_row = be_cash_series.iloc[0]
        be_cash_month = int(be_cash_row['Month'])
        burn_to_cash_be_df = df.loc[df['Month'] <= be_cash_month]
        total_burn_to_be = abs(burn_to_cash_be_df[burn_to_cash_be_df['Monthly Profit/Loss (€)'] < 0]['Monthly Profit/Loss (€)'].sum())
        print(f"CASH break-even: {_fmt_month(be_cash_month)} → Total Revenue €{_fmt_euro(be_cash_row['Total Revenue (€)'])} ≥ Costs €{_fmt_euro(be_cash_row['Monthly Cost (€)'])}.")
        print(f"Capital burned until cash BE: €{_fmt_euro(total_burn_to_be)}.")
    else:
        be_cash_month = None
        total_burn_to_be = None
        total_burn_all = abs(df[df['Monthly Profit/Loss (€)'] < 0]['Monthly Profit/Loss (€)'].sum())
        print(f"CASH break-even not reached. Capital burned in the period: €{_fmt_euro(total_burn_all)}.")

    # Additional metrics
    print("\n--- ADDITIONAL METRICS (for business plan) ---")
    min_net = df['Monthly Profit/Loss (€)'].min()
    peak_burn = -min_net if min_net < 0 else 0.0
    print(f"Monthly peak burn: €{_fmt_euro(peak_burn)}")

    min_cash = float(df['Cash Balance (€)'].min())
    print(f"Minimum cash in period: €{_fmt_euro(min_cash)}")

    finite_runways = df['Runway (months)'][np.isfinite(df['Runway (months)']) & (df['Runway (months)'] > 0)]
    if not finite_runways.empty:
        min_runway = float(finite_runways.min())
        print(f"Minimum runway (MA {RUNWAY_MA_WINDOW}m): {min_runway:.1f} months")
    else:
        print(f"Minimum runway (MA {RUNWAY_MA_WINDOW}m): ∞")

    # Remaining reserve
    try:
        round_amount = float(INITIAL_CAPITAL)
    except NameError:
        round_amount = float(df.iloc[0]['Cash Balance (€)'])

    if 'total_burn_to_be' in locals() and total_burn_to_be is not None:
        reserve_eur = round_amount - float(total_burn_to_be)
        ctx = "at cash BE"
    else:
        total_burn = abs(df[df['Monthly Profit/Loss (€)'] < 0]['Monthly Profit/Loss (€)'].sum())
        reserve_eur = round_amount - float(total_burn)
        ctx = "at period end"

    reserve_pct = (reserve_eur / round_amount) if round_amount > 0 else 0.0
    print(f"Estimated remaining reserve {ctx}: €{_fmt_euro(reserve_eur)} ({reserve_pct:.1%})")


# ==============================================================================
# --- PLOT ---
# ==============================================================================

def plot_results(df: pd.DataFrame):
    """Chart with two bands: Total PLG vs SLG, plus costs and runway."""
    print("\nGenerating chart...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Costs
    ax1.plot(df['Month'], df['Monthly Cost (€)'], label='Monthly Operating Cost', color='red', linestyle='--')

    # Total PLG and SLG separately (two-level stack)
    revenue_components = {
        'PLG – Total (Plans + Add-ons)': df['Total PLG MRR (€)'],
        'SLG – Enterprise': df['MRR SLG (€)']
    }
    colors = ['#2ca02c', '#1f77b4']  # PLG green, SLG blue
    ax1.stackplot(df['Month'], revenue_components.values(), labels=revenue_components.keys(), colors=colors, alpha=0.8)

    # Axes
    ax1.set_xlabel('Months from Start', fontsize=12)
    ax1.set_ylabel('Euro (€)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    # Runway (right axis)
    ax2 = ax1.twinx()
    runway_plot = df['Runway (months)'].replace(float('inf'), np.nan)
    ax2.plot(df['Month'], runway_plot, label='Runway (months)', color='purple', linestyle=':', marker='.')
    ax2.set_ylabel('Runway (Months)', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    if not runway_plot.dropna().empty:
        ax2.set_ylim(bottom=0, top=max(runway_plot.dropna()) * 1.1)

    # Operating break-even
    be_oper_series = df[df['Recurring MRR (€)'] >= df['Monthly Cost (€)']]
    if not be_oper_series.empty:
        r = be_oper_series.iloc[0]
        be_m = int(r['Month'])
        ax1.axvline(x=be_m, color='blue', linestyle=':', linewidth=2, label=f'Operating BE (Month {be_m})')
        ax1.annotate(
            f'Operating BE\nMonth {be_m}\n€{_fmt_euro(r["Recurring MRR (€)"])}',
            xy=(be_m, r['Recurring MRR (€)']),
            xytext=(be_m + 3, r['Recurring MRR (€)'] / 2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8)
        )

    # Cash break-even
    be_cash_series = df[df['Monthly Profit/Loss (€)'] >= 0]
    if not be_cash_series.empty:
        r = be_cash_series.iloc[0]
        be_m = int(r['Month'])
        ax1.axvline(x=be_m, color='black', linestyle='--', linewidth=1.5, label=f'Cash BE (Month {be_m})')

    fig.suptitle('PLG Total vs SLG – MRR, Break-Even and Runway', fontsize=16, fontweight='bold')
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)

    plt.xticks(np.arange(min(df['Month']), max(df['Month']) + 1, 6))
    ax1.set_xlim(left=1)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    plt.savefig('financial_projection.png')
    print('Chart saved to financial_projection.png')
    plt.show()

# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    results_df = run_simulation(simulation_years=5, initial_capital=INITIAL_CAPITAL)
    display_results(results_df)
    plot_results(results_df)
