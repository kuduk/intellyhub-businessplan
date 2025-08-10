import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# --- SCRIPT CONFIGURATION ---
# Modify these values to test different scenarios.
# ==============================================================================

# --- 1a. Initial Capital (€) ---
INITIAL_CAPITAL = 7150000

# --- 1b. Annual Operational Costs (€) ---
# Annual costs by category, based on the new hiring roadmap.

# Management & Leadership Costs
MGMT_COSTS = {
    1: 420000, 2: 453600, 3: 547200, 4: 547200, 5: 547200
}
# R&D (Product & Engineering) Team Costs
RND_COSTS = {
    1: 266500, 2: 312000, 3: 351000, 4: 351000, 5: 351000
}
# Partner Enablement Team Costs
PARTNER_TEAM_COSTS = {
    1: 0, 2: 46800, 3: 145600, 4: 145600, 5: 145600
}
# Infrastructure & Platform Costs
INFRA_COSTS = {
    1: 100000, 2: 200000, 3: 250000, 4: 250000, 5: 250000
}
# General & Administrative (G&A) Costs
GA_COSTS = {
    1: 300000, 2: 300000, 3: 450000, 4: 450000, 5: 450000
}

# --- 1c. Dynamic Cost Buffer ---
# A buffer percentage applied to all operational costs for unforeseen expenses.
COST_BUFFER_PERCENTAGE = 0.20 # 20% buffer

# ==============================================================================
# --- SCENARIO SELECTION ---
# Choose one of the following scenarios by uncommenting the desired block.
# ==============================================================================

# --- SCENARIO 1: PRUDENTIAL GROWTH (Conservative) ---
# PLG Team (Marketing & Community) Costs
PLG_TEAM_COSTS = {
    1: 46800, 2: 85800, 3: 117000, 4: 117000, 5: 117000
}
# SLG Team (Sales, Success & Partners) Costs
SLG_TEAM_COSTS = {
    1: 45000, 2: 127440, 3: 232440, 4: 314880, 5: 404880
}
# Customer Acquisition Rates (PLG)
ACQUISITION_PLG = {
    1: (3, 1, 0), 
    2: (7, 2, 1), 
    3: (12, 4, 2), 
    4: (12, 4, 2), 
    5: (12, 4, 2)
}
# Customer Acquisition Rates (SLG)
ACQUISITION_SLG = {
    1: 2, 2: 8, 3: 18, 4: 24, 5: 36
}

# ==============================================================================

# Calculate total monthly operational costs including the buffer
COSTS = {}
for year in range(1, 6):
    # Sum of all planned costs for the year
    base_annual_cost = (MGMT_COSTS[year] + RND_COSTS[year] + PLG_TEAM_COSTS[year] +
                        SLG_TEAM_COSTS[year] + PARTNER_TEAM_COSTS[year] +
                        INFRA_COSTS[year] + GA_COSTS[year])
    
    # Apply the dynamic cost buffer
    final_annual_cost = base_annual_cost * (1 + COST_BUFFER_PERCENTAGE)
        
    COSTS[year] = final_annual_cost / 12


# --- 2. Pricing Model (€) ---
# Monthly prices for each plan and for add-on pods.
PRICES = {
    'standard': 49,
    'business': 299,
    'scale': 999,
    'enterprise_mrr': 40000 / 12,  # Based on a 40k € ACV
    'add_on_pod': 25
}

# --- 4. Churn Rates (Conservative) ---
# Monthly churn rates.
CHURN_PLG_MONTHLY = 0.05  # 5% monthly churn for self-service plans
CHURN_SLG_ANNUAL = 0.10   # 10% annual churn for enterprise customers

# --- 5. Add-on Pod Adoption Assumptions ---
# Assumptions on how many customers purchase extra pods.
ADD_ON_ADOPTION = {
    'standard': {'adoption_rate': 0.15, 'avg_pods': 2},
    'business': {'adoption_rate': 0.25, 'avg_pods': 4},
    'scale':    {'adoption_rate': 0.30, 'avg_pods': 8}
}

# --- 6. Store Revenue Assumptions ---
# Assumptions for one-time purchases from the automation store.
STORE_REVENUE = {
    'avg_sale_price': 50,           # Prezzo medio di un'automazione venduta
    'purchase_rate_monthly': 0.05,  # 5% dei clienti attivi totali effettua un acquisto ogni mese
    'avg_purchases_per_customer': 1.2 # Numero medio di acquisti per cliente acquirente
}


# ==============================================================================
# --- SIMULATION LOGIC ---
# Do not modify this part unless you want to change the calculation model.
# ==============================================================================

def run_simulation(simulation_years=5, initial_capital=0):
    """
    Esegue la simulazione finanziaria mese per mese.
    """
    
    # Convert annual enterprise churn to an equivalent monthly rate
    churn_slg_monthly = 1 - (1 - CHURN_SLG_ANNUAL)**(1/12)
    
    simulation_months = simulation_years * 12
    simulation_data = []
    
    # Initial state
    customers = {'standard': 0.0, 'business': 0.0, 'scale': 0.0, 'enterprise': 0.0}
    cash_balance = initial_capital
    
    print("Esecuzione della simulazione...")

    for month in range(1, simulation_months + 1):
        year = ((month - 1) // 12) + 1
        
        # --- Calcolo Churn (a inizio mese) ---
        customers['standard'] *= (1 - CHURN_PLG_MONTHLY)
        customers['business'] *= (1 - CHURN_PLG_MONTHLY)
        customers['scale']    *= (1 - CHURN_PLG_MONTHLY)
        customers['enterprise'] *= (1 - churn_slg_monthly)

        # --- Calcolo Acquisizioni (durante il mese) ---
        # PLG
        new_std, new_biz, new_scl = ACQUISITION_PLG[year]
        customers['standard'] += new_std
        customers['business'] += new_biz
        customers['scale']    += new_scl
        
        # SLG (i contratti annuali vengono aggiunti trimestralmente per smussare la crescita)
        if month % 3 == 0:
            new_ent_deals_quarterly = ACQUISITION_SLG[year] / 4.0
            customers['enterprise'] += new_ent_deals_quarterly

        # --- Calcolo MRR ---
        # 1. MRR dai piani di abbonamento
        mrr_base_plg = (customers['standard'] * PRICES['standard'] +
                        customers['business'] * PRICES['business'] +
                        customers['scale']    * PRICES['scale'])
        mrr_base_slg = customers['enterprise'] * PRICES['enterprise_mrr']
        
        # 2. MRR dagli add-on
        mrr_addons = 0
        for plan in ['standard', 'business', 'scale']:
            adopters = customers[plan] * ADD_ON_ADOPTION[plan]['adoption_rate']
            pods_sold = adopters * ADD_ON_ADOPTION[plan]['avg_pods']
            mrr_addons += pods_sold * PRICES['add_on_pod']

        # 3. Ricavi dallo Store (una tantum, trattati come MRR per semplicità di modello)
        total_customers = customers['standard'] + customers['business'] + customers['scale'] + customers['enterprise']
        purchasing_customers = total_customers * STORE_REVENUE['purchase_rate_monthly']
        items_sold = purchasing_customers * STORE_REVENUE['avg_purchases_per_customer']
        mrr_store = items_sold * STORE_REVENUE['avg_sale_price']
            
        total_mrr = mrr_base_plg + mrr_base_slg + mrr_addons + mrr_store
        
        # --- Calcolo Netto ---
        monthly_cost = COSTS[year]
        net_monthly = total_mrr - monthly_cost
        
        # --- Calcolo Runway ---
        cash_balance += net_monthly
        burn_rate = -net_monthly if net_monthly < 0 else 0
        runway_months = cash_balance / burn_rate if burn_rate > 0 else float('inf')


        # Salva i risultati del mese
        simulation_data.append({
            'Mese': month,
            'Anno': year,
            'Costo Mensile (€)': monthly_cost,
            'Clienti Self-Service': int(customers['standard'] + customers['business'] + customers['scale']),
            'Clienti Enterprise': int(customers['enterprise']),
            'MRR da Piani (€)': int(mrr_base_plg + mrr_base_slg),
            'MRR da Add-on (€)': int(mrr_addons),
            'MRR da Store (€)': int(mrr_store),
            'MRR Totale (€)': int(total_mrr),
            'Utile/Perdita Mensile (€)': int(net_monthly),
            'Saldo di Cassa (€)': int(cash_balance),
            'Runway (mesi)': runway_months
        })

    print("Simulazione completata.")
    return pd.DataFrame(simulation_data)


def display_results(df):
    """
    Formatta e stampa i risultati della simulazione.
    """
    
    # --- Tabella di Riepilogo ---
    summary_data = []
    for year in df['Anno'].unique():
        year_end_data = df[df['Anno'] == year].iloc[-1]
        summary_data.append(year_end_data)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.index = [f'Fine Anno {i}' for i in summary_df['Anno']]
    
    print("\n--- RIEPILOGO PROIEZIONE FINANZIARIA (FINE ANNO) ---")
    
    # Formattazione colonne per la visualizzazione
    summary_df_display = summary_df.copy()
    summary_df_display['Runway (mesi)'] = summary_df_display['Runway (mesi)'].apply(lambda x: f"{x:.1f}" if np.isfinite(x) else "Profittevole")
    
    display_cols = ['Costo Mensile (€)', 'MRR Totale (€)', 'Utile/Perdita Mensile (€)', 'Saldo di Cassa (€)', 'Runway (mesi)']
    formatters = {col: '{:,.0f}'.format for col in display_cols if col != 'Runway (mesi)'}
    
    print(summary_df_display[display_cols].to_string(formatters=formatters))
    
    # --- Metriche Finanziarie Chiave ---
    print("\n--- METRICHE FINANZIARIE CHIAVE ---")
    
    # --- Punto di Break-Even ---
    break_even_series = df[df['Utile/Perdita Mensile (€)'] >= 0]
    
    if not break_even_series.empty:
        break_even_month_row = break_even_series.iloc[0]
        be_month_index = break_even_month_row['Mese']
        
        # Calcola il burn totale FINO al punto di break-even
        burn_up_to_be_df = df.loc[df['Mese'] <= be_month_index]
        total_burn_at_be = abs(burn_up_to_be_df[burn_up_to_be_df['Utile/Perdita Mensile (€)'] < 0]['Utile/Perdita Mensile (€)'].sum())

        print(f"Capitale totale bruciato per raggiungere il break-even: €{int(total_burn_at_be):,}")
        
        print(f"\n--- PUNTO DI BREAK-EVEN RAGGIUNTO ---")
        print(f"Il punto di pareggio operativo viene raggiunto intorno al mese {int(be_month_index)}.")
        print(f"In quel mese, l'MRR totale stimato di €{int(break_even_month_row['MRR Totale (€)']):,} supera i costi di €{int(break_even_month_row['Costo Mensile (€)']):,}.")

    else:
        total_burned_all_years = abs(df[df['Utile/Perdita Mensile (€)'] < 0]['Utile/Perdita Mensile (€)'].sum())
        print(f"Capitale totale bruciato in {len(df)//12} anni (break-even non raggiunto): €{int(total_burned_all_years):,}")
        
        print("\n--- PUNTO DI BREAK-EVEN NON RAGGIUNTO ---")
        print(f"Con i parametri attuali, il break-even non viene raggiunto entro {len(df)//12} anni.")
        print(f"Alla fine del periodo, il deficit mensile è di €{int(df.iloc[-1]['Utile/Perdita Mensile (€)']):,}.")


def plot_results(df):
    """
    Crea un grafico dei costi vs ricavi per visualizzare il break-even.
    """
    print("\nGenerazione grafico...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Asse Y sinistro per Costi e Ricavi
    ax1.plot(df['Mese'], df['Costo Mensile (€)'], label='Costo Operativo Mensile', color='red', linestyle='--')
    
    # Grafico ad area cumulata per le componenti dei ricavi
    revenue_components = {
        'MRR Piani': df['MRR da Piani (€)'],
        'MRR Add-on': df['MRR da Add-on (€)'],
        'MRR Store': df['MRR da Store (€)']
    }
    colors = ['#2ca02c', '#98df8a', '#ff7f0e'] # Verde, Verde chiaro, Arancione
    ax1.stackplot(df['Mese'], revenue_components.values(), labels=revenue_components.keys(), colors=colors, alpha=0.7)

    ax1.set_xlabel('Mesi dalla Partenza', fontsize=12)
    ax1.set_ylabel('Euro (€)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Asse Y destro per il Runway
    ax2 = ax1.twinx()
    runway_plot_data = df['Runway (mesi)'].replace(float('inf'), np.nan)
    ax2.plot(df['Mese'], runway_plot_data, label='Runway (mesi)', color='purple', linestyle=':', marker='.')
    ax2.set_ylabel('Runway (Mesi)', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(bottom=0, top=max(runway_plot_data.dropna()) * 1.1 if not runway_plot_data.dropna().empty else 100)


    # Evidenzia il punto di break-even
    break_even_series = df[df['Utile/Perdita Mensile (€)'] >= 0]
    if not break_even_series.empty:
        be_month_row = break_even_series.iloc[0]
        be_month, be_value = be_month_row['Mese'], be_month_row['MRR Totale (€)']
        ax1.axvline(x=be_month, color='blue', linestyle=':', linewidth=2, label=f'Break-Even (Mese {int(be_month)})')
        ax1.annotate(f'Break-Even\n~Mese {int(be_month)}\n~€{int(be_value):,}',
                     xy=(be_month, be_value), xytext=(be_month + 3, be_value / 2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

    # Formattazione del grafico
    fig.suptitle('Analisi Finanziaria: Break-Even e Runway', fontsize=16, fontweight='bold')
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=5)
    
    # CORREZIONE: Fai partire l'asse X da 1 invece che da 0
    plt.xticks(np.arange(min(df['Mese']), max(df['Mese'])+1, 6))
    ax1.set_xlim(left=1)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Salva il grafico su file
    plt.savefig("financial_projection.png")
    print("\nGrafico salvato in financial_projection.png")
    
    # Mostra il grafico a video
    plt.show()


# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    results_df = run_simulation(simulation_years=5, initial_capital=INITIAL_CAPITAL)
    display_results(results_df)
    plot_results(results_df)
