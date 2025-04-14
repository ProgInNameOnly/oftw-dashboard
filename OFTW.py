import pandas_datareader.data as web
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import openai
import traceback
import logging
import sys
import os
import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Set up OpenAI API
openai.api_key = "YOUR_AI_API_KEY"

# Fetch historical exchange rates
def get_historical_exchange_rates(start_date="2014-03-01", end_date=None):
    if end_date is None:
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    exchange_rates = {}
    currencies = ['DEXUSUK', 'DEXCAUS', 'DEXUSAL', 'DEXUSEU', 'DEXSIUS', 'DEXSZUS']
    try:
        for currency in currencies:
            df = web.DataReader(currency, "fred", start_date, end_date)
            df.reset_index(inplace=True)
            df['DATE'] = pd.to_datetime(df['DATE'])
            complete_dates = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='D')
            complete_df = pd.DataFrame({'DATE': complete_dates})
            merged_df = pd.merge(complete_df, df, on='DATE', how='left')
            merged_df.set_index('DATE', inplace=True)
            merged_df[currency] = merged_df[currency].interpolate(method='linear')
            exchange_rates[currency] = merged_df
            logger.debug(f"Fetched and processed exchange rates for {currency}")
        return exchange_rates
    except Exception as e:
        logger.error(f"Error fetching exchange rates: {e}")
        raise

# Convert amounts to USD
def convert_to_usd(df, amount_col, currency_col, date_col, exchange_rates):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(by=date_col)
    for currency, rate_df in exchange_rates.items():
        df = pd.merge_asof(df, rate_df.reset_index(), left_on=date_col, right_on='DATE', direction='nearest')
        df = df.drop(columns=['DATE'])
    usd_col = f"{amount_col}_usd"
    df[usd_col] = df.apply(
        lambda row: row[amount_col] * row['DEXUSUK'] if row[currency_col] == 'GBP' else
                    row[amount_col] / row['DEXCAUS'] if row[currency_col] == 'CAD' else
                    row[amount_col] * row['DEXUSAL'] if row[currency_col] == 'AUD' else
                    row[amount_col] * row['DEXUSEU'] if row[currency_col] == 'EUR' else
                    row[amount_col] / row['DEXSIUS'] if row[currency_col] == 'SGD' else
                    row[amount_col] / row['DEXSZUS'] if row[currency_col] == 'CHF' else
                    row[amount_col], axis=1
    )
    df = df.drop(columns=['DEXUSUK', 'DEXCAUS', 'DEXUSAL', 'DEXUSEU', 'DEXSIUS', 'DEXSZUS'], errors='ignore')
    logger.debug(f"Converted {amount_col} to {usd_col} in DataFrame")
    return df

try:
    logger.debug("Fetching exchange rates...")
    exchange_rates = get_historical_exchange_rates("2014-03-01", "2025-02-28")
    logger.debug("Loading data...")
    pledges_url = "https://storage.googleapis.com/plotly-app-challenge/one-for-the-world-pledges.json"
    payments_url = "https://storage.googleapis.com/plotly-app-challenge/one-for-the-world-payments.json"
    pledges_df = pd.read_json(pledges_url)
    payments_df = pd.read_json(payments_url)
    logger.debug("Converting currencies to USD...")
    pledges_df = convert_to_usd(pledges_df, 'contribution_amount', 'currency', 'pledge_created_at', exchange_rates)
    payments_df = convert_to_usd(payments_df, 'amount', 'currency', 'date', exchange_rates)
    logger.debug("Merging pledges and payments...")
    merged_df = pd.merge(pledges_df, payments_df, on='pledge_id', how='outer', suffixes=('_pledge', '_payment'))
    merged_df['pledge_created_at'] = pd.to_datetime(merged_df['pledge_created_at'])
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df[~merged_df['portfolio'].isin(['One for the World Discretionary Fund', 'One for the World Operating Costs'])]
    merged_df['counterfactual_mm'] = merged_df['amount_usd'] * merged_df['counterfactuality'].fillna(0)
    ytd_start = pd.Timestamp('2024-07-01')
    ytd_end = pd.Timestamp('2025-03-09')
    ytd_df = merged_df[(merged_df['date'] >= ytd_start) & (merged_df['date'] <= ytd_end)]
    total_mm_ytd = ytd_df['counterfactual_mm'].sum()
    attrition_pledges = merged_df[merged_df['pledge_status'].isin(['Payment failure', 'Churned donor'])]
    total_pledges = len(merged_df)
    attrition_rate = (len(attrition_pledges) / total_pledges * 100) if total_pledges > 0 else 0
    if 'donor_id' in merged_df.columns:
        active_donors = merged_df[merged_df['pledge_status'].isin(['Active donor', 'one-time'])]['donor_id'].nunique()
    elif 'id' in merged_df.columns:
        active_donors = merged_df[merged_df['pledge_status'].isin(['Active donor', 'one-time'])]['id'].nunique()
    else:
        active_donors = 0
    arr_data = (
        merged_df[merged_df['pledge_status'] == 'Active donor']
        .groupby('donor_chapter')['amount_usd']
        .sum()
        .reset_index()
    )
    arr_data = arr_data.nlargest(10, 'amount_usd')

    # Initialize Dash app with Bootstrap stylesheet
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Create a Navbar with the One for the World logo on the top left
    navbar = dbc.Navbar(
        children=[
            dbc.Row([
                dbc.Col(
                    html.Img(src='assets/oftw_logo.png', height="60px"),
                    width="auto"
                ),
                dbc.Col(
                    html.H2("OFTW Dashboard", style={'marginLeft': '10px'}),
                    width="auto",
                ),
            ], align='center', className="g-0"),
        ],
        color="light",
        dark=False,
        style={'padding': '10px'}
    )

    # Define chart components
    arr_chart = dcc.Graph(id='arr-chart')
    attrition_chart = dcc.Graph(id='attrition-chart')
    time_lag_chart = dcc.Graph(id='time-lag-chart')

    # DataTable
    data_table = dash_table.DataTable(
        id='merged-data-table',
        columns=[{'name': i, 'id': i} for i in merged_df.columns],
        data=merged_df.to_dict('records'),
        page_action='native',
        page_size=10,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'color': '#2C3E50'},
        style_data={'backgroundColor': 'rgb(255, 255, 255)', 'color': '#2C3E50'}
    )

    # Dropdown filters
    donor_chapters = [{'label': 'All Chapters', 'value': 'All'}] + [
        {'label': chapter, 'value': chapter} for chapter in merged_df['donor_chapter'].dropna().unique()
    ]
    pledge_statuses = [{'label': 'All Statuses', 'value': 'All'}] + [
        {'label': status, 'value': status} for status in merged_df['pledge_status'].dropna().unique()
    ]

    glossary_data = [
        ("donor_id", "A unique identifier assigned to each donor.", "Pledges, Payments", "Used to track individual donors across pledges and payments."),
        ("pledge_id", "A unique identifier for each pledge; a new pledge is created if a donor changes amount or frequency.", "Pledges, Payments", "Key for merging datasets; multiple pledges per donor are possible."),
        ("donor_chapter", "The channel or organization where a donor first signed their pledge (e.g., university chapter).", "Pledges", "No difference between 'n/a' and empty cells; both indicate unknown."),
        ("chapter_type", "Categories of `donor_chapter` (e.g., UG for Undergraduate).", "Pledges", "Helps group chapters by type for analysis."),
        ("pledge_status", "Status of the pledge (e.g., Active donor, Pledged donor, Payment failure, Churned donor).", "Pledges", "Focus on 'Active' or 'Pledged' for key metrics; 'Payment failure' and 'Churned' indicate attrition."),
        ("pledge_created_at", "The date and time when the pledge was created.", "Pledges", "Used for tracking pledge initiation and USD conversion."),
        ("pledge_starts_at", "The date when the pledge payment schedule begins.", "Pledges", "Relevant for future pledge analysis."),
        ("pledge_ended_at", "The date when the pledge payment schedule ends (if applicable).", "Pledges", "Indicates completed or cancelled pledges."),
        ("contribution_amount", "The amount of money the donor pledged to contribute, in original currency.", "Pledges", "Converted to `contribution_amount_usd` for consistency."),
        ("contribution_amount_usd", "The pledged amount converted to USD.", "Pledges", "Basis for Annualized Run Rate (ARR) calculations when using pledged amounts."),
        ("currency", "The currency in which the pledged payments are to be made.", "Pledges, Payments", "Used to convert amounts to USD for consistency."),
        ("frequency", "The frequency of pledged payments (e.g., monthly, one-time).", "Pledges", "Affects ARR and payment scheduling."),
        ("payment_platform", "The platform that processed the payment (e.g., Benevity, Donational).", "Pledges, Payments", "Useful for analyzing platform performance."),
        ("id", "A unique identifier for each payment record.", "Payments", "Distinct from `pledge_id`; used for payment tracking."),
        ("portfolio", "The allocation of the donation (e.g., OFTW Top Picks, Entire OFTW Portfolio).", "Payments", "Excludes 'Discretionary Fund' and 'Operating Costs' for Money Moved calculations."),
        ("amount", "The amount of money donated in a payment, in original currency.", "Payments", "Converted to `amount_usd` for Money Moved calculations."),
        ("amount_usd", "The payment amount converted to USD.", "Payments", "Basis for Money Moved; multiplied by `counterfactuality` for impact assessment."),
        ("date", "The date when the payment was made.", "Payments", "Used for YTD, time lag calculations, and USD conversion."),
        ("counterfactual", "A value between 0 and 1 indicating the likelihood that the donation wouldn’t have occurred without OFTW (0 = 0%, 1 = 100%).", "Payments", "Multiplied by `amount_usd` to calculate counterfactual Money Moved."),
        ("Money Moved (YTD)", "Total amount of money moved year-to-date, adjusted by `counterfactuality`, in USD.", "KPIs", "Current YTD is July 1, 2024, to March 09, 2025; excludes discretionary/operating costs."),
        ("Counterfactual MM", "Money Moved multiplied by the `counterfactuality` value to reflect OFTW’s unique impact, in USD.", "KPIs", "Key impact metric; excludes certain portfolios."),
        ("Active Annualized Run Rate (ARR)", "Total donation amount from active pledges, in USD.", "KPIs", "Based on `amount_usd` for `Active donor` pledges; could use `contribution_amount_usd` for pledged amounts."),
        ("Pledge Attrition Rate", "Proportion of pledges with status 'Payment failure' or 'Churned donor' relative to all pledges.", "KPIs", "Indicates donor retention; helps target interventions."),
        ("Total Number of Active Donors", "Count of unique `donor_id` with `pledge_status` 'Active donor' or 'one-time'.", "KPIs", "Tracks active donor base."),
        ("Total Number of Active Pledges", "Count of unique `pledge_id` with `pledge_status` 'Active donor'.", "KPIs", "Measures current payment commitments."),
        ("Chapter ARR", "ARR broken down by `donor_chapter` and `chapter_type`, in USD.", "KPIs", "Identifies high-performing chapters."),
        ("Fiscal Year", "The OFTW financial year, running from July 1 to June 30.", "KPIs", "Current fiscal year is July 1, 2024, to June 30, 2025.")
    ]
    glossary = html.Table(
        [html.Tr([html.Th(col, style={'padding': '5px'}) for col in ['Term', 'Definition', 'Source', 'Notes']])] +
        [html.Tr([html.Td(term, style={'padding': '5px'}),
                  html.Td(definition, style={'padding': '5px'}),
                  html.Td(source, style={'padding': '5px'}),
                  html.Td(notes, style={'padding': '5px'})])
         for term, definition, source, notes in glossary_data],
        style={'width': '100%', 'border': '1px solid #ddd', 'margin': '20px 0', 'border-collapse': 'collapse'},
        id='glossary-table'
    )

    def get_ai_response(query):
        logger.debug(f"AI query received: {query}")
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-search-preview-2025-03-11",  # Most up-to-date model
                messages=[{"role": "user", "content": query}],
                max_tokens=500
            )
            logger.debug("AI response generated successfully")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in get_ai_response: {traceback.format_exc()}")
            return f"Error: {str(e)}"

    # ARR by chapter (already computed)
    arr_data = (
        merged_df[merged_df['pledge_status'] == 'Active donor']
        .groupby('donor_chapter')['amount_usd']
        .sum()
        .reset_index()
    )
    arr_data = arr_data.nlargest(10, 'amount_usd')
    # Currency conversion (simplified; replace with actual logic)
    merged_df['amount_usd'] = merged_df['amount']  # Placeholder
    merged_df['contribution_amount_usd'] = merged_df['contribution_amount']  # Placeholder
    merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

    # Calculate counterfactual_mm using counterfactuality
    merged_df['counterfactual_mm'] = merged_df.apply(
        lambda row: row['amount_usd'] if row['counterfactuality'] == True else 0, axis=1
    )

    # Export to CSV with counterfactual_mm
    merged_df.to_csv('merged_dataset_with_usd.csv', index=False)
    logger.debug(f"Exported merged_df to CSV with columns: {merged_df.columns.tolist()}")

    # Filter for YTD period (July 1, 2024 - March 9, 2025)
    ytd_df = merged_df[merged_df['date'].between(pd.Timestamp('2024-07-01'), pd.Timestamp('2025-03-09'))].copy()
    exclude_portfolios = ["One for the World Discretionary Fund", "One for the World Operating Costs"]
    ytd_df = ytd_df[~ytd_df['portfolio'].isin(exclude_portfolios)]

    # Recalculate total_mm_ytd after exclusion
    total_mm_ytd = ytd_df['counterfactual_mm'].sum()
    active_donors = ytd_df[ytd_df['pledge_status'] == 'Active donor']['donor_id_pledge'].nunique()
    attrition_pledges = merged_df[merged_df['pledge_status'].isin(['Payment failure', 'Churned donor'])]['pledge_id'].unique()
    attrition_rate = len(attrition_pledges) / merged_df['pledge_id'].nunique() * 100 if merged_df['pledge_id'].nunique() > 0 else 0

    # Monthly Counterfactual MM
    ytd_df.loc[:, 'month'] = ytd_df['date'].dt.to_period('M')
    monthly_mm = ytd_df.groupby('month')['counterfactual_mm'].sum().reset_index()
    monthly_mm['month'] = monthly_mm['month'].astype(str)

    # Debug info
    july_2024_mm = monthly_mm[monthly_mm['month'] == '2024-07']['counterfactual_mm'].sum()
    logger.debug(f"Monthly Counterfactual Money Moved for July 2024: ${july_2024_mm:,.2f}")
    logger.debug(f"Number of rows in ytd_df for July 2024: {len(ytd_df[ytd_df['month'] == '2024-07'])}")
    logger.debug(f"Sample of counterfactuality in ytd_df: {ytd_df['counterfactuality'].head().tolist()}")
    logger.debug(f"Counterfactuality unique values in ytd_df: {ytd_df['counterfactuality'].unique()}")

    # Other calculations
    arr_data = merged_df[merged_df['pledge_status'] == 'Active donor'].groupby('donor_chapter').agg({
        'amount_usd': 'sum'
    }).reset_index().sort_values('amount_usd', ascending=False).head(10)
    # New Calculations for Additional Metrics
    total_arr = arr_data['amount_usd'].sum()
    logger.debug(f"Total ARR calculated: ${total_arr:,.2f}")
    future_arr = merged_df[merged_df['pledge_status'] == 'Pledged donor']['contribution_amount_usd'].sum()
    logger.debug(f"Future ARR calculated: ${future_arr:,.2f}")
    total_active_pledges = merged_df[merged_df['pledge_status'] == 'Active donor']['pledge_id'].nunique()
    logger.debug(f"Total Active Pledges: {total_active_pledges}")
    total_pledges_all = merged_df[merged_df['pledge_status'].isin(['Active donor', 'Pledged donor'])]['pledge_id'].nunique()
    total_future_pledges = merged_df[merged_df['pledge_status'] == 'Pledged donor']['pledge_id'].nunique()
    logger.debug(f"Total Pledges (Active + Future): {total_pledges_all}")
    logger.debug(f"Future Pledges: {total_future_pledges}")
    chapter_type_arr = (
        merged_df[merged_df['pledge_status'] == 'Active donor']
        .groupby('chapter_type')['amount_usd']
        .sum()
        .reset_index()
    )
    total_chapter_arr = chapter_type_arr['amount_usd'].sum()
    logger.debug(f"Chapter ARR by Type: ${total_chapter_arr:,.2f}")

    if 'payment_platform' not in ytd_df.columns:
        ytd_df = ytd_df.merge(payments_df[['pledge_id', 'payment_platform']], on='pledge_id', how='left')
        logger.debug(f"Added payment_platform from payments_df. New columns: {ytd_df.columns.tolist()}")
    mm_by_platform = ytd_df.groupby('payment_platform')['counterfactual_mm'].sum().reset_index()
    mm_by_source = ytd_df.groupby('chapter_type')['counterfactual_mm'].sum().reset_index()
    if 'frequency' not in ytd_df.columns:
        ytd_df = ytd_df.merge(merged_df[['pledge_id', 'frequency']], on='pledge_id', how='left')
        logger.debug(f"Added frequency from payments_df. New columns: {ytd_df.columns.tolist()}")
    mm_by_frequency = ytd_df.groupby('frequency')['counterfactual_mm'].sum().reset_index()
    attrition_data = merged_df.groupby(['donor_chapter', 'frequency']).agg({
        'pledge_status': lambda x: (x.isin(['Payment failure', 'Churned donor']).sum() / len(x) * 100) if len(x) > 0 else 0,
        'pledge_id': 'nunique'
    }).reset_index()
    attrition_data = attrition_data.rename(columns={'pledge_status': 'attrition_rate', 'pledge_id': 'pledge_count'})
    top_chapters = merged_df.groupby('donor_chapter')['pledge_id'].nunique().nlargest(10).index
    attrition_data = attrition_data[attrition_data['donor_chapter'].isin(top_chapters)]

    # Define goals from the table
    goal_mm = 1800000  
    goal_arr = 1200000  
    goal_attrition = 18  
    goal_active_donors = 1200  
    goal_active_pledges = 850  
    goal_chapter_arr = 670000  
    goal_total_pledges = 1850  
    goal_future_pledges = 1000  
    goal_future_arr = 600000  

    percent_mm = (total_mm_ytd / goal_mm * 100) if goal_mm > 0 and not pd.isna(total_mm_ytd) else 0
    percent_arr = (total_arr / goal_arr * 100) if goal_arr > 0 and not pd.isna(total_arr) else 0
    percent_attrition = (attrition_rate / goal_attrition * 100) if goal_attrition > 0 and not pd.isna(attrition_rate) else 0
    percent_active_donors = (active_donors / goal_active_donors * 100) if goal_active_donors > 0 and not pd.isna(active_donors) else 0
    percent_active_pledges = (total_active_pledges / goal_active_pledges * 100) if goal_active_pledges > 0 and not pd.isna(total_active_pledges) else 0
    percent_chapter_arr = (total_chapter_arr / goal_chapter_arr * 100) if goal_chapter_arr > 0 and not pd.isna(total_chapter_arr) else 0
    percent_total_pledges = (total_pledges_all / goal_total_pledges * 100) if goal_total_pledges > 0 and not pd.isna(total_pledges_all) else 0
    percent_future_pledges = (total_future_pledges / goal_future_pledges * 100) if goal_future_pledges > 0 and not pd.isna(total_future_pledges) else 0
    percent_future_arr = (future_arr / goal_future_arr * 100) if goal_future_arr > 0 and not pd.isna(future_arr) else 0

    # App layout: include the Navbar (with logo) at the top followed by a Dark Mode switch and Tabs
    app.layout = html.Div([
        navbar,
        dbc.Row(
            dbc.Col(
                dbc.Switch(id='dark-mode-switch', label='Dark Mode', value=False, style={'margin': '10px'}),
                width={'size': 2, 'offset': 10}
            )
        ),
        dcc.Tabs([
            dcc.Tab(label='Overview', style={'padding': '20px'}, children=[
                html.H1("OFTW Dashboard", id='dashboard-title', style={'textAlign': 'center', 'marginBottom': '30px'}),
                html.H3("Key Metrics (2025 Targets)", style={'textAlign': 'center', 'marginBottom': '30px'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id='money-moved-card', children=[
                            dbc.CardBody([
                                html.H5("Total Counterfactual Money Moved (YTD)", id='money-moved-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"${total_mm_ytd:,.2f}", id='money-moved-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_mm:.1f}% of ${goal_mm:,.0f} goal)", id='money-moved-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Sum of counterfactual money moved (USD) for payments from July 1, 2024, to March 9, 2025, "
                            "excluding 'One for the World Discretionary Fund' and 'One for the World Operating Costs' portfolios.",
                            target='money-moved-card',
                            placement='top'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Card(id='arr-total-card', children=[
                            dbc.CardBody([
                                html.H5("Active Annualized Run Rate (ARR)", id='arr-total-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"${total_arr:,.2f}", id='arr-total-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_arr:.1f}% of ${goal_arr:,.0f} goal)", id='arr-total-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Sum of USD amounts for active donor pledges, annualized based on pledge frequency.",
                            target='arr-total-card',
                            placement='top'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Card(id='attrition-rate-card', children=[
                            dbc.CardBody([
                                html.H5("Pledge Attrition Rate", id='attrition-rate-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"{attrition_rate:.2f}%", id='attrition-rate-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_attrition:.1f}% of {goal_attrition}% target)", id='attrition-rate-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Percentage of pledges with status 'Payment failure' or 'Churned donor' out of total unique pledges.",
                            target='attrition-rate-card',
                            placement='top'
                        )
                    ], width=4),
                ], justify='around', style={'marginBottom': '20px'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id='active-donors-card', children=[
                            dbc.CardBody([
                                html.H5("Total Active Donors", id='active-donors-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"{active_donors:,}", id='active-donors-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_active_donors:.1f}% of {goal_active_donors} goal)", id='active-donors-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Number of unique donors with active pledge status within the YTD period.",
                            target='active-donors-card',
                            placement='top'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Card(id='active-pledges-card', children=[
                            dbc.CardBody([
                                html.H5("Total Active Pledges", id='active-pledges-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"{total_active_pledges:,}", id='active-pledges-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_active_pledges:.1f}% of {goal_active_pledges} goal)", id='active-pledges-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Number of unique pledge IDs with 'Active donor' status.",
                            target='active-pledges-card',
                            placement='top'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Card(id='chapter-arr-card', children=[
                            dbc.CardBody([
                                html.H5("Chapter ARR (by Chapter Type)", id='chapter-arr-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"${total_chapter_arr:,.2f}", id='chapter-arr-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_chapter_arr:.1f}% of ${goal_chapter_arr:,.0f} goal)", id='chapter-arr-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Sum of USD amounts for active donor pledges, aggregated by chapter type.",
                            target='chapter-arr-card',
                            placement='top'
                        )
                    ], width=4),
                ], justify='around', style={'marginBottom': '20px'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id='total-pledges-card', children=[
                            dbc.CardBody([
                                html.H5("Total Pledges (Active + Future)", id='total-pledges-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"{total_pledges_all:,}", id='total-pledges-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_total_pledges:.1f}% of {goal_total_pledges} goal)", id='total-pledges-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Number of unique pledge IDs with 'Active donor' or 'Pledged donor' status.",
                            target='total-pledges-card',
                            placement='top'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Card(id='future-pledges-card', children=[
                            dbc.CardBody([
                                html.H5("Future Pledges", id='future-pledges-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"{total_future_pledges:,}", id='future-pledges-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_future_pledges:.1f}% of {goal_future_pledges} goal)", id='future-pledges-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Number of unique pledge IDs with 'Pledged donor' status.",
                            target='future-pledges-card',
                            placement='top'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Card(id='future-arr-card', children=[
                            dbc.CardBody([
                                html.H5("Future ARR", id='future-arr-title',
                                        style={'textAlign': 'center', 'fontSize': 18}),
                                html.P(f"${future_arr:,.2f}", id='future-arr-value',
                                       style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
                                html.P(f"({percent_future_arr:.1f}% of ${goal_future_arr:,.0f} goal)", id='future-arr-percent',
                                       style={'textAlign': 'center', 'fontSize': 14})
                            ])
                        ], style={'marginBottom': '20px'}),
                        dbc.Tooltip(
                            "Sum of USD amounts for pledged donor contributions, annualized based on pledge frequency.",
                            target='future-arr-card',
                            placement='top'
                        )
                    ], width=4),
                ], justify='around', style={'marginBottom': '20px'}),
            ]),
            dcc.Tab(label='Money Moved Breakdowns', style={'padding': '10px'}, children=[
                html.H2("Money Moved Breakdowns", style={'textAlign': 'center', 'marginBottom': '20px'}),
                dcc.Graph(
                    id='monthly-mm-chart',
                    figure=px.line(monthly_mm, x='month', y='counterfactual_mm', title='Monthly Counterfactual Money Moved (YTD)',
                                   labels={'counterfactual_mm': 'Money Moved (USD)', 'month': 'Month'}).update_layout(
                        xaxis_title="Month", yaxis_title="Money Moved (USD)", xaxis_tickangle=45, height=300,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', font_color='black',
                        margin=dict(l=50, r=50, t=50, b=50)
                    ).update_traces(line_color='#3399FF')
                ),
                dcc.Graph(
                    id='mm-by-platform-chart',
                    figure=px.bar(mm_by_platform, x='payment_platform', y='counterfactual_mm', title='Money Moved by Platform (YTD)',
                                  labels={'counterfactual_mm': 'Money Moved (USD)', 'payment_platform': 'Platform'}).update_layout(
                        height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', font_color='black'
                    ).update_traces(marker_color='#3399FF')
                ),
                dcc.Graph(
                    id='mm-by-source-chart',
                    figure=px.bar(mm_by_source, x='chapter_type', y='counterfactual_mm', title='Money Moved by Source (Chapter Type) (YTD)',
                                  labels={'counterfactual_mm': 'Money Moved (USD)', 'chapter_type': 'Chapter Type'}).update_layout(
                        height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', font_color='black'
                    ).update_traces(marker_color='#3399FF')
                ),
                dcc.Graph(
                    id='mm-by-frequency-chart',
                    figure=px.bar(mm_by_frequency, x='frequency', y='counterfactual_mm', title='Money Moved by Frequency (YTD)',
                                  labels={'counterfactual_mm': 'Money Moved (USD)', 'frequency': 'Frequency'}).update_layout(
                        height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', font_color='black'
                    ).update_traces(marker_color='#3399FF')
                ),
            ]),
            dcc.Tab(label='Attrition Analysis', style={'padding': '10px'}, children=[
                html.H2("Attrition Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
                dcc.Graph(
                    id='attrition-by-chapter-chart',
                    figure=px.bar(attrition_data, x='frequency', y='attrition_rate', color='donor_chapter',
                                  title='Attrition Rate by Chapter and Pledge Frequency (Top 10 Chapters)',
                                  labels={'attrition_rate': 'Attrition Rate (%)', 'frequency': 'Pledge Frequency'},
                                  height=500).update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', font_color='black',
                        margin=dict(l=50, r=50, t=100, b=50)
                    ).update_traces(marker_color='#3399FF')
                ),
                dcc.Loading(id='loading-arr', children=[arr_chart], type='default'),
                dcc.Loading(id='loading-grid', children=[dbc.Row([dbc.Col(attrition_chart, width=6),
                                                                  dbc.Col(time_lag_chart, width=6)],
                                                                 style={'margin-top': '20px'})], type='default'),
            ]),
            dcc.Tab(label='Data Explorer', style={'padding': '10px'}, children=[
                html.H3("Merged Data Sample", id='table-title'),
                dbc.Row([
                    dbc.Col([html.Label("Filter by Donor Chapter:", id='chapter-filter-label'),
                             dcc.Dropdown(id='chapter-filter', options=donor_chapters, value='All', style={'width': '100%'})], width=3),
                    dbc.Col([html.Label("Filter by Pledge Status:", id='status-filter-label'),
                             dcc.Dropdown(id='status-filter', options=pledge_statuses, value='All', style={'width': '100%'})], width=3),
                ], style={'margin-bottom': '10px'}),
                dbc.Row([dbc.Col(dcc.Download(id="download-data-csv"), width=2),
                         dbc.Col(html.Button("Export to CSV", id='btn-csv', n_clicks=0), width=2)]),
                dcc.Loading(id='loading-table', children=[data_table], type='default')
            ]),
            dcc.Tab(label='Glossary', style={'padding': '0'}, children=[
                html.H2("OFTW Data Glossary", id='glossary-title', style={'textAlign': 'center', 'margin': '10px 0'}),
                html.P("This glossary defines key terms and metrics.", id='glossary-description',
                       style={'textAlign': 'center', 'margin': '0'}),
                html.Div(glossary, id='glossary-table-container', style={'padding': '0', 'margin': '0'})
            ]),
            dcc.Tab(label='AI Assistant', style={'padding': '0'}, children=[
                html.H2("OFTW AI Assistant", id='ai-title', style={'textAlign': 'center', 'margin': '10px 0'}),
                html.P("Ask questions about data fields, metrics, or charts.", id='ai-description',
                       style={'textAlign': 'center', 'margin': '0'}),
                dcc.Input(id='ai-query-input', type='text', placeholder='Enter your question...',
                          style={'width': '80%', 'margin': '10px'}),
                html.Button('Submit', id='ai-submit-button', n_clicks=0, style={'margin': '10px'}),
                html.Div(id='ai-response', style={'margin': '10px', 'whiteSpace': 'pre-wrap', 'minHeight': '100px'})
            ])
        ])
    ], id='main-container', style={'minHeight': '100vh', 'margin': '0', 'padding': '10px', 'backgroundColor': '#ECF0F1'})

    @app.callback(
        [Output('main-container', 'style'),
         Output('dashboard-title', 'style'),
         Output('money-moved-title', 'style'),
         Output('money-moved-value', 'style'),
         Output('money-moved-percent', 'style'),
         Output('arr-total-title', 'style'),
         Output('arr-total-value', 'style'),
         Output('arr-total-percent', 'style'),
         Output('attrition-rate-title', 'style'),
         Output('attrition-rate-value', 'style'),
         Output('attrition-rate-percent', 'style'),
         Output('active-donors-title', 'style'),
         Output('active-donors-value', 'style'),
         Output('active-donors-percent', 'style'),
         Output('active-pledges-title', 'style'),
         Output('active-pledges-value', 'style'),
         Output('active-pledges-percent', 'style'),
         Output('chapter-arr-title', 'style'),
         Output('chapter-arr-value', 'style'),
         Output('chapter-arr-percent', 'style'),
         Output('total-pledges-title', 'style'),
         Output('total-pledges-value', 'style'),
         Output('total-pledges-percent', 'style'),
         Output('future-pledges-title', 'style'),
         Output('future-pledges-value', 'style'),
         Output('future-pledges-percent', 'style'),
         Output('future-arr-title', 'style'),
         Output('future-arr-value', 'style'),
         Output('future-arr-percent', 'style'),
         Output('table-title', 'style'),
         Output('monthly-mm-chart', 'figure'),
         Output('mm-by-platform-chart', 'figure'),
         Output('mm-by-source-chart', 'figure'),
         Output('mm-by-frequency-chart', 'figure'),
         Output('attrition-by-chapter-chart', 'figure'),
         Output('arr-chart', 'figure'),
         Output('attrition-chart', 'figure'),
         Output('time-lag-chart', 'figure'),
         Output('merged-data-table', 'style_header'),
         Output('merged-data-table', 'style_data'),
         Output('ai-query-input', 'style'),
         Output('ai-response', 'style'),
         Output('glossary-table-container', 'style'),
         Output('glossary-table', 'style'),
         Output('glossary-title', 'style'),
         Output('glossary-description', 'style'),
         Output('ai-title', 'style'),
         Output('ai-description', 'style'),
         Output('chapter-filter-label', 'style'),
         Output('status-filter-label', 'style')],
        Input('dark-mode-switch', 'value')
    )
    def update_dark_mode(dark_mode):
        if dark_mode:
            background_color = '#1E1E1E'
            text_color = '#CCCCCC'
            chart_paper_color = '#2E2E2E'
            chart_text_color = '#CCCCCC'
            table_header_bg = '#333333'
            table_data_bg = '#2E2E2E'
            table_text_color = '#CCCCCC'
            input_bg = '#2E2E2E'
            input_text = '#FFFFFF'
            bar_color = '#3399FF'
            table_container_bg = '#2E2E2E'
            response_bg = '#2E2E2E'
            table_bg = '#2E2E2E'
            card_bg = '#2E2E2E'
        else:
            background_color = '#ECF0F1'
            text_color = '#2C3E50'
            chart_paper_color = '#FFFFFF'
            chart_text_color = '#2C3E50'
            table_header_bg = 'rgb(230, 230, 230)'
            table_data_bg = 'rgb(255, 255, 255)'
            table_text_color = '#2C3E50'
            input_bg = '#FFFFFF'
            input_text = '#2C3E50'
            bar_color = '#1f77b4'
            table_container_bg = '#FFFFFF'
            response_bg = '#FFFFFF'
            table_bg = '#FFFFFF'
            card_bg = '#FFFFFF'

        logger.debug(f"Updating charts with dark_mode={dark_mode}, bar_color={bar_color}")
        monthly_mm_fig = px.line(monthly_mm, x='month', y='counterfactual_mm', title='Monthly Counterfactual Money Moved (YTD)',
                                 labels={'counterfactual_mm': 'Money Moved (USD)', 'month': 'Month'}).update_layout(
            xaxis_title="Month", yaxis_title="Money Moved (USD)", xaxis_tickangle=45, height=300,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=chart_paper_color, font_color=chart_text_color,
            margin=dict(l=50, r=50, t=50, b=50)
        ).update_traces(line_color=bar_color)
        mm_by_platform_fig = px.bar(mm_by_platform, x='payment_platform', y='counterfactual_mm', title='Money Moved by Platform (YTD)',
                                    labels={'counterfactual_mm': 'Money Moved (USD)', 'payment_platform': 'Platform'}).update_layout(
            height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=chart_paper_color, font_color=chart_text_color
        ).update_traces(marker_color=bar_color)
        mm_by_source_fig = px.bar(mm_by_source, x='chapter_type', y='counterfactual_mm', title='Money Moved by Source (Chapter Type) (YTD)',
                                  labels={'counterfactual_mm': 'Money Moved (USD)', 'chapter_type': 'Chapter Type'}).update_layout(
            height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=chart_paper_color, font_color=chart_text_color
        ).update_traces(marker_color=bar_color)
        mm_by_frequency_fig = px.bar(mm_by_frequency, x='frequency', y='counterfactual_mm', title='Money Moved by Frequency (YTD)',
                                     labels={'counterfactual_mm': 'Money Moved (USD)', 'frequency': 'Frequency'}).update_layout(
            height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=chart_paper_color, font_color=chart_text_color
        ).update_traces(marker_color=bar_color)
        attrition_by_chapter_fig = px.bar(attrition_data, x='frequency', y='attrition_rate', color='donor_chapter',
                                          title='Attrition Rate by Chapter and Pledge Frequency (Top 10 Chapters)',
                                          labels={'attrition_rate': 'Attrition Rate (%)', 'frequency': 'Pledge Frequency'},
                                          height=500).update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=chart_paper_color, font_color=chart_text_color,
            margin=dict(l=50, r=50, t=100, b=50), barmode='group'
        ).update_traces(marker_color=bar_color)
        logger.debug(f"attrition_by_chapter_fig updated with bar_color={bar_color}")
        arr_fig = px.bar(
            arr_data, x='amount_usd', y='donor_chapter', title='Active Annualized Run Rate by Top 10 Chapters',
            labels={'amount_usd': 'Annualized Run Rate (USD)', 'donor_chapter': 'Chapter'}, orientation='h', height=600
        ).update_traces(marker_color=bar_color).update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=chart_paper_color, font_color=chart_text_color,
            xaxis={'title': 'Annualized Run Rate (USD)', 'gridcolor': 'rgba(255,255,255,0.1)'},
            yaxis={'tickangle': 0, 'automargin': True, 'gridcolor': 'rgba(255,255,255,0.1)'},
            margin=dict(l=200, r=50, t=50, b=50)
        )
        attrition_fig = px.pie(
            values=[len(merged_df) - len(attrition_pledges), len(attrition_pledges)], names=['Active', 'Attrition'],
            title='Pledge Attrition Rate', color_discrete_sequence=['#3399FF', '#FF6F61'] if dark_mode else ['#4682B4', '#FF6F61']
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=chart_paper_color, font_color=chart_text_color
        )
        time_lag_fig = px.histogram(
            merged_df.dropna(subset=['date', 'pledge_created_at']),
            x=(merged_df['date'] - merged_df['pledge_created_at']).dt.days, nbins=30, title='Time Lag Distribution (Days)',
            color_discrete_sequence=[bar_color]
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=chart_paper_color, font_color=chart_text_color,
            xaxis={'gridcolor': 'rgba(255,255,255,0.1)'}, yaxis={'gridcolor': 'rgba(255,255,255,0.1)'}
        )

        title_style = {'textAlign': 'center', 'fontSize': 18, 'color': text_color}
        value_style = {'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold', 'color': text_color}
        percent_style = {'textAlign': 'center', 'fontSize': 14, 'color': text_color}

        return (
            {'minHeight': '100vh', 'margin': '0', 'padding': '10px', 'backgroundColor': background_color},
            {'textAlign': 'center', 'marginBottom': '30px', 'color': text_color},
            title_style,
            value_style,
            percent_style,
            title_style,
            value_style,
            percent_style,
            title_style,
            value_style,
            percent_style,
            title_style,
            value_style,
            percent_style,
            title_style,
            value_style,
            percent_style,
            title_style,
            value_style,
            percent_style,
            title_style,
            value_style,
            percent_style,
            title_style,
            value_style,
            percent_style,
            title_style,
            value_style,
            percent_style,
            {'color': text_color},
            monthly_mm_fig,
            mm_by_platform_fig,
            mm_by_source_fig,
            mm_by_frequency_fig,
            attrition_by_chapter_fig,
            arr_fig,
            attrition_fig,
            time_lag_fig,
            {'backgroundColor': table_header_bg, 'color': table_text_color, 'fontWeight': 'bold'},
            {'backgroundColor': table_data_bg, 'color': table_text_color},
            {'width': '80%', 'margin': '10px', 'color': input_text, 'backgroundColor': input_bg},
            {'margin': '10px', 'whiteSpace': 'pre-wrap', 'color': text_color, 'backgroundColor': response_bg, 'minHeight': '100px'},
            {'backgroundColor': table_container_bg, 'padding': '0', 'margin': '0', 'color': text_color},
            {'width': '100%', 'border': '1px solid #ddd', 'margin': '20px 0', 'border-collapse': 'collapse', 'backgroundColor': table_bg, 'color': text_color},
            {'textAlign': 'center', 'margin': '10px 0', 'color': text_color},
            {'textAlign': 'center', 'margin': '0', 'color': text_color},
            {'textAlign': 'center', 'margin': '10px 0', 'color': text_color},
            {'textAlign': 'center', 'margin': '0', 'color': text_color},
            {'color': text_color},
            {'color': text_color}
        )
    @app.callback(
        Output('merged-data-table', 'data'),
        Input('chapter-filter', 'value'),
        Input('status-filter', 'value')
    )
    def update_table(chapter_filter, status_filter):
        df = merged_df.copy()
        if chapter_filter != 'All':
            df = df[df['donor_chapter'] == chapter_filter]
        if status_filter != 'All':
            df = df[df['pledge_status'] == status_filter]
        return df.to_dict('records')

    @app.callback(
        Output("download-data-csv", "data"),
        Input("btn-csv", "n_clicks"),
        State('chapter-filter', 'value'),
        State('status-filter', 'value'),
        prevent_initial_call=True
    )
    def export_table(n_clicks, chapter_filter, status_filter):
        try:
            df = merged_df.copy()
            if chapter_filter != 'All':
                df = df[df['donor_chapter'] == chapter_filter]
            if status_filter != 'All':
                df = df[df['pledge_status'] == status_filter]
            csv_string = df.to_csv(index=False)
            return dcc.send_bytes(csv_string.encode(), filename="merged_data.csv")
        except Exception:
            return None

    @app.callback(
        Output('ai-response', 'children'),
        Input('ai-submit-button', 'n_clicks'),
        State('ai-query-input', 'value')
    )
    def update_ai_response(n_clicks, query):
        if n_clicks > 0 and query:
            return get_ai_response(query)
        return "Please enter a question and click Submit"

except Exception as e:
    logger.error(f"Critical error during app setup: {traceback.format_exc()}")
    raise

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Server error: {traceback.format_exc()}")
        raise
