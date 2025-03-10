import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import openai
import traceback
import logging
import sys
import os


# Set up logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Set up OpenAI API (ensure this is your valid API key)
openai.api_key = os.getenv("OPENAI_API_KEY")


try:
    # Load and merge data
    logger.debug("Loading data...")
    pledges_url = "https://storage.googleapis.com/plotly-app-challenge/one-for-the-world-pledges.json"
    payments_url = "https://storage.googleapis.com/plotly-app-challenge/one-for-the-world-payments.json"
    pledges_df = pd.read_json(pledges_url)
    payments_df = pd.read_json(payments_url)
    merged_df = pd.merge(pledges_df, payments_df, on='pledge_id', how='outer')

    # Preprocess
    merged_df['pledge_created_at'] = pd.to_datetime(merged_df['pledge_created_at'])
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df[~merged_df['portfolio'].isin(['One for the World Discretionary Fund',
                                                        'One for the World Operating Costs'])]
    merged_df['counterfactual_mm'] = merged_df['amount'] * merged_df['counterfactuality'].fillna(0)

    # YTD range for fiscal year (July 1, 2024 - March 09, 2025)
    ytd_start = pd.Timestamp('2024-07-01')
    ytd_end = pd.Timestamp('2025-03-09')
    ytd_df = merged_df[(merged_df['date'] >= ytd_start) & (merged_df['date'] <= ytd_end)]
    total_mm_ytd = ytd_df['counterfactual_mm'].sum()

    # Attrition
    attrition_pledges = merged_df[merged_df['pledge_status'].isin(['Payment failure', 'Churned donor'])]
    total_pledges = len(merged_df)
    attrition_rate = (len(attrition_pledges) / total_pledges * 100) if total_pledges > 0 else 0

    # Active donors
    if 'donor_id' in merged_df.columns:
        active_donors = merged_df[merged_df['pledge_status'].isin(['Active donor', 'one-time'])]['donor_id'].nunique()
    elif 'id' in merged_df.columns:
        active_donors = merged_df[merged_df['pledge_status'].isin(['Active donor', 'one-time'])]['id'].nunique()
    else:
        active_donors = 0

    # ARR by chapter (top 10)
    arr_data = (
        merged_df[merged_df['pledge_status'] == 'Active donor']
        .groupby('donor_chapter')['contribution_amount']
        .sum()
        .reset_index()
    )
    arr_data = arr_data.nlargest(10, 'contribution_amount')

    # Initialize Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
        {'label': chapter, 'value': chapter}
        for chapter in merged_df['donor_chapter'].dropna().unique()
    ]
    pledge_statuses = [{'label': 'All Statuses', 'value': 'All'}] + [
        {'label': status, 'value': status}
        for status in merged_df['pledge_status'].dropna().unique()
    ]

    # Full glossary content
    glossary_data = [
        ("donor_id", "A unique identifier assigned to each donor.", "Pledges, Payments", "Used to track individual donors across pledges and payments."),
        ("pledge_id", "A unique identifier for each pledge; a new pledge is created if a donor changes amount or frequency.", "Pledges, Payments", "Key for merging datasets; multiple pledges per donor are possible."),
        ("donor_chapter", "The channel or organization where a donor first signed their pledge (e.g., university chapter).", "Pledges", "No difference between 'n/a' and empty cells; both indicate unknown."),
        ("chapter_type", "Categories of `donor_chapter` (e.g., UG for Undergraduate).", "Pledges", "Helps group chapters by type for analysis."),
        ("pledge_status", "Status of the pledge (e.g., Active donor, Pledged donor, Payment failure, Churned donor).", "Pledges", "Focus on 'Active' or 'Pledged' for key metrics; 'Payment failure' and 'Churned' indicate attrition."),
        ("pledge_created_at", "The date and time when the pledge was created.", "Pledges", "Used for tracking pledge initiation."),
        ("pledge_starts_at", "The date when the pledge payment schedule begins.", "Pledges", "Relevant for future pledge analysis."),
        ("pledge_ended_at", "The date when the pledge payment schedule ends (if applicable).", "Pledges", "Indicates completed or cancelled pledges."),
        ("contribution_amount", "The amount of money the donor pledged to contribute.", "Pledges", "Basis for Annualized Run Rate (ARR) calculations; may need USD conversion."),
        ("currency", "The currency in which the pledged payments are to be made.", "Pledges, Payments", "Metrics often require conversion to USD for consistency."),
        ("frequency", "The frequency of pledged payments (e.g., monthly, one-time).", "Pledges", "Affects ARR and payment scheduling."),
        ("payment_platform", "The platform that processed the payment (e.g., Benevity, Donational).", "Pledges, Payments", "Useful for analyzing platform performance."),
        ("id", "A unique identifier for each payment record.", "Payments", "Distinct from `pledge_id`; used for payment tracking."),
        ("portfolio", "The allocation of the donation (e.g., OFTW Top Picks, Entire OFTW Portfolio).", "Payments", "Excludes 'Discretionary Fund' and 'Operating Costs' for Money Moved calculations."),
        ("amount", "The amount of money donated in a payment.", "Payments", "Basis for Money Moved; multiplied by `counterfactuality` for impact assessment."),
        ("date", "The date when the payment was made.", "Payments", "Used for YTD and time lag calculations."),
        ("counterfactual", "A value between 0 and 1 indicating the likelihood that the donation wouldn’t have occurred without OFTW (0 = 0%, 1 = 100%).", "Payments", "Multiplied by `amount` to calculate counterfactual Money Moved."),
        ("Money Moved (YTD)", "Total amount of money moved year-to-date, adjusted by `counterfactuality`, for the fiscal year (July 1 to June 30).", "KPIs", "Current YTD is July 1, 2024, to March 09, 2025; excludes discretionary/operating costs."),
        ("Counterfactual MM", "Money Moved multiplied by the `counterfactuality` value to reflect OFTW’s unique impact.", "KPIs", "Key impact metric; excludes certain portfolios."),
        ("Active Annualized Run Rate (ARR)", "Total monthly donation amount from active pledges, converted to USD.", "KPIs", "Based on `contribution_amount` for `Active donor` pledges."),
        ("Pledge Attrition Rate", "Proportion of pledges with status 'Payment failure' or 'Churned donor' relative to all pledges.", "KPIs", "Indicates donor retention; helps target interventions."),
        ("Total Number of Active Donors", "Count of unique `donor_id` with `pledge_status` 'Active donor' or 'one-time'.", "KPIs", "Tracks active donor base."),
        ("Total Number of Active Pledges", "Count of unique `pledge_id` with `pledge_status` 'Active donor'.", "KPIs", "Measures current payment commitments."),
        ("Chapter ARR", "ARR broken down by `donor_chapter` and `chapter_type`.", "KPIs", "Identifies high-performing chapters."),
        ("Fiscal Year", "The OFTW financial year, running from July 1 to June 30.", "KPIs", "Current fiscal year is July 1, 2024, to June 30, 2025.")
    ]

    # Glossary table with full rows
    glossary = html.Table(
        [html.Tr([html.Th(col, style={'padding': '5px'}) for col in ['Term', 'Definition', 'Source', 'Notes']])] +
        [
            html.Tr([
                html.Td(term, style={'padding': '5px'}),
                html.Td(definition, style={'padding': '5px'}),
                html.Td(source, style={'padding': '5px'}),
                html.Td(notes, style={'padding': '5px'})
            ])
            for term, definition, source, notes in glossary_data
        ],
        style={
            'width': '100%',
            'border': '1px solid #ddd',
            'margin': '20px 0',
            'border-collapse': 'collapse'
        },
        id='glossary-table'
    )

    # AI query function with OpenAI integration
    def get_ai_response(query):
        logger.debug(f"AI query received: {query}")
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are an AI assistant for the One For The World (OFTW) dashboard. Use the following glossary to answer questions about data fields and metrics:
                        Term,Definition,Source,Notes
                        donor_id,A unique identifier assigned to each donor.,Pledges, Payments,Used to track individual donors across pledges and payments.
                        pledge_id,A unique identifier for each pledge; a new pledge is created if a donor changes amount or frequency.,Pledges, Payments,Key for merging datasets; multiple pledges per donor are possible.
                        donor_chapter,The channel or organization where a donor first signed their pledge (e.g., university chapter).,Pledges,No difference between 'n/a' and empty cells; both indicate unknown.
                        chapter_type,Categories of `donor_chapter` (e.g., UG for Undergraduate).,Pledges,Helps group chapters by type for analysis.
                        pledge_status,Status of the pledge (e.g., Active donor, Pledged donor, Payment failure, Churned donor).,Pledges,Focus on 'Active' or 'Pledged' for key metrics; 'Payment failure' and 'Churned' indicate attrition.
                        pledge_created_at,The date and time when the pledge was created.,Pledges,Used for tracking pledge initiation.
                        pledge_starts_at,The date when the pledge payment schedule begins.,Pledges,Relevant for future pledge analysis.
                        pledge_ended_at,The date when the pledge payment schedule ends (if applicable).,Pledges,Indicates completed or cancelled pledges.
                        contribution_amount,The amount of money the donor pledged to contribute.,Pledges,Basis for Annualized Run Rate (ARR) calculations; may need USD conversion.
                        currency,The currency in which the pledged payments are to be made.,Pledges, Payments,Metrics often require conversion to USD for consistency.
                        frequency,The frequency of pledged payments (e.g., monthly, one-time).,Pledges,Affects ARR and payment scheduling.
                        payment_platform,The platform that processed the payment (e.g., Benevity, Donational).,Pledges, Payments,Useful for analyzing platform performance.
                        id,A unique identifier for each payment record.,Payments,Distinct from `pledge_id`; used for payment tracking.
                        portfolio,The allocation of the donation (e.g., OFTW Top Picks, Entire OFTW Portfolio).,Payments,Excludes 'Discretionary Fund' and 'Operating Costs' for Money Moved calculations.
                        amount,The amount of money donated in a payment.,Payments,Basis for Money Moved; multiplied by `counterfactuality` for impact assessment.
                        date,The date when the payment was made.,Payments,Used for YTD and time lag calculations.
                        counterfactual,A value between 0 and 1 indicating the likelihood that the donation wouldn’t have occurred without OFTW (0 = 0%, 1 = 100%).,Payments,Multiplied by `amount` to calculate counterfactual Money Moved.
                        Money Moved (YTD),Total amount of money moved year-to-date, adjusted by `counterfactuality`, for the fiscal year (July 1 to June 30).,KPIs,Current YTD is July 1, 2024, to March 09, 2025; excludes discretionary/operating costs.
                        Counterfactual MM,Money Moved multiplied by the `counterfactuality` value to reflect OFTW’s unique impact.,KPIs,Key impact metric; excludes certain portfolios.
                        Active Annualized Run Rate (ARR),Total monthly donation amount from active pledges, converted to USD.,KPIs,Based on `contribution_amount` for `Active donor` pledges.
                        Pledge Attrition Rate,Proportion of pledges with status 'Payment failure' or 'Churned donor' relative to all pledges.,KPIs,Indicates donor retention; helps target interventions.
                        Total Number of Active Donors,Count of unique `donor_id` with `pledge_status` 'Active donor' or 'one-time'.,KPIs,Tracks active donor base.
                        Total Number of Active Pledges,Count of unique `pledge_id` with `pledge_status` 'Active donor'.,KPIs,Measures current payment commitments.
                        Chapter ARR,ARR broken down by `donor_chapter` and `chapter_type`.,KPIs,Identifies high-performing chapters.
                        Fiscal Year,The OFTW financial year, running from July 1 to June 30.,KPIs,Current fiscal year is July 1, 2024, to June 30, 2025.

                        The dashboard contains the following charts:
                        Chart,Description
                        Active Annualized Run Rate by Top 10 Chapters,A horizontal bar chart displaying the total monthly donation amounts from active pledges (pledge_status 'Active donor') for the top 10 chapters by `contribution_amount`. The x-axis shows the Annualized Run Rate in USD, and the y-axis lists the chapters. It helps identify which chapters are contributing the most to recurring revenue.
                        Pledge Attrition Rate,A pie chart showing the proportion of pledges that have the status 'Payment failure' or 'Churned donor' compared to all pledges. It indicates donor retention challenges by visualizing the percentage of pledges that have failed or been discontinued versus those that remain active.
                        Time Lag Distribution (Days),A histogram illustrating the distribution of the time lag (in days) between the pledge creation date (`pledge_created_at`) and the payment date (`date`). It helps understand delays in donor payments, highlighting how long it typically takes for donors to fulfill their pledges after making them.

                        The dashboard also contains the following non-chart elements (do not describe these as charts):
                        Element,Description
                        Total Counterfactual Money Moved (YTD: July 1, 2024 - March 09, 2025),A text metric showing the total amount of money moved year-to-date, adjusted by `counterfactuality`, reflecting OFTW’s unique impact for the fiscal year from July 1, 2024, to March 09, 2025. It excludes donations to 'Discretionary Fund' and 'Operating Costs'. This is not a chart but a summary statistic.
                        Merged Data Sample,A table displaying the merged dataset, combining pledges and payments data on `pledge_id`. It includes columns like `donor_id`, `pledge_status`, `amount`, etc. This is not a chart but a data table for reference.

                        When asked to explain charts, use simple language as if explaining to a novice. For example, if asked 'explain each chart in this tool like I'm a novice,' list all three charts and describe them in easy terms. For other inquiries, provide clear and concise explanations about fields, metrics, or charts as requested.
                        """
                    },
                    {"role": "user", "content": query}
                ],
                max_tokens=500  # Increased to allow more detailed responses
            )
            logger.debug("AI response generated successfully")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in get_ai_response: {traceback.format_exc()}")
            return f"Error: Unable to process your request. Please try again later. Details: {str(e)}"

    # App layout (tabs + main container)
    app.layout = html.Div([
        # Dark Mode switch
        dbc.Row(
            dbc.Col(
                dbc.Switch(
                    id='dark-mode-switch',
                    label='Dark Mode',
                    value=False,
                    style={'margin': '10px'}
                ),
                width={'size': 2, 'offset': 10}
            )
        ),

        # Tabs
        dcc.Tabs([
            dcc.Tab(
                label='Dashboard',
                style={'padding': '0'},
                children=[
                    html.H1("OFTW Dashboard", id='dashboard-title', style={'margin': '0'}),
                    html.H3("Key Metrics", style={'textAlign': 'center', 'margin': '10px 0'}),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Total Counterfactual Money Moved (YTD: July 1, 2024 - March 09, 2025)",
                                    id='money-moved-title'),
                            html.P(f"${total_mm_ytd:,.2f}",
                                   style={'textAlign': 'center', 'fontSize': 24},
                                   id='money-moved-value')
                        ], width=4),
                        dbc.Col([
                            html.H4("Total Active Donors", id='active-donors-title'),
                            html.P(f"{active_donors:,}",
                                   style={'textAlign': 'center', 'fontSize': 24},
                                   id='active-donors-value')
                        ], width=4),
                        dbc.Col([
                            html.H4("Pledge Attrition Rate", id='attrition-rate-title'),
                            html.P(f"{attrition_rate:.2f}%",
                                   style={'textAlign': 'center', 'fontSize': 24},
                                   id='attrition-rate-value')
                        ], width=4),
                    ], style={'margin-bottom': '20px'}),

                    # Charts
                    dcc.Loading(id='loading-arr', children=[arr_chart], type='default'),
                    dcc.Loading(id='loading-grid', children=[
                        dbc.Row([
                            dbc.Col(attrition_chart, width=6),
                            dbc.Col(time_lag_chart, width=6)
                        ], style={'margin-top': '20px'})
                    ], type='default'),

                    # Table
                    html.H3("Merged Data Sample", id='table-title'),
                    dbc.Row([
                        dbc.Col([
                            # Give the label an ID
                            html.Label("Filter by Donor Chapter:", id='chapter-filter-label'),
                            dcc.Dropdown(
                                id='chapter-filter',
                                options=donor_chapters,
                                value='All',
                                style={'width': '100%'}
                            )
                        ], width=3),
                        dbc.Col([
                            # Give the label an ID
                            html.Label("Filter by Pledge Status:", id='status-filter-label'),
                            dcc.Dropdown(
                                id='status-filter',
                                options=pledge_statuses,
                                value='All',
                                style={'width': '100%'}
                            )
                        ], width=3),
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col(dcc.Download(id="download-data-csv"), width=2),
                        dbc.Col(html.Button("Export to CSV", id="btn-csv", n_clicks=0), width=2)
                    ]),
                    dcc.Loading(id='loading-table', children=[data_table], type='default')
                ]
            ),
            dcc.Tab(
                label='Glossary',
                style={'padding': '0'},
                children=[
                    html.H2("OFTW Data Glossary",
                            id='glossary-title',
                            style={'textAlign': 'center', 'margin': '10px 0'}),
                    html.P("This glossary defines key terms and metrics used in the OFTW dashboard.",
                           id='glossary-description',
                           style={'textAlign': 'center', 'margin': '0'}),
                    html.Div(
                        glossary,
                        id='glossary-table-container',
                        style={'padding': '0', 'margin': '0'}
                    )
                ]
            ),
            dcc.Tab(
                label='AI Assistant',
                style={'padding': '0'},
                children=[
                    html.H2("OFTW AI Assistant",
                            id='ai-title',
                            style={'textAlign': 'center', 'margin': '10px 0'}),
                    html.P("Ask questions about data fields, metrics, or charts. For example, 'Explain the Time Lag Distribution chart'.",
                           id='ai-description',
                           style={'textAlign': 'center', 'margin': '0'}),
                    dcc.Input(
                        id='ai-query-input',
                        type='text',
                        placeholder='Enter your question...',
                        style={'width': '80%', 'margin': '10px'}
                    ),
                    html.Button('Submit', id='ai-submit-button', n_clicks=0, style={'margin': '10px'}),
                    html.Div(
                        id='ai-response',
                        style={'margin': '10px', 'whiteSpace': 'pre-wrap', 'minHeight': '100px'}
                    )
                ]
            )
        ])
    ],
    id='main-container',
    style={'minHeight': '100vh', 'margin': 0, 'padding': 0, 'backgroundColor': '#ECF0F1'}
    )

    # Dark mode callback
    @app.callback(
        # 24 outputs total (22 old + 2 for label styles)
        Output('main-container', 'style'),
        Output('dashboard-title', 'style'),
        Output('money-moved-title', 'style'),
        Output('money-moved-value', 'style'),
        Output('active-donors-title', 'style'),
        Output('active-donors-value', 'style'),
        Output('attrition-rate-title', 'style'),
        Output('attrition-rate-value', 'style'),
        Output('table-title', 'style'),
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
        Output('status-filter-label', 'style'),
        Input('dark-mode-switch', 'value')
    )
    def update_dark_mode(dark_mode):
        if dark_mode:
            background_color  = '#1E1E1E'
            text_color        = '#CCCCCC'
            chart_paper_color = '#2E2E2E'
            chart_text_color  = '#CCCCCC'
            table_header_bg   = '#333333'
            table_data_bg     = '#2E2E2E'
            table_text_color  = '#CCCCCC'
            input_bg          = '#2E2E2E'
            input_text        = '#FFFFFF'
            bar_color         = '#3399FF'
            table_container_bg= '#2E2E2E'
            response_bg       = '#2E2E2E'
            table_bg          = '#2E2E2E'
        else:
            background_color  = '#ECF0F1'
            text_color        = '#2C3E50'
            chart_paper_color = '#FFFFFF'
            chart_text_color  = '#2C3E50'
            table_header_bg   = 'rgb(230, 230, 230)'
            table_data_bg     = 'rgb(255, 255, 255)'
            table_text_color  = '#2C3E50'
            input_bg          = '#FFFFFF'
            input_text        = '#2C3E50'
            bar_color         = '#1f77b4'
            table_container_bg= '#FFFFFF'
            response_bg       = '#FFFFFF'
            table_bg          = '#FFFFFF'

        # Figures
        arr_fig = px.bar(
            arr_data,
            x='contribution_amount', y='donor_chapter',
            title='Active Annualized Run Rate by Top 10 Chapters',
            labels={'contribution_amount': 'Annualized Run Rate (USD)', 'donor_chapter': 'Chapter'},
            orientation='h',
            height=600
        ).update_traces(marker_color=bar_color).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=chart_paper_color,
            font_color=chart_text_color,
            xaxis={'title': 'Annualized Run Rate (USD)', 'gridcolor': 'rgba(255,255,255,0.1)'},
            yaxis={'tickangle': 0, 'automargin': True, 'gridcolor': 'rgba(255,255,255,0.1)'},
            margin=dict(l=200, r=50, t=50, b=50)
        )

        attrition_fig = px.pie(
            values=[len(merged_df) - len(attrition_pledges), len(attrition_pledges)],
            names=['Active', 'Attrition'],
            title='Pledge Attrition Rate',
            color_discrete_sequence=['#3399FF', '#FF6F61'] if dark_mode else ['#4682B4', '#FF6F61']
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=chart_paper_color,
            font_color=chart_text_color
        )

        time_lag_fig = px.histogram(
            merged_df.dropna(subset=['date', 'pledge_created_at']),
            x=(merged_df['date'] - merged_df['pledge_created_at']).dt.days,
            nbins=30,
            title='Time Lag Distribution (Days)',
            color_discrete_sequence=[bar_color]
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=chart_paper_color,
            font_color=chart_text_color,
            xaxis={'gridcolor': 'rgba(255,255,255,0.1)'},
            yaxis={'gridcolor': 'rgba(255,255,255,0.1)'}
        )

        # Return all 24 outputs
        return (
            # 1. main-container.style
            {'minHeight': '100vh', 'margin': 0, 'padding': 0, 'backgroundColor': background_color},
            # 2. dashboard-title.style
            {'textAlign': 'center', 'color': text_color},
            # 3. money-moved-title.style
            {'textAlign': 'center', 'color': text_color},
            # 4. money-moved-value.style
            {'textAlign': 'center', 'fontSize': 24, 'color': text_color},
            # 5. active-donors-title.style
            {'textAlign': 'center', 'color': text_color},
            # 6. active-donors-value.style
            {'textAlign': 'center', 'fontSize': 24, 'color': text_color},
            # 7. attrition-rate-title.style
            {'textAlign': 'center', 'color': text_color},
            # 8. attrition-rate-value.style
            {'textAlign': 'center', 'fontSize': 24, 'color': text_color},
            # 9. table-title.style
            {'color': text_color},
            # 10. arr-chart.figure
            arr_fig,
            # 11. attrition-chart.figure
            attrition_fig,
            # 12. time-lag-chart.figure
            time_lag_fig,
            # 13. merged-data-table.style_header
            {'backgroundColor': table_header_bg, 'color': table_text_color, 'fontWeight': 'bold'},
            # 14. merged-data-table.style_data
            {'backgroundColor': table_data_bg, 'color': table_text_color},
            # 15. ai-query-input.style
            {'width': '80%', 'margin': '10px', 'color': input_text, 'backgroundColor': input_bg},
            # 16. ai-response.style
            {'margin': '10px', 'whiteSpace': 'pre-wrap', 'color': text_color,
             'backgroundColor': response_bg, 'minHeight': '100px'},
            # 17. glossary-table-container.style
            {'backgroundColor': table_container_bg, 'padding': '0', 'margin': '0', 'color': text_color},
            # 18. glossary-table.style
            {'width': '100%', 'border': '1px solid #ddd', 'margin': '20px 0',
             'border-collapse': 'collapse', 'backgroundColor': table_bg, 'color': text_color},
            # 19. glossary-title.style
            {'textAlign': 'center', 'margin': '10px 0', 'color': text_color},
            # 20. glossary-description.style
            {'textAlign': 'center', 'margin': '0', 'color': text_color},
            # 21. ai-title.style
            {'textAlign': 'center', 'margin': '10px 0', 'color': text_color},
            # 22. ai-description.style
            {'textAlign': 'center', 'margin': '0', 'color': text_color},
            # 23. chapter-filter-label.style
            {'color': text_color},
            # 24. status-filter-label.style
            {'color': text_color}
        )

    # Table filtering callback
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

    # Export CSV callback
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

    # AI response callback
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

# Run the app
if __name__ == '__main__':
    try:
        app.run_server(debug=True)
    except Exception as e:
        logger.error(f"Server error: {traceback.format_exc()}")
        raise