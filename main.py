import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import skfolio as sk
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from datetime import datetime
from sklearn.model_selection import train_test_split
from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.optimization import MeanRisk, InverseVolatility, ObjectiveFunction, HierarchicalRiskParity
from skfolio import Population, RiskMeasure
from skfolio.preprocessing import prices_to_returns
from skfolio.optimization import InverseVolatility, Random, EqualWeighted

import streamlit as st

# CSS pour centrer le titre
st.markdown("""
<style>
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Votre titre centré


# Créez des colonnes avec la première colonne plus petite si nécessaire
col1, col2 = st.columns([1, 4])

# Dans la première colonne, mettez votre logo
with col1:
    logo_path = 'https://project.intellcap.fr/wp-content/uploads/2023/11/IMG_1422-768x641.jpeg'  # Remplacez par le chemin vers votre image
    st.image(logo_path, width=150)  # Ajustez la largeur selon la taille souhaitée pour votre logo


# Streamlit app layout
st.title('Portfolio Optimization Tool')

# Number of assets
num_assets = 10


# Main section for selecting optimization goal
optimization_goal = st.selectbox(
    "Choose an optimization goal",
    ["Mean-Risk with Minimization of Risk", "Mean-Risk with Sharpe Ratio Maximization",
     "Mean-Risk with Utility Maximization", "Hierarchical Risk Parity"]
)

risk_aversion = st.slider('Risk Aversion (Applicable in case of Utility Maximization)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# Asset Constraints
asset_constraints = st.checkbox('Asset Constraints')

# Group Constraints
group_constraints = st.checkbox('Group Constraints')


# Create a function to display the asset input table
def display_asset_table():
    st.subheader("Portfolio Assets")
    cols = st.columns([2, 2, 2, 2]) if asset_constraints and group_constraints else st.columns(
        [2, 2, 2]) if asset_constraints or group_constraints else st.columns([2, 2])

    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'XOM', 'TSLA']

    with cols[0]:
        st.write("Ticker symbol")
    if asset_constraints:
        with cols[1]:
            st.write("Min. Weight")
        with cols[2]:
            st.write("Max. Weight")
    if group_constraints:
        with cols[-1]:
            st.write("Group")

    asset_info = []
    for i in range(num_assets):
        cols = st.columns([2, 2, 2, 2]) if asset_constraints and group_constraints else st.columns(
            [2, 2, 2]) if asset_constraints or group_constraints else st.columns([2, 2])

        with cols[0]:
            default_ticker = default_tickers[i] if i < len(default_tickers) else ''
            ticker = st.text_input(f'Ticker symbol {i + 1}', default_ticker)
        if asset_constraints:
            with cols[1]:
                min_weight = st.number_input(f'Min. Weight {i + 1} (%)', min_value=0.0, max_value=100.0, value=0.0,
                                             step=0.01, format="%.2f")
            with cols[2]:
                max_weight = st.number_input(f'Max. Weight {i + 1} (%)', min_value=0.0, max_value=100.0, value=100.0,
                                             step=0.01, format="%.2f")
        else:
            min_weight, max_weight = 0.0, 100.0
        if group_constraints:
            with cols[-1]:
                group = st.selectbox(f'Group {i + 1}', ['None', 'A', 'B', 'C', 'D', 'E', 'F'], index=0)
        else:
            group = 'None'
        asset_info.append((ticker, min_weight, max_weight, group))
    return asset_info




# Create a function to display the group constraints table
def display_group_table():
    st.subheader("Asset Groups")
    cols = st.columns(4)
    with cols[0]:
        st.write("Group")
    with cols[1]:
        st.write("Name (optional)")
    with cols[2]:
        st.write("Min. Weight")
    with cols[3]:
        st.write("Max. Weight")

    group_info = []
    for group in ['A', 'B', 'C', 'D', 'E', 'F']:
        cols = st.columns(4)
        with cols[0]:
            st.write(group)
        with cols[1]:
            group_name = st.text_input(f'Group {group} name (optional)')
        with cols[2]:
            min_group_weight = st.number_input(f'Group {group} Min. Weight (%)', min_value=0.0, max_value=100.0,
                                               value=0.0, step=0.01, format="%.2f")
        with cols[3]:
            max_group_weight = st.number_input(f'Group {group} Max. Weight (%)', min_value=0.0, max_value=100.0,
                                               value=100.0, step=0.01, format="%.2f")
        group_info.append((group, group_name, min_group_weight, max_group_weight))
    return group_info


# Collect asset information
asset_info = display_asset_table()

if group_constraints:
    group_info = display_group_table()
else:
    group_info = None


# Main section for selecting benchmark type
benchmark_type = st.selectbox(
    "Choose a benchmark",
    ["EqualWeighted", "InverseVolatility", "Random"]
)

# User input for date range
start_date = st.date_input('Start Date', pd.to_datetime('2016-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2019-12-30'))

# Explain test_size parameter to the user
st.info("The test size parameter determines the proportion of the dataset to include in the test split. "
        "A value of 0.33 means that 33% of the data will be used for testing, while the remaining 67% will be used for training. "
        "The training phase is where the model estimates the optimal weights for the portfolio. "
        "The test phase is used to evaluate the performance of the portfolio using the estimated weights on unseen data.")

# User input for test size
test_size = st.slider('Select the test size for splitting the data:', min_value=0.1, max_value=0.5, value=0.33,
                      step=0.01)

if st.button('Optimize Portfolio'):
    # Extract tickers and filter out empty entries
    assets = [info[0] for info in asset_info if info[0]]

    # Downloading data
    data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
    # Ensure data is a DataFrame and handle missing data
    data = data.dropna()

    # Calculating returns
    returns = data.pct_change().dropna()

    # Ensure returns is a DataFrame
    if not isinstance(returns, pd.DataFrame):
        st.error('Returns calculation did not result in a DataFrame.')
        st.stop()

    # Transforming prices to returns and splitting the dataset
    X = prices_to_returns(data)
    X_train, X_test = train_test_split(X, test_size=test_size, shuffle=False)

    # Collect weight constraints from user input
    min_weights = [info[1] / 100 for info in asset_info if info[0]]
    max_weights = [info[2] / 100 for info in asset_info if info[0]]

    if optimization_goal == "Mean-Risk with Minimization of Risk":
        objective_function = ObjectiveFunction.MINIMIZE_RISK
    elif optimization_goal == "Mean-Risk with Sharpe Ratio Maximization":
        objective_function = ObjectiveFunction.MAXIMIZE_RATIO
    elif optimization_goal == "Mean-Risk with Utility Maximization":
        objective_function = ObjectiveFunction.MAXIMIZE_UTILITY
    else:
        objective_function = None  # Not used for HRP

    if optimization_goal in ["Mean-Risk with Minimization of Risk", "Mean-Risk with Sharpe Ratio Maximization",
                             "Mean-Risk with Utility Maximization"]:
        # Creating and fitting the MeanRisk model with constraints
        model = MeanRisk(
            risk_measure=RiskMeasure.VARIANCE,
            objective_function=objective_function,
            portfolio_params=dict(
                name="Max Sharpe" if objective_function == ObjectiveFunction.MAXIMIZE_RATIO else "Min Risk"),
            min_weights=min_weights,
            max_weights=max_weights,
            risk_aversion=risk_aversion if optimization_goal == "Mean-Risk with Utility Maximization" else None
        )
        model.fit(X_train)

        st.write("Model Weights with Asset Names:")
        weights_df = pd.DataFrame({
            'Asset': model.feature_names_in_,
            'Weight': model.weights_
        })
        st.write(weights_df)

        # Creating and fitting the MeanRisk model with efficient frontier
        model_fr = MeanRisk(
            risk_measure=RiskMeasure.VARIANCE,
            efficient_frontier_size=30,
            portfolio_params=dict(name="Variance")
        )
        model_fr.fit(X_train)

        # Predicting on the test set and training set
        population_train_fr = model_fr.predict(X_train)
        population_test_fr = model_fr.predict(X_test)
        population_train_fr.set_portfolio_params(tag="Train")
        population_test_fr.set_portfolio_params(tag="Test")

        # Creating a population for analysis and plotting
        population_fr = population_train_fr + population_test_fr

        # Create benchmark based on selection
        if benchmark_type == "EqualWeighted":
            benchmark = EqualWeighted()
        elif benchmark_type == "InverseVolatility":
            benchmark = InverseVolatility()
        elif benchmark_type == "Random":
            benchmark = Random()

        benchmark.fit(X_train)

        # Predicting on the test set
        pred_model = model.predict(X_test)
        pred_bench = benchmark.predict(X_test)

        # Creating a population for analysis and plotting
        population = Population([pred_model, pred_bench])

        st.write("Portfolio Compositions")
        fig1 = population.plot_composition()
        st.plotly_chart(fig1)

        fig2 = population.plot_cumulative_returns()
        st.plotly_chart(fig2)
        st.write('Efficient Frontier')
        # Plotting efficient frontier
        fig3 = population_fr.plot_measures(
            x=RiskMeasure.ANNUALIZED_VARIANCE,
            y=PerfMeasure.ANNUALIZED_MEAN,
            color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
            hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
        )
        st.plotly_chart(fig3)

        st.write("Risk and Return Metrics")
        st.write(population.summary())

        # Extract the covariance matrix
        covariance_matrix = model.prior_estimator_.covariance_estimator_.covariance_

        # Plot the covariance matrix
        fig_cov = go.Figure(data=go.Heatmap(
            z=covariance_matrix,
            x=model.feature_names_in_,
            y=model.feature_names_in_,
            colorscale='Viridis'
        ))

        fig_cov.update_layout(
            title='Covariance Matrix',
            xaxis_nticks=36
        )
        st.write("Estimated Asset Covariance")
        st.plotly_chart(fig_cov)

        def plot_rolling_measures(population, measure):
            """
            Creates a Plotly figure of the rolling measure (e.g., Sharpe Ratio) for each portfolio in the population.

            Parameters:
            - population: Population object containing a list of Portfolio objects.
            - measure: The rolling measure to be calculated (e.g., RatioMeasure.SHARPE_RATIO).

            Returns:
            - A Plotly figure displaying the rolling measure for each portfolio.
            """
            rolling_measures = []

            # Calculate rolling measure for each portfolio in the population
            for i, portfolio in enumerate(population):
                rolling_measure = portfolio.rolling_measure(measure=measure)
                rolling_measures.append(rolling_measure)

            # Combine the results into a single DataFrame
            combined_df = pd.DataFrame({
                f'Portfolio {i}': rm for i, rm in enumerate(rolling_measures)
            })

            # Create a Plotly figure
            fig = go.Figure()

            for column in combined_df.columns:
                fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[column], mode='lines', name=column))

            fig.update_layout(
                title=f'Rolling {measure} for Portfolios',
                xaxis_title='Date',
                yaxis_title=f'Rolling {measure}',
                legend_title='Portfolio',
                template='plotly_white'
            )

            return fig


        # Example usage
        fig8 = plot_rolling_measures(population, RatioMeasure.SHARPE_RATIO)
        st.plotly_chart(fig8)

    elif optimization_goal == "Hierarchical Risk Parity":
        # Creating and fitting the HierarchicalRiskParity model
        model1 = HierarchicalRiskParity(
            risk_measure=RiskMeasure.CVAR,
            portfolio_params=dict(name="HRP-CVaR-Ward-Pearson")
        )
        model1.fit(X_train)

        st.write("Model Weights with Asset Names:")
        weights_df1 = pd.DataFrame({
            'Asset': model1.feature_names_in_,
            'Weight': model1.weights_
        })
        st.write(weights_df1)

        # Predicting on the training set
        ptf1 = model1.predict(X_train)

        # Create benchmark based on selection
        if benchmark_type == "EqualWeighted":
            benchmark = EqualWeighted()
        elif benchmark_type == "InverseVolatility":
            benchmark = InverseVolatility()
        elif benchmark_type == "Random":
            benchmark = Random()

        benchmark.fit(X_train)

        # Predicting on the test set
        pred_model_hrp = model1.predict(X_test)
        pred_bench = benchmark.predict(X_test)

        # Creating a population for analysis and plotting
        population_hrp = Population([pred_model_hrp, pred_bench])

        st.write("Portfolio Compositions")
        fig4 = population_hrp.plot_composition()
        st.plotly_chart(fig4)

        fig5 = population_hrp.plot_cumulative_returns()
        st.plotly_chart(fig5)

        st.write("Risk and Return Metrics")
        st.write(population_hrp.summary())

        st.write("Contribution to CVaR")
        fig6 = ptf1.plot_contribution(measure=RiskMeasure.CVAR)
        st.plotly_chart(fig6)

        st.write("Hierarchical Clustering Dendrogram")
        fig7 = model1.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=False)
        st.plotly_chart(fig7)



