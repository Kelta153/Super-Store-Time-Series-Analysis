import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")

# Custom CSS to center elements
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .center-table {
        width: 80%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
dash_1 = st.container()
with dash_1:
    st.markdown("<h2 style='text-align: center;'>Superstore Sales Dashboard</h2>", unsafe_allow_html=True)
    st.write("Welcome to the Sales Analytics Hub. This application allows you to predict future sales for Furniture, Office Supplies, and Technology categories. Please enter the number of months for the sales prediction below and click on 'Predict' to see the forecasted sales.")

# Load models
with open('fb_furniture_model.pkl','rb') as file:
    fb_furniture_model = joblib.load(file) 

with open('sarimax_office_model.pkl','rb') as file:
    sarimax_office_model = joblib.load(file)

with open('fb_tech_model.pkl','rb') as file:
    fb_tech_model = joblib.load(file)

# User input for number of months
num_of_months = st.number_input('Enter Number of Months of Sales Prediction', min_value=1, step=1)

# Prediction button
if st.button("Predict"):
    # Load datasets
    monthly_furniture_df = pd.read_csv('furniture_data.csv')
    monthly_office_supplies_df = pd.read_csv('office_supplies_data.csv')
    monthly_technology_df = pd.read_csv('technology_data.csv')
    
    # Convert date column to datetime and set as index
    monthly_furniture_df['Order Date'] = pd.to_datetime(monthly_furniture_df['Order Date'])
    monthly_office_supplies_df['Order Date'] = pd.to_datetime(monthly_office_supplies_df['Order Date'])
    monthly_technology_df['Order Date'] = pd.to_datetime(monthly_technology_df['Order Date'])
    
    monthly_furniture_df.set_index('Order Date', inplace=True)
    monthly_office_supplies_df.set_index('Order Date', inplace=True)
    monthly_technology_df.set_index('Order Date', inplace=True)
    
    # Find the last date in the datasets
    last_date_furniture = monthly_furniture_df.index[-1]
    last_date_office = monthly_office_supplies_df.index[-1]
    last_date_tech = monthly_technology_df.index[-1]
    
    # Increment the last date by one month to get the start date for forecasting
    start_date_furniture = last_date_furniture + pd.DateOffset(months=1)
    start_date_office = last_date_office + pd.DateOffset(months=1)
    start_date_tech = last_date_tech + pd.DateOffset(months=1)
    
    # Create DataFrames with future dates
    future_furniture = pd.DataFrame({'ds': pd.date_range(start=start_date_furniture, periods=num_of_months, freq='MS')})
    future_tech = pd.DataFrame({'ds': pd.date_range(start=start_date_tech, periods=num_of_months, freq='MS')})
    forecast_end_date_office = start_date_office + pd.DateOffset(months=num_of_months-1)

    # Generate forecasts
    furniture_pred = fb_furniture_model.predict(future_furniture)
    office_pred = sarimax_office_model.predict(start=start_date_office, end=forecast_end_date_office)
    technology_pred = fb_tech_model.predict(future_tech)
    
    # Rename columns and set index
    furniture_pred = furniture_pred.rename(columns = {'ds': 'Order Date'})
    technology_pred = technology_pred.rename(columns = {'ds': 'Order Date'})
    furniture_pred.set_index('Order Date', inplace=True)
    technology_pred.set_index('Order Date', inplace=True)
    furniture_pred['Furniture'] = furniture_pred['yhat']
    
    # Combine predictions
    sales_prediction = furniture_pred['Furniture']
    combined_predictions = pd.concat([sales_prediction, office_pred, technology_pred['yhat']], axis=1)
    combined_predictions.columns = ['Furniture', 'Office', 'Technology']

    # Display predictions in a centered format with a pie chart beside it
    dash_2 = st.container()
    with dash_2:
        col_center, col_pie = st.columns([2, 4])
        with col_center:
            st.markdown("<h3>Predicted Sales</h3>", unsafe_allow_html=True)
            st.markdown("<div class='center'><div class='center-table'>", unsafe_allow_html=True)
            st.dataframe(combined_predictions)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with col_pie:
            total_sales = combined_predictions.sum()
            sales_distribution = pd.DataFrame({'Category': total_sales.index, 'Sales': total_sales.values})
            fig_pie = px.pie(sales_distribution, values='Sales', names='Category', title='Sales Distribution by Category')
            st.plotly_chart(fig_pie)

    # Container for bar graph and line graph side by side
    dash_3 = st.container()
    with dash_3:
        col1, col2 = st.columns(2)
        with col1:
            # Line graph for forecasted store sales
            fig_line = px.line(combined_predictions, 
                               x=combined_predictions.index, 
                               y=['Furniture', 'Office', 'Technology'],
                               labels={'value': 'Sales', 'Order Date': 'Order Date'},
                               title='Forecasted Store Sales')
            st.plotly_chart(fig_line)
        
        with col2:
            # Bar graph for sales comparison by category
            fig_bar = px.bar(sales_distribution, x='Category', y='Sales', title='Sales Comparison by Category')
            st.plotly_chart(fig_bar)

    # Container for scatter plots side by side
    dash_4 = st.container()
    with dash_4:
        col3, col4 = st.columns(2)
        with col3:
            fig_scatter_furniture = px.scatter(combined_predictions.reset_index(), 
                                               x='index', y='Furniture', 
                                               title='Furniture Sales Trends', 
                                               labels={'index': 'Order Date', 'Furniture': 'Sales'})
            st.plotly_chart(fig_scatter_furniture)
        
        with col4:
            fig_scatter_office = px.scatter(combined_predictions.reset_index(), 
                                            x='index', y='Office', 
                                            title='Office Supplies Sales Trends', 
                                            labels={'index': 'Order Date', 'Office': 'Sales'})
            st.plotly_chart(fig_scatter_office)
    
    fig_scatter_technology = px.scatter(combined_predictions.reset_index(), 
                                        x='index', y='Technology', 
                                        title='Technology Sales Trends', 
                                        labels={'index': 'Order Date', 'Technology': 'Sales'})
    st.plotly_chart(fig_scatter_technology)

    # Container for highest and lowest sales month per category
    dash_5 = st.container()
    with dash_5:
        st.markdown("<h3 style='text-align: center;'>Highest and Lowest Sales Month</h3>", unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        
        def create_bar_chart(df, category):
            max_month = df[category].idxmax()
            min_month = df[category].idxmin()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[max_month, min_month],
                y=[df.loc[max_month, category], df.loc[min_month, category]],
                text=[df.loc[max_month, category], df.loc[min_month, category]],
                textposition='auto',
                name=category
            ))
            fig.update_layout(
                title=f'Highest and Lowest Sales Month for {category}',
                xaxis_title='Month',
                yaxis_title='Sales'
            )
            return fig

        with col5:
            st.plotly_chart(create_bar_chart(combined_predictions, 'Furniture'))
        
        with col6:
            st.plotly_chart(create_bar_chart(combined_predictions, 'Office'))
        
        with col7:
            st.plotly_chart(create_bar_chart(combined_predictions, 'Technology'))

    # Additional Information section
    with st.expander("View Data Sources"):
        col8, col9, col10 = st.columns(3)

        with col8:
            st.write("### Furniture Data")
            st.dataframe(monthly_furniture_df)
    
        with col9:
            st.write("### Office Supplies Data")
            st.dataframe(monthly_office_supplies_df)
    
        with col10:
            st.write("### Technology Data")
            st.dataframe(monthly_technology_df)
