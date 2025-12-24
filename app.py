import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import plotly.graph_objects as go

def convert_lat_lon(value):
    value = str(value).strip()
    if value[-1] in ['N', 'E']:
        return float(value[:-1])
    elif value[-1] in ['S', 'W']:
        return -float(value[:-1])
    else:
        return float(value)


st.set_page_config(
    page_title='Climate Trend Predictor',
    layout='centered'
)

st.title('Climate Trend Predictor')
st.write('Predict **monthly average temperature** using historical climate data')

# Model
@st.cache_resource
def load_model():
    model = joblib.load('temperature_model.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

model, feature_columns = load_model()

# load dataset 
@st.cache_data
def load_data():
    df = pd.read_csv('GlobalLandTemperaturesByCity.csv')
    df = df.dropna(subset=['AverageTemperature'])
    df['dt'] = pd.to_datetime(df['dt'])
    df['Year'] = df['dt'].dt.year
    df['Month'] = df['dt'].dt.month
    return df

data = load_data()

# Sidebar 
st.sidebar.header('Select Location')

country = st.sidebar.selectbox('Country',sorted(data['Country'].unique()))
country_df = data[data['Country'] == country]

city = st.sidebar.selectbox('City',sorted(country_df['City'].unique()))
city_data = country_df[country_df['City']== city]

# City data 
city_row = city_data.iloc[-1]
lat =convert_lat_lon(city_row['Latitude'])
lon = convert_lat_lon(city_row['Longitude'])
uncertainty =city_row['AverageTemperatureUncertainty']

# Map
st.subheader('Selected City')

map = pd.DataFrame({'lat':[lat],'lon':[lon],'city':[city]})

layer = pdk.Layer(
    'ScatterplotLayer',data=map,get_position='[lon, lat]',get_radius=50000,get_fill_color=[255,0,0,180],pickable=True
)

view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=5)

st.pydeck_chart(
    pdk.Deck(layers=[layer],initial_view_state=view_state,tooltip={'text':'{city}'}
    )
)

#time selection 
st.subheader('Select Prediction Time')

month_names = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']
year = st.slider('Year',1850,2100,2030)
month_name = st.selectbox('Month',month_names)
month = month_names.index(month_name) + 1


# After pradiction 
if st.button('Predict Temperature'):

    input_data = pd.DataFrame([[year, month, lat, lon, uncertainty]],columns=feature_columns)

    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Monthly Average Temperature: **{prediction:.2f}°C**')
    st.caption(f'{city}, {country} | {month}, {year}')

    # Historical graph 

    st.subheader('Historical Temperature Trend')

    yearly_avg = (city_data.groupby('Year')['AverageTemperature'].mean().reset_index())

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(x=yearly_avg['Year'],y=yearly_avg['AverageTemperature'],mode='lines',
            line=dict(color='royalblue', width=3),
            name='Historical Average Temperature'
        )
    )

    fig_trend.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Monthly Temperature',
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    #seasonal graph
    st.subheader('Seasonal Temperature Pattern')

    monthly_avg = (city_data.groupby('Month')['AverageTemperature'].mean().reset_index() )

    monthly_avg['MonthName'] = monthly_avg['Month'].apply(lambda x: month_names[x - 1])

    fig_season = go.Figure()
    fig_season.add_trace(
        go.Scatter(
            x=monthly_avg['MonthName'],
            y=monthly_avg['AverageTemperature'],
            mode='lines+markers',
            line=dict(color='orange', width=3),
            marker=dict(size=8),
            name='Seasonal Average Temperature'
        )
    )

    fig_season.update_layout(
        xaxis_title='Month',
        yaxis_title='Average Monthly Temperature',
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig_season,use_container_width=True)

    st.info(
        'The map updates based on your selection. '
        'Predictions are based on historical climate trends, not daily weather.'
    )
# footer 
st.markdown('---')
st.caption('Climate Trend Prediction | Built with ML and Streamlit | Created by Ruchit')
