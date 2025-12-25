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
st.caption(
    'This app predicts **monthly average temperature** for a selected city '
    'based on long-term historical climate data.'
)

# load model
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

# sidebar 
st.sidebar.markdown('### Location')

country = st.sidebar.selectbox('Country',sorted(data['Country'].unique()),help='Select a country')

country_data = data[data['Country'] == country]

city = st.sidebar.selectbox('City',sorted(country_data['City'].unique()),help='Select a city')

city_data = country_data[country_data['City'] == city]

# City data
city_row = city_data.iloc[-1]
lat = convert_lat_lon(city_row['Latitude'])
lon = convert_lat_lon(city_row['Longitude'])
uncertainty = city_row['AverageTemperatureUncertainty']

# Map
st.subheader('Selected City')

map_data = pd.DataFrame({'lat': [lat], 'lon': [lon], 'city': [city]})

layer = pdk.Layer(
    'ScatterplotLayer',
    data=map_data,
    get_position='[lon, lat]',
    get_radius=50000,
    get_fill_color=[255, 0, 0, 180],
    pickable=True
)

view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=5)

st.pydeck_chart(
    pdk.Deck(layers=[layer],initial_view_state=view_state,tooltip={'text':'{city}'}
    )
)


# time selection
st.markdown('### Time Selection')

month = st.slider('Month',1, 12, 7,help='1 = January, 12 = December')

year = st.selectbox('Year',list(range(1850, 2101)),index=list(range(1850, 2101)).index(2030),help='Select a year for prediction')

st.info(
    f'Selected city: {city}, {country}  \n'
    f'Selected time: Month {month}, Year {year}'
)


# prediction
if st.button('Predict Temperature'):

    input_data = pd.DataFrame([[year, month, lat, lon, uncertainty]],columns=feature_columns)

    with st.spinner('Calculating temperature...'):
        prediction = model.predict(input_data)[0]

    st.metric(
        label='Predicted Monthly Average Temperature',
        value=f'{prediction:.2f} °C'
    )

    st.caption(
        'Prediction is based on historical monthly averages, '
        'not real-time weather data.'
    )

   # Charts 
    with st.expander('View Historical & Seasonal Charts'):

        # Historical trend
        st.subheader('Historical Temperature Trend')

        yearly_avg = (
            city_data.groupby('Year')['AverageTemperature']
            .mean()
            .reset_index()
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=yearly_avg['Year'],
                y=yearly_avg['AverageTemperature'],
                mode='lines',
                line=dict(color='royalblue', width=3),
                name='Historical Average Temperature'
            )
        )

        fig_trend.update_layout(
            xaxis_title='Year',
            yaxis_title='Average Monthly Temperature (°C)',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig_trend, use_container_width=True)

        # Seasonal pattern
        st.subheader('Seasonal Temperature Pattern')

        month_names = ['January','February','March','April','May','June','July','August','September','October','November','December']

        monthly_avg = (city_data.groupby('Month')['AverageTemperature'].mean().reset_index())

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
            yaxis_title='Average Monthly Temperature (°C)',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig_season, use_container_width=True)

#footer 
st.markdown('---')
st.caption(
    'Climate Trend Prediction |  Built with ML & Streamlit | created by Ruchit'
)
