
# coding: utf-8
# # Choropleth Maps Exercise 
# Pradeep K. Pant
# [Full Documentation Reference](https://plot.ly/python/reference/#choropleth)

# ## Plotly Imports
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True) 
# Q1: World Power Consumption 2014
# Basic preparation
# Import pandas and read the csv file: 2014_World_Power_Consumption
import pandas as pd
df = pd.read_csv('2014_World_Power_Consumption')
# Check the head of the DataFrame. 
df.head()
# We need to create data and layout variable which contains a dict
data = dict(type='choropleth',
                locations = df['Country'],
                locationmode = 'country names',
                z = df['Power Consumption KWH'],
                text = df['Country'],
                colorbar = {'title':'Power Consumption KWH'},
                colorscale = 'Viridis',
                reversescale = True
                )

# Lets make a layout
layout = dict(title='2014 World Power Consumption',
geo = dict(showframe=False,projection={'type':'Mercator'}))
# Pass the data and layout and plot using iplot
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)

# Q2: USA Choropleth

# Import the 2012_Election_Data csv file using pandas.
usadf = pd.read_csv('2012_Election_Data')
# Check the head of the DataFrame.
usadf.head()
# Now create a plot that displays the Voting-Age Population (VAP) per state. 
# First make data dict
data = dict(type='choropleth',
            locations=usadf['State Abv'],
            locationmode = 'USA-states',
            z = usadf['Voting-Age Population (VAP)'],
            text = usadf['State'],
            colorbar = {'title':'Voting Age Polulation (VAP)'},
            colorscale = 'Viridis',
            reversescale = True)

# Make a nice layout to show all the USA states
layout = dict(title='2012 US Elections: Voting Age Population',
geo = dict(scope='usa', showlakes=True, lakecolor='rgb(85,173.240)'))
# Finally make plot using data and layout
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)

# END