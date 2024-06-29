import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# Show the page title and description.
st.set_page_config(page_title="Movies dataset", page_icon="🎬")
st.title("🎬 Movies dataset")
st.write(
    """
    This app visualizes data from [The Movie Database (TMDB)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
    It shows which movie genre performed best at the box office over the years. Just 
    click on the widgets below to explore!
    """
)

# Load the movie data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache
def load_movie_data():
    df = pd.read_csv("data/data/movies_genres_summary3.csv")
    return df

movie_df = load_movie_data()

# Load the species data from a CSV.
@st.cache
def load_species_data():
    df = pd.read_csv("data/species_strategies.csv")
    return df

species_df = load_species_data()

# Show a multiselect widget with the genres using `st.multiselect`.

genres = st.multiselect("Genres",["Romance", "Film-Noir", "Music", "Comedy", "Biography", "Sport", "Drama", "Animation", "Sci-Fi", "Western", "War", "Adventure", "Musical","Action", "Horror", "Thriller", "Fantasy", "Mystery", "Crime", "Family", "History" ])

  
    

# Show a slider widget with the years using `st.slider`.
years = st.slider("Years", 1986, 2016, (2000, 2016))

# Filter the movie dataframe based on the widget input and reshape it.
df_filtered = movie_df[(movie_df["genre"].isin(genres)) & (movie_df["year"].between(years[0], years[1]))]
df_reshaped = df_filtered.pivot_table(
    index="year", columns="genre", values="Country",  aggfunc="Title", fill_value=0)
df_reshaped = df_reshaped.sort_values(by="year", ascending=False)

# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_reshaped,
    use_container_width=True,
    column_config={"year": st.column_config.TextColumn("Year")},
)

# Display the data as an Altair chart using `st.altair_chart`.
df_chart = pd.melt(
    df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name="gross"
)
chart = (
    alt.Chart(df_chart)
    .mark_line()
    .encode(
        x=alt.X("year:N", title="Year"),
        y=alt.Y("gross:Q", title="Gross earnings ($)"),
        color="genre:N",
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)

# Display a bar chart for species strategies.
species_chart = alt.Chart(species_df).mark_bar().encode(
    x='species:N',
    y='protection:Q',
    color='species:N',
    tooltip=['protection', 'defense', 'attack', 'feeding', 'satisfaction', 'sexual_reproduction']
).properties(height=320, width=640)
st.altair_chart(species_chart, use_container_width=True)

# Prepare data for linear regression plot.
x = np.array(species_df['feeding']).reshape(-1, 1)
y = np.array(species_df['satisfaction']).reshape(-1, 1)

# Fit the regression model.
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Create a DataFrame with the regression results.
regression_df = pd.DataFrame({
    'feeding': species_df['feeding'],
    'satisfaction': species_df['satisfaction'],
    'predicted_satisfaction': y_pred.flatten()
})

# Display the linear regression chart.
regression_chart = alt.Chart(regression_df).mark_point().encode(
    x='feeding:Q',
    y='satisfaction:Q'
) + alt.Chart(regression_df).mark_line(color='red').encode(
    x='feeding:Q',
    y='predicted_satisfaction:Q'
)
st.altair_chart(regression_chart, use_container_width=True)

# Display a line chart for the movie data.
movie_line_chart = alt.Chart(df_chart).mark_line().encode(
    x='year:N',
    y='gross:Q',
    color='genre:N'
).properties(height=320, width=640)
st.altair_chart(movie_line_chart, use_container_width=True)
