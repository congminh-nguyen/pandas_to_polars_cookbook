# %%
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# Make the graphs a bit prettier, and bigger
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["font.family"] = "sans-serif"

# This is necessary to show lots of columns in pandas 0.12.
# Not necessary in pandas 0.13.
pd.set_option("display.width", 5000)
pd.set_option("display.max_columns", 60)

# %% Load the data
bikes = pd.read_csv(
    "../data/bikes.csv",
    sep=";",
    encoding="latin1",
    parse_dates=["Date"],
    dayfirst=True,
    index_col="Date",
)
bikes["Berri 1"].plot()
plt.show()

# %% Load the data
# TODO: Load the data using Polars
pl_bikes = pl.read_csv(
    "../data/bikes.csv",
    separator=";",
    encoding="latin1",
    try_parse_dates=True,
)

# Plot Berri 1 data
plt.figure(figsize=(15, 5))
plt.plot(pl_bikes["Date"], pl_bikes["Berri 1"])
plt.title("Berri 1 Bike Path Usage")
plt.xlabel("Date")
plt.ylabel("Number of Cyclists")
plt.show()

# %% Plot Berri 1 data
# Next up, we're just going to look at the Berri bike path. Berri is a street in Montreal, with a pretty important bike path. I use it mostly on my way to the library now, but I used to take it to work sometimes when I worked in Old Montreal.

# So we're going to create a dataframe with just the Berri bikepath in it
berri_bikes = bikes[["Berri 1"]].copy()
berri_bikes[:5]

# TODO: Create a dataframe with just the Berri bikepath using Polars
# Hint: Use pl.DataFrame.select() and call the data frame pl_berri_bikes

pl_berri_bikes = pl_bikes.select(["Date", "Berri 1"])
print(pl_berri_bikes.head())


# %% Add weekday column
# Next, we need to add a 'weekday' column. Firstly, we can get the weekday from the index. We haven't talked about indexes yet, but the index is what's on the left on the above dataframe, under 'Date'. It's basically all the days of the year.

berri_bikes.index

# You can see that actually some of the days are missing -- only 310 days of the year are actually there. Who knows why.

# Pandas has a bunch of really great time series functionality, so if we wanted to get the day of the month for each row, we could do it like this:
berri_bikes.index.day

# We actually want the weekday, though:
berri_bikes.index.weekday

# These are the days of the week, where 0 is Monday. I found out that 0 was Monday by checking on a calendar.

# Now that we know how to *get* the weekday, we can add it as a column in our dataframe like this:
berri_bikes.loc[:, "weekday"] = berri_bikes.index.weekday
berri_bikes[:5]

# TODO: Add a weekday column using Polars.
# Hint: Polars does not use an index.
# %%
pl_berri_bikes = pl_berri_bikes.with_columns(
    (pl.col("Date").dt.weekday() - 1).alias("weekday")
)
print(pl_berri_bikes.head())

# The Polars version uses 1 more than the original because Polars' weekday() function
# returns values from 1 (Monday) to 7 (Sunday), while Pandas' weekday attribute
# returns values from 0 (Monday) to 6 (Sunday). This difference in indexing
# conventions between the two libraries results in the Polars version having
# weekday values that are one higher than the Pandas version.



# %%
# Let's add up the cyclists by weekday
# This turns out to be really easy!

# Dataframes have a `.groupby()` method that is similar to SQL groupby, if you're familiar with that. I'm not going to explain more about it right now -- if you want to to know more, [the documentation](http://pandas.pydata.org/pandas-docs/stable/groupby.html) is really good.

# In this case, `berri_bikes.groupby('weekday').aggregate(sum)` means "Group the rows by weekday and then add up all the values with the same weekday".
weekday_counts = berri_bikes.groupby("weekday").aggregate(sum)
weekday_counts

# TODO: Group by weekday and sum using Polars
pl_weekday_counts = pl_berri_bikes.group_by("weekday").agg(pl.sum("Berri 1").alias("total_cyclists")).sort("weekday")
print(pl_weekday_counts)


# %%
# %% Rename index
weekday_counts.index = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
# %%

# %%
# TODO: Rename index using Polars, if possible.
weekday_map = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

# Apply the mapping
pl_weekday_counts = pl_weekday_counts.with_columns(
    pl.col("weekday").replace_strict(weekday_map).alias("weekday")
)


# %% Plot results
weekday_counts.plot(kind="bar")
plt.show()

# TODO: Plot results using Polars and matplotlib
weekdays = pl_weekday_counts["weekday"].to_list()
total_cyclists = pl_weekday_counts["total_cyclists"].to_list()

# Plot the results
plt.bar(weekdays, total_cyclists)
plt.xlabel("Weekday")
plt.ylabel("Total Cyclists")
plt.title("Total Cyclists by Weekday")
plt.show()

# %% Final message
print("Analysis complete!")
