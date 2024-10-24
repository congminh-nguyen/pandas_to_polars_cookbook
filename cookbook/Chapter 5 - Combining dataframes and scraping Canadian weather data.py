# %%
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os

# Make the graphs a bit prettier, and bigger
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 3)
plt.rcParams["font.family"] = "sans-serif"

# Get the directory of the current script
current_dir = os.getcwd()
# %%
# By the end of this chapter, we're going to have downloaded all of Canada's weather data for 2012, and saved it to a CSV. We'll do this by downloading it one month at a time, and then combining all the months together.
# Here's the temperature every hour for 2012!

# Read the CSV file
weather_2012_final = pl.read_csv(f"{current_dir}/data/weather_2012.csv").with_columns(pl.col("date_time").str.to_datetime()).sort("date_time")

# %%# Create the plot
plt.figure(figsize=(15, 6))
plt.plot(weather_2012_final['date_time'], weather_2012_final['temperature_c'])
plt.title('Temperature over time')
plt.xlabel('Date'), plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Okay, let's start from the beginning.
# We're going to get the data for March 2012, and clean it up
# You can directly download a csv with a URL using Pandas!
# Note, the URL the repo provides is faulty but kindly, someone submitted a PR fixing it. Have a look
# here: https://github.com/jvns/pandas-cookbook/pull/74 and click on "Files changed" and then fix the url.# %%

# This URL has to be fixed first!
url_template = "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=5415&Year={year}&Month={month}&timeframe=1&submit=Download+Data"
url_march = url_template.format(month=3, year=2012)

weather_mar2012 = pl.read_csv(
    url_march,
    encoding="latin1",
    try_parse_dates=True
).sort("Date/Time (LST)")

# Display the first few rows
print(weather_mar2012.head())

# %%
# Let's clean up the data a bit.
# You'll notice in the summary above that there are a few columns which are are either entirely empty or only have a few values in them. Let's get rid of all of those with `dropna`.
# The argument `axis=1` to `dropna` means "drop columns", not rows", and `how='any'` means "drop the column if any value is null".
# Drop columns with any null values
def select_no_nulls_or_nan(df: pl.DataFrame) -> pl.DataFrame:
    # Check which columns have nulls
    has_nulls = df.select(pl.all().is_null().any()).to_dict()

    # Check which columns have NaN values (only for float columns)
    has_nan = df.select(
        [pl.col(col).is_nan().any() for col in df.columns if df.schema[col] == pl.Float64]
    ).to_dict()

    # Combine results for nulls and NaN values
    selection = []
    for name in df.columns:
        has_null = has_nulls.get(name, [False])[0]
        has_nan_value = has_nan.get(name, [False])[0]
        
        # Include column if it has neither null nor NaN values
        if not has_null and not has_nan_value:
            selection.append(name)

    # Select only columns without nulls or NaN values
    return df.select(selection)

weather_mar2012 = weather_mar2012.pipe(select_no_nulls_or_nan)

# Display the first 5 rows
print(weather_mar2012.head(5))

# %%
# Let's get rid of columns that we do not need.
# For example, the year, month, day, time columns are redundant (we have Date/Time (LST) column).
# Let's get rid of those. The `axis=1` argument means "Drop columns", like before. The default for operations like `dropna` and `drop` is always to operate on rows.
weather_mar2012 = weather_mar2012.drop(["Year", "Month", "Day", "Time (LST)"])
print(weather_mar2012.head(5))

# %%
# When you look at the data frame, you see that some column names have some weird characters in them.
# Let's clean this up, too.
# Let's print the column names first:
print(weather_mar2012.columns)

# And now rename the columns to make it easier to work with
weather_mar2012 = weather_mar2012.rename(
    {col: col.replace('ï»¿"', " ").replace("Â", "").replace(')"', ")") for col in weather_mar2012.columns}
)

# Print the updated column names
print(weather_mar2012.columns)

# %%
# Optionally, you can also rename columns more manually for specific cases:
# Create a dictionary to rename columns, matching exactly with current column names
# Rename the columns
weather_mar2012 = weather_mar2012.with_columns([
    pl.col('Date/Time (LST)').alias('datetime'),
    pl.col('Station Name').alias('Station_Name'),
    pl.col('Climate ID').alias('Climate_ID'),
    pl.col('Temp (°C)').alias('Temperature_C'),
    pl.col('Dew Point Temp (°C)').alias('Dew_Point_Temp_C'),
    pl.col('Rel Hum (%)').alias('Relative_Humidity'),
    pl.col('Wind Spd (km/h)').alias('Wind_Speed_kmh'),
    pl.col('Visibility (km)').alias('Visibility_km'),
    pl.col('Stn Press (kPa)').alias('Station_Pressure_kPa')
])

# Print the new column names
print(weather_mar2012.columns)

# Check the new column names
# print(weather_mar2012.columns)

# Some people also prefer lower case column names.
# Convert column names to lowercase
weather_mar2012 = weather_mar2012.select(pl.all().name.to_lowercase())

# Print column names
print(weather_mar2012.columns)

# Notice how it goes up to 25° C in the middle there? That was a big deal. It was March, and people were wearing shorts outside.
plt.figure(figsize=(15, 5))
plt.plot(weather_mar2012['temperature_c'])
plt.title('Temperature in March 2012')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.show()

# %%
# This one's just for fun -- we've already done this before, using groupby and aggregate! We will learn whether or not it gets colder at night. Well, obviously. But let's do it anyway.
temperatures = weather_mar2012.select(["temperature_c"])
print(temperatures.head())
temperatures = weather_mar2012.select(["temperature_c", "datetime"])
temperatures = temperatures.with_columns(pl.col("datetime").dt.hour().alias("Hour"))
temp_medians = temperatures.group_by("Hour").agg(pl.col("temperature_c").median()).sort("Hour")

# Plotting using matplotlib
plt.plot(temp_medians["Hour"], temp_medians["temperature_c"])
plt.xlabel("Hour")
plt.ylabel("Median Temperature (°C)")
plt.title("Median Temperature by Hour")
plt.grid(True)
plt.show()
# %%
# Okay, so what if we want the data for the whole year? Ideally the API would just let us download that, but I couldn't figure out a way to do that.
# First, let's put our work from above into a function that gets the weather for a given month.

def clean_data(data):
    data = data.pipe(select_no_nulls_or_nan)
    data = data.drop(["Year", "Month", "Day", "Time (LST)"])
    data = data.rename(
    {col: col.replace('ï»¿"', " ").replace("Â", "").replace(')"', ")") for col in data.columns})
    data = data.with_columns([
    pl.col('Date/Time (LST)').alias('datetime'),
    pl.col('Station Name').alias('Station_Name'),
    pl.col('Climate ID').alias('Climate_ID'),
    pl.col('Temp (°C)').alias('Temperature_C'),
    pl.col('Dew Point Temp (°C)').alias('Dew_Point_Temp_C'),
    pl.col('Rel Hum (%)').alias('Relative_Humidity'),
    pl.col('Wind Spd (km/h)').alias('Wind_Speed_kmh'),
    pl.col('Visibility (km)').alias('Visibility_km'),
    pl.col('Stn Press (kPa)').alias('Station_Pressure_kPa')
])
    data = data.select(pl.all().name.to_lowercase())
    return data

def download_weather_month(year, month):
    url_template = "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=5415&Year={year}&Month={month}&timeframe=1&submit=Download+Data"
    url = url_template.format(year=year, month=month)
    weather_data = pl.read_csv(url, encoding="latin1", try_parse_dates=True, truncate_ragged_lines=True).sort("Date/Time (LST)")
    weather_data_clean = clean_data(weather_data)
    return weather_data_clean

# %%
print(download_weather_month(2013, 1).head())
# %%
# Now, let's use a list comprehension to download all our data and then just concatenate these data frames
# This might take a while
# Download data for each month and store in a list
data_by_month = [download_weather_month(2012, i) for i in range(1, 13)]

# Get all unique column names across all DataFrames
all_columns = set()
for df in data_by_month:
    all_columns.update(df.columns)

# Add missing columns to each DataFrame with None values
data_by_month_aligned = []
for df in data_by_month:
    missing_columns = all_columns - set(df.columns)
    for col in missing_columns:
        df = df.with_columns(pl.lit(None).alias(col))  # Add missing column with None
    data_by_month_aligned.append(df)

# Ensure column order is the same across all DataFrames
data_by_month_aligned = [df.select(sorted(all_columns)) for df in data_by_month_aligned]

# Concatenate the DataFrames
weather_2012 = pl.concat(data_by_month_aligned)

# Print the result
print(weather_2012.head())
# %%
# Now, let's save the data.
weather_2012.write_csv(f"{current_dir}/data/weather_2012.csv")# %%
