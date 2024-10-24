# %%
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 3)
plt.rcParams["font.family"] = "sans-serif"


# %%
# We saw earlier that pandas is really good at dealing with dates. It is also amazing with strings! We're going to go back to our weather data from Chapter 5, here.
weather_2012 = pl.read_csv(
    "../data/weather_2012.csv", try_parse_dates=True
)
weather_2012[:100]
# %%
# You'll see that the 'Weather' column has a text description of the weather that was going on each hour. We'll assume it's snowing if the text description contains "Snow".
# Pandas provides vectorized string functions, to make it easy to operate on columns containing text. There are some great examples: "http://pandas.pydata.org/pandas-docs/stable/basics.html#vectorized-string-methods" in the documentation.
weather_description = weather_2012["weather"]
is_snowing = weather_description.str.contains("Snow")

# Let's plot when it snowed and when it did not:
is_snowing = is_snowing.cast(pl.Float64)

plt.figure(figsize=(15, 6))
plt.plot(weather_2012["date_time"], is_snowing)
plt.title("Snowing over time")
plt.xlabel("Date")
plt.ylabel("Is Snowing (1.0 = Yes, 0.0 = No)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
# If we wanted the median temperature each month, we could use the `resample()` method like this:
result = weather_2012.group_by(pl.col("date_time").dt.month()).agg(
    pl.col("temperature_c").median()
).sort("date_time")

# Now let's plot the result
plt.figure(figsize=(15, 6))
plt.bar(result["date_time"], result["temperature_c"])
plt.title("Median Temperature by Month")
plt.xlabel("Month")
plt.ylabel("Median Temperature (°C)")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.show()

# Unsurprisingly, July and August are the warmest.
# %%
# # So we can think of snowiness as being a bunch of 1s and 0s instead of `True`s and `False`s:
# Convert boolean to float and show first 10 values
print(is_snowing.cast(pl.Float64).head(10))

# Calculate the percentage of time it was snowing each month
snow_percentage = weather_2012.with_columns(
    pl.col("date_time").dt.month().alias("month"),
    is_snowing.cast(pl.Float64).alias("is_snowing_float")
).group_by("month").agg(
    pl.col("is_snowing_float").mean().alias("snow_percentage")
).sort("month")

# Plot the results
plt.figure(figsize=(15, 6))
plt.bar(snow_percentage["month"], snow_percentage["snow_percentage"])
plt.title("Percentage of Time Snowing by Month")
plt.xlabel("Month")
plt.ylabel("Percentage of Time Snowing")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.show()

# So now we know! In 2012, December was the snowiest month. Also, this graph suggests something that I feel -- it starts snowing pretty abruptly in November, and then tapers off slowly and takes a long time to stop, with the last snow usually being in April or May.

# %%
