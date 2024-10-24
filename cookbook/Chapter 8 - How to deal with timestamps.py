# %%
import pandas as pd
import polars as pl

# %%
# Parsing Unix timestamps
# It's not obvious how to deal with Unix timestamps in pandas -- it took me quite a while to figure this out. The file we're using here is a popularity-contest file of packages.

# Read it, and remove the last row
popcon = pd.read_csv(
    "../data/popularity-contest",
    sep=" ",
)[:-1]
popcon.columns = ["atime", "ctime", "package-name", "mru-program", "tag"]
popcon[:5]

# TODO: please reimplement this using Polars
popcon_pl = pl.read_csv("../data/popularity-contest", separator=' ', infer_schema_length = 10000)[:-1]
popcon_pl.columns = ["atime", "ctime", "package-name", "mru-program", "tag"]
popcon_pl.head()

# %%
# The magical part about parsing timestamps in pandas is that numpy datetimes are already stored as Unix timestamps. So all we need to do is tell pandas that these integers are actually datetimes -- it doesn't need to do any conversion at all.
# We need to convert these to ints to start:
popcon["atime"] = popcon["atime"].astype(int)
popcon["ctime"] = popcon["ctime"].astype(int)

# TODO: please reimplement this using Polars
popcon_pl = popcon_pl.with_columns([
    pl.col("atime").cast(pl.Int64),
    pl.col("ctime").cast(pl.Int64)
])

popcon_pl.head()

# %%
# Every numpy array and pandas series has a dtype -- this is usually `int64`, `float64`, or `object`. Some of the time types available are `datetime64[s]`, `datetime64[ms]`, and `datetime64[us]`. There are also `timedelta` types, similarly.
# We can use the `pd.to_datetime` function to convert our integer timestamps into datetimes. This is a constant-time operation -- we're not actually changing any of the data, just how pandas thinks about it.

popcon["atime"] = pd.to_datetime(popcon["atime"], unit="s")
popcon["ctime"] = pd.to_datetime(popcon["ctime"], unit="s")
#print(popcon.head())

# TODO: please reimplement this using Polars

# Polars datetime doesn't accept seconds as a param so convert to milliseconds
popcon_pl = popcon_pl.with_columns([
    pl.col("atime").mul(1000).alias("atime_ms"),
    pl.col("ctime").mul(1000).alias("ctime_ms")])

# # Cast 'atime_ms' and 'ctime_ms' to datetime with unit milliseconds
popcon_pl = popcon_pl.with_columns([
    pl.col("atime_ms").cast(pl.Datetime(time_unit="ms")),
    pl.col("ctime_ms").cast(pl.Datetime(time_unit="ms"))
])

popcon_pl = popcon_pl.drop(["atime", "ctime"]).rename({
    "atime_ms": "atime",
    "ctime_ms": "ctime"
})


# Display the DataFrame
print(popcon_pl.head())

# %%
# Now suppose we want to look at all packages that aren't libraries.

# First, I want to get rid of everything with timestamp 0. Notice how we can just use a string in this comparison, even though it's actually a timestamp on the inside? That is because pandas is amazing.
popcon = popcon[popcon["atime"] > "1970-01-01"]

# Now we can use pandas' magical string abilities to just look at rows where the package name doesn't contain 'lib'.
nonlibraries = popcon[~popcon["package-name"].str.contains("lib")]
nonlibraries.sort_values("ctime", ascending=False)[:10]

# TODO: please reimplement this using Polars
popcon_pl = popcon_pl.filter(
    pl.col("atime") > pl.datetime(1970, 1, 1)
)

# Filter rows where 'package-name' does NOT contain 'lib'
nonlibraries_pl = popcon_pl.filter(
    ~pl.col("package-name").str.contains("lib")
)

# Sort by 'ctime' in descending order and display the top 10 rows
nonlibraries_pl = nonlibraries_pl.sort("ctime", descending=True).head(10)

print(nonlibraries_pl)

# The whole message here is that if you have a timestamp in seconds or milliseconds or nanoseconds, then you can just "cast" it to a `'datetime64[the-right-thing]'` and pandas/numpy will take care of the rest.

# %%
