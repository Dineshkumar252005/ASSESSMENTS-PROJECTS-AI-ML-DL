import pandas as pd

# ---- INPUT SECTION ----
# Just edit these lists with your data

dates = [
    "2015-12-06", "2011-12-27", "2015-09-07", "2012-12-21", "2020-02-13", 
    "2015-06-09", "2013-03-21", "2012-09-22", "2013-06-19", "2016-03-05"
]

ids = [498, 721, 375, 464, 813, 853, 918, 422, 380, 403]

physics = [22, 45, 1, 65, 22, 17, 12, 51, 37, 40]
chemistry = [52, 56, 32, 50, 24, 61, 17, 0, 28, 39]
maths = [63, 37, 68, 62, 43, 42, 56, 36, 50, 7]

# ---- PROCESSING ----
df = pd.DataFrame({
    "Date": pd.to_datetime(dates),
    "ID": ids,
    "Physics": physics,
    "Chemistry": chemistry,
    "Maths": maths
})

# Extract month
df["Month"] = df["Date"].dt.strftime("%b").str.upper()

# Count registrations
month_counts = df["Month"].value_counts()
max_reg = month_counts.max()
months_with_max = month_counts[month_counts == max_reg].index

# Month order for resolving ties
month_order = {
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12
}
selected_month = max(months_with_max, key=lambda m: month_order[m])

# Filter for selected month
df_month = df[df["Month"] == selected_month]

# Averages
avg_physics = round(df_month["Physics"].mean(), 2)
avg_chemistry = round(df_month["Chemistry"].mean(), 2)
avg_maths = round(df_month["Maths"].mean(), 2)

# ---- OUTPUT ----
print(selected_month, max_reg, avg_physics, avg_chemistry, avg_maths)