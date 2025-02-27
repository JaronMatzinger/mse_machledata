import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame(
    {"strings": ["Adam", "Mike"], "ints": [1, 3], "floats": [1.123, 1000.23]}
)

# You can chain styling methods
st.dataframe(
    df.style.format(  # This adjusts the float previsions.
        precision=3, thousands=".", decimal=","
    )  # This converts the headers to upper case.
    .format_index(str.upper, axis=1)  # And this relabels the rows.
    .relabel_index(["row 1", "row 2"], axis=0)
)

weather_df = pd.DataFrame(
    np.random.rand(10, 2) * 5,
    index=pd.date_range(start="2021-01-01", periods=10),
    columns=["Tokyo", "Beijing"],
)


def rain_condition(v):
    if v < 1.75:
        return "Dry"
    elif v < 2.75:
        return "Rain"
    return "Heavy Rain"


# You can of course tidy your styling logic up and wrap it in a function.
def make_pretty(styler):
    # This sets the caption for your table.
    styler.set_caption("Weather Conditions")
    # Conditional formatting using the `rain_condition` function.
    styler.format(rain_condition)
    # Let's format the index.
    styler.format_index(lambda v: v.strftime("%A"))
    # This colors the cell background on a gradient from 1 to 5.
    styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="YlGnBu")
    return styler


"# Before: "
st.dataframe(weather_df)

"# After: "
st.dataframe(make_pretty(weather_df.style))
