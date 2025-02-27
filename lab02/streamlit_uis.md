# Building beautiful UIs for you data science application

Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. It's developers claim that you can build and deploy powerful data apps in minutes - let's put this to the test!

## Installation

As almost every python package, you install `streamlit` via `pip` or `conda`. For pip, it's the usual

```shell
pip install streamlit
```

and for conda, you install `streamlit` from `conda-forge`:

```shell
conda install -c conda-forge streamlit
```

However, you don't have to worry about this as we have already taken care of it for you in the lab environment.

## Main concepts

Streamlit comes with its own CLI. You add some Streamlit commands to your python script and then you run it using

```shell
streamlit run your_script.py [-- script args]
```

> As soon as you run the script as shown above, a local Streamlit server will spin up and your app will open in a new tab in your default web browser. The app is your canvas, where you'll draw charts, text, widgets, tables, and more.

In case you were wondering, yes, Streamlit comes with auto-reloading.

### Data flow

Streamlit's architecture allows you to write apps the same way you write plain Python scripts. To unlock this, Streamlit apps have a unique data flow: **any time something must be updated on the screen, Streamlit reruns your entire Python script from top to bottom**. Keeps this in mind, it's both powerful and burdensome when your apps grow.

Streamlit reruns your app whenever you modify its source code or interact with the app (e.g. by dragging a slider or entering text in a textbox). Whenever a widget is interacted with, its callback (`on_change` or `on_click`) will run before the rest of the script.

So, how come this system works even though the app is reloaded every time your interact with it? Under the hood, Streamlit relies on _caching_. So, expensive computations are only re-run when they absolutely **have** to be rerun.

But enough theory, let's look at some code!

_You can follow along by running the examples in the `snippets` directory.

## Display and style data

Streamlit offers you a few ways to display data tables, arrays, data frames - you get the idea. We'll first look at _magic_ and `st.write`, both of which can be used to write anything from text to tables.

Using _magic_ you can create Streamlit apps entirely without writing Streamlit code. Magic commands allow you to write almost anything (markdown, data, charts) without having to type an explicit command at all. Just put the thing you want to show on its own line of code, and it will appear in your app.

Here's the example from the Streamlit documentation:

```python
# Draw a title and some text to the app:
'''
# This is the document title

This is some _markdown_.
'''

import pandas as pd
df = pd.DataFrame({'col1': [1,2,3]})
df  # 👈 Draw the dataframe

x = 10
'x', x  # 👈 Draw the string 'x' and then the value of x

# Also works with most supported chart types
import matplotlib.pyplot as plt
import numpy as np

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

fig  # 👈 Draw a Matplotlib chart
```

You can find this file in `snippets/magic.py`.

That's an impressive trick, but what's the _real_ way to do this?
Glad you asked. _magic_ is not **really** magic, it's [`st.write`](https://docs.streamlit.io/library/api-reference/write-magic/st.write).

`st.write` writes its arguments to the app. Streamlit calls it the

> Swiss Army knife of Streamlit commands: it does different things depending on what you throw at it.

`st.write` has some unique abilities. First and foremost, you can pass as many arguments as you like, they'll all be written to the app. Second, the behavior depends on the type of the argument. You can find the whole list [here](https://docs.streamlit.io/library/api-reference/write-magic/st.write).

Using it is as easy as using _magic_, albeit with a bit more typing. Here's the example from above, but this time with `st.write`.

```python
import streamlit as st
# Draw a title and some text to the app:
st.write('''
# This is the document title

This is some _markdown_.
''')

import pandas as pd
df = pd.DataFrame({'col1': [1,2,3]})
st.write(df)  # 👈 Draw the dataframe

x = 10
st.write('x', x)  # 👈 Draw the string 'x' and then the value of x

# Also works with most supported chart types
import matplotlib.pyplot as plt
import numpy as np

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.write(fig)  # 👈 Draw a Matplotlib chart
```

You can find this snippet in `snippets/nomagic.py`.

As you can see, the difference is small but notable.

### Your turn

Modify `snippets/nomagic.py` see what types `st.write` does and does not support. What about torch models? Can it display a transformer from `transformers`? Let your creativity flow freely.
Any surprises?

## Everything but `st.write`

Why would you ever want to use something other than `st.write`?
There are actually a few reasons:

> Magic and st.write() inspect the type of data that you've passed in, and then decide how to best render it in the app. Sometimes you want to draw it another way. For example, instead of drawing a dataframe as an interactive table, you may want to draw it as a static table by using st.table(df).
>
> The second reason is that other methods return an object that can be used and modified, either by adding data to it or replacing it.
>
> Finally, if you use a more specific Streamlit method you can pass additional arguments to customize its behavior.
>
> [streamlit docs](https://docs.streamlit.io/get-started/fundamentals/main-concepts#write-a-data-frame)

For instance, there's the `st.dataframe` method for rendering tabular data (that is, any type that can be converted into a pandas dataframe) as dataframes.

```python
import streamlit as st
import numpy as np

dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)
```

You can find this snippet in `snippets/numpy_dataframe.py`.

When used with pandas `DataFrames`, you benefit from pandas' `Styler` objects!

```python
import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10, 20), columns=("col %d" % i for i in range(20)))

st.dataframe(df.style.highlight_min(axis=0))
```

C.f. `snippets/dataframe.py`.

If you have never heard of pandas' table visualization features, we highly recommend to read through the corresponding page in the [pandas docs](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html).

Streamlit has support for pandas' table styling. The next snippets demonstrates some examples from the previously mentioned page.

```python
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
```

You can find this snippet in `snippets/styled_dataframe.py`.

## Beyond dataframes

Streamlit can draw a wide array of elements. You can find the complete list in the [API reference](https://docs.streamlit.io/library/api-reference). As a little demo, the following snippet plots a scatter plot on a map of San Francisco:

```python
import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "col1": np.random.randn(1000) / 50 + 37.76,
    "col2": np.random.randn(1000) / 50 + -122.4,
    "col3": np.random.randn(1000) * 100,
    "col4": np.random.rand(1000, 4).tolist(),
})

st.map(df,
    latitude='col1',
    longitude='col2',
    size='col3',
    color='col4')
```

You can find this snippet in `snippets/map.py`.

## Widgets

So far, our UIs have displayed data, but there was not really any possibility for user to interact with the data. Widgets make your UIs interactive by adding components like sliders, buttons, and checkboxes that the user can interact with.

For example, here's a slider:

```python
import streamlit as st
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)
```

You can find this snippet in `snippets/slider.py`.

If you run this snippet, you will experience what we've mentioned in the section about dataflow: Whenever you interact with a widget, the app is re-run. Here, the `x * x` is re-evaluated as soon as you let go of the slider.

Streamlit also keeps what is known as the _session state_:

> Session State is a way to share variables between reruns, for each user session. In addition to the ability to store and persist state, Streamlit also exposes the ability to manipulate state using Callbacks. Session state also persists across apps inside a multipage app.

Every widget is automatically added to the session state, and you can access widgets via their key in the session state:

```python
import streamlit as st
st.text_input("Your name", key="name")  # <-- Set the key of this text box to "name".

# You can access the value at any point with:
st.session_state.name
```

You can find this snippet in `snippets/session_state.py`.
We will leverage the session state in the last part of this lab, so a closer look at the [relevant section in the docs](https://docs.streamlit.io/library/advanced-features/session-state) is recommended!

Sometimes, one wants to show or hide a specific chart or section in an app. `st.checkbox` takes a single argument, which is the widget label. In this sample, the checkbox is used to toggle a conditional statement. Similarly, you can use `st.selectbox` to choose from a series. You can write in the options you want, or pass through an array or data frame column.

Here's an example using both.

```python
import streamlit as st
import numpy as np
import pandas as pd

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option
```

As always, you can find this snippet in `snippets/check_and_select.py`

## Layout

So far, we've been placing all elements in a single, centered column. Using layout functions, you can control the placement of your widgets on the page. Examples include:

- `st.sidebar`: Each element that's passed to `st.sidebar` is pinned to the left.
- `st.columns`: `st.columns` lets you place widgets side-by-side.
- `st.expander`: Conserves space by hiding away large elements.

You can find the complete list in the [API reference](https://docs.streamlit.io/library/api-reference/layout).
Here is an example using the three layout functions mentioned before.

```python
import streamlit as st

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")

st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

expander = st.expander("See explanation")
expander.write(\"\"\"
    The chart above shows some numbers I picked for you.
    I rolled actual dice for these, so they're *guaranteed* to
    be random.
\"\"\")
expander.image("https://static.streamlit.io/examples/dice.jpg")

# `st.expander` also supports `with`

st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})
with st.expander("See explanation"):
    st.write(\"\"\"
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    \"\"\")
    st.image("https://static.streamlit.io/examples/dice.jpg")
```

You can find this snippet in `snippets/layout.py`.

## Where to go from here

There are many advanced features such as multi-page apps we haven't covered yet, but you can learn more about them in the Streamlit docs! For instance:

- [Advanced concepts](https://docs.streamlit.io/get-started/fundamentals/advanced-concepts)
- [Additional features](https://docs.streamlit.io/get-started/fundamentals/additional-features)