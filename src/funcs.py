# %%
# Library imports
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
# import seaborn as sns
# import statsmodels as sm
# import pyarrow as pa
# pa.__file__
# pa.__version__

import polars as pl
import polars.selectors as cs

from datetime import datetime, timedelta, date
from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd
from pandas.tseries.frequencies import to_offset
from scipy.stats import percentileofscore

# import ISLP as islp

# import xlsxwriter

# plt.style.use('ggplot')
# ax1 = sns.set_style(style=None, rc=None )

# sns.axes_style()
# sns.set_style()
# sns.plotting_context()
# sns.set_context()
# sns.despine()

# %%
def pairs(
    # dataset
    data: pl.DataFrame,
    # variable names
    x_vars: list[str] = None, 
    y_vars: list[str] = None, 
    hue: str = None,
    
    # define visualizations based on variable types and pairs based on
    # a tuple in the form (seaborn visualization function, dict[kwargs])
    # each variable type pair has 2 visualizations, where the
    # first visualization (_1) is for lower triangle/non-square grids, and the
    # second visualization (_2) is for the upper triangle. 
    # Where a variable matches with itself, regardless of its position, 
    # the diag_{numerical, categorical} graphs are plotted instead
    
    # numeric against numeric, visualizations #1 and #2
    numerical_numerical_1 = (
        sns.regplot,
        {
        "ci": None, "line_kws": dict(color="black")
        }
    ),
    numerical_numerical_2 = (
        sns.histplot, 
        {
        "bins": 20
        }
    ),
    
    # categorical against numeric, visualizations #1 and #2
    categorical_numerical_1 = (sns.boxplot, {}),
    categorical_numerical_2 = (sns.violinplot, {}),
    
    # categorical against categorical, visualizations #1, and #2
    categorical_categorical_1 = (
        sns.heatmap,
        {
        "annot": True, "fmt": "d"
        }
    ),
    categorical_categorical_2 = (sns.histplot, {}),
    
    # a variable plotted against itself, dependent on categorical or numeric
    diag_numerical = (sns.histplot, {}),
    diag_categorical = (sns.countplot, {}),
    
    # kwargs for plt.subplots
    **subplots_kwargs
    ):
    
    ## plot all vars against all vars if not specified
    # if x_vars is None:
    #     x_vars = data.columns

    # if y_vars is None:
    #     y_vars = data.columns
    
    # g = sns.PairGrid(data=data, x_vars=x_vars, y_vars=y_vars, **kwargs)
    _, g = plt.subplots(len(y_vars),len(x_vars), **subplots_kwargs)

    # for ax in g.axes.flatten():
    for i in range(len(x_vars)):
        for j in range(len(y_vars)):
            # x, y variable names and data types
            x = x_vars[i]
            y = y_vars[j]
            x_dtype = data.dtypes[data.get_column_index(x)]
            y_dtype = data.dtypes[data.get_column_index(y)]
            
            # assumptions on categorical vs numerical based on polars data type
            if x_dtype in [pl.Categorical, pl.Enum, pl.String]:
                x_dtype = "categorical"
            else:
                x_dtype = "numerical"
                
            if y_dtype in [pl.Categorical, pl.Enum, pl.String]:
                y_dtype = "categorical"
            else:
                y_dtype = "numerical"
            
            # visualizations for a variable matched with itself
            if x == y:
                y = None
                if x_dtype == "categorical":        
                    func, func_kwargs = diag_categorical
                else:
                    func, func_kwargs = diag_numerical
                              
            # lower triangle or non-square grid visualization logic
            elif i <= j or len(x_vars) != len(y_vars):
                if x_dtype == "categorical" and y_dtype == "categorical":
                    func, func_kwargs = categorical_categorical_1
                elif x_dtype == "numerical" and y_dtype == "numerical":
                    func, func_kwargs = numerical_numerical_1
                else:
                    func, func_kwargs = categorical_numerical_1
            
            # upper triangle
            else:
                if x_dtype == "categorical" and y_dtype == "categorical":
                    func, func_kwargs = categorical_categorical_2                 
                elif x_dtype == "numerical" and y_dtype == "numerical":
                    func, func_kwargs = numerical_numerical_2
                else:
                    func, func_kwargs = categorical_numerical_2

            ### draw the graph on the corresponding axis
            # skip when no function is defined 
            if func is None:
                continue

            # matplotlib's plt.subplots has 0, 1, or 2 dimensions depending on
            # length of x_vars and y_vars specified
            if (len(x_vars) == 1) and (len(y_vars) == 1):
                ax = g
            elif len(x_vars) == 1:
                ax = g[j]
            elif len(y_vars) == 1:
                ax = g[i]
            else:
                ax = g[j,i]
            
            # customize plot logic by seaborn plot function
            # some functions may not have a 'hue' argument etc.
            if func in [sns.regplot]:
                func(data=data, x=x, y=y, ax=ax, **func_kwargs)
                ax.set(xlabel=x, ylabel=y)
            elif func in [sns.heatmap]:
                pivot_table = (
                    data.select(x, y)
                    .group_by(x, y)
                    .len()
                    .pivot(on=x, index=y)
                    .fill_null(0)
                    .to_pandas()
                    .set_index(y)
                )
                func(data=pivot_table, ax=ax, **func_kwargs)
                ax.set(xlabel=x, ylabel=y)
            else:
                func(data=data, x=x, y=y, ax=ax, hue=hue, **func_kwargs)

    return g

# %%    ###### CONVERT TO ENUM FUNCTION ########


def to_ordered_enum(
    data: pl.DataFrame, 
    colnames: list[str],
    ):
    """
    Converts selected columns of a polars dataframe to an ordered enum type.
    Works for string and integer columns, but note that it does not work for 
    integer-based string type columns that are not zero padded. 
    E.g. ['1', '4', '12', '25']. In this case, need to cast the string column 
    to an integer type column first
    """
    # initialise list of expressions to cast 'data' to
    exprs = []
    
    for col in colnames:
        dtype = data.dtypes[data.get_column_index(col)]
        
        # generate list for the Enum type to cast to
        if dtype.is_(pl.Categorical):
            sorted_values = data[col].unique().cat.get_categories().sort()
        elif dtype.is_(pl.Enum):
            sorted_values = data[col].unique().cast(pl.String).sort()
        elif dtype.is_numeric():
            max_digits = len(str(data[col].max()))
            sorted_values = data[col].unique().sort().cast(pl.String).str.zfill(max_digits) 
        else:
            sorted_values = data[col].unique().sort()

        
        # append an expression to list depending on if it is a numeric type
        if dtype.is_numeric():
            exprs.append(pl.col(col).cast(pl.String).str.zfill(max_digits).cast(pl.Enum(sorted_values)))
        else:
            exprs.append(pl.col(col).cast(pl.Enum(sorted_values)))

    # change data types of 'data' to Enum for the list of expressions provided
    data = data.with_columns(*exprs)
    
    return data

# %%    ### Fix for -inf/inf categories from qcut/cut function in polars ###
def fix_cut(data: pl.DataFrame, colnames, col_mins, col_maxs):

    exprs = []
    
    ### For each column, replace -infs/infs with the supplied column mins/maxs
    ### Also converts the column to an Enum
    for col, col_min, col_max in zip(colnames, col_mins, col_maxs):
        min_format = "[{}".format(col_min)
        max_format = "{}]".format(col_max)
        
        enum_list = (
            data[col]
            .unique().cast(pl.String).str.replace_many(
            ["(-inf",   "[-inf",     "inf)",     "inf]" ], 
            [min_format, min_format, max_format, max_format])
            .to_list()
        )

        exprs.append(
            pl.col(col).cast(pl.String)
            .str.replace_many(
                ["(-inf",   "[-inf",     "inf)",     "inf]" ], 
                [min_format, min_format, max_format, max_format])
            .cast(pl.Enum(enum_list))
        )
        
    return data.with_columns(exprs)

test_sql = pl.DataFrame([pl.Series("some_name", [1,4,5,1]), pl.Series("some_name2", [2,6,1,7])])
fix_cut(test_sql.select(pl.col("some_name").qcut(2), pl.col("some_name2").qcut(3)), 
        ["some_name", "some_name2"], [1, 1], [5, 7]).dtypes

# %% Test-Train split
def train_test_split_lazy(
    df: pl.LazyFrame, train_fraction: float = 0.75
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split polars dataframe into two sets.
    Args:
        df (pl.DataFrame): Dataframe to split
        train_fraction (float, optional): Fraction that goes to train. Defaults to 0.75.
    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple of train and test dataframes
    """
    df = df.with_columns(pl.all().shuffle(seed=1)).with_row_index()
    # df = df.with_columns(pl.all().shuffle(seed=1))
    
    # df_train = df.filter(pl.col("index") < pl.col("index").max() * train_fraction)
    # df_test = df.filter(pl.col("index") >= pl.col("index").max() * train_fraction)
    
    # df_train = df.filter(pl.col("index") < pl.len() * train_fraction)
    # df_test = df.filter(pl.col("index") >= pl.len() * train_fraction)

    # this is better and faster than above
    df_height = df.select(pl.len()).collect().item()
    train_num = round(df_height * train_fraction)
    test_num = df_height - train_num
    df_train = df.head( train_num )
    df_test = df.tail( test_num )
    
    return df_train, df_test

# %%  ### Variable overview across dataset, output to an Excel file ###
def var_overview(data, filename):
    with xlsxwriter.Workbook(filename) as wb:  
        # create format for percent-formatted columns
        perc_format = wb.add_format({'num_format': '#,##0.00%'})  
        
        for col in data.columns:
            # create the worksheet for the variable
            ws = wb.add_worksheet(col)
            
            # 1. { ... }
            temp = (
                data.group_by(col).agg(
                    pl.len().alias("count"),
                    (pl.len() / data.height).alias("count_perc"),
                # pl.sum("Exposure").alias("Total_Exposure"),
                # pl.sum("ClaimNb").alias("Total_ClaimNb"),
                # )
                # .with_columns(
                #     (pl.col("Total_ClaimNb") / pl.col("Total_Exposure")).alias("Claim_Freq"),
                #     (pl.sum("Total_ClaimNb") / pl.sum("Total_Exposure")).alias("average_freq")
                ).sort(col)
            )
            # print(temp)
            
            # output this section only if lower than 100,000 categories
            max_height_1 = 100_000
            if temp.height <= max_height_1:
                temp.write_excel(
                    workbook=wb, 
                    worksheet=col,
                    position="A1",
                    table_name=col,
                    table_style="Table Style Medium 26",
                    hide_gridlines=True,
                    column_formats={'count_perc': '0.00%'
                                    # , 'Claim_Freq': '0.00%'
                                    # , 'Total_Exposure': '#,##0'
                                    },
                    autofit=True
                )
                
                
            # 2. { ... }
            summary = data.select(pl.col(col)).to_series().describe()
            additional = temp.select(
                pl.col(col).len().alias("distinct_count")
                ).unpivot(variable_name="statistic", value_name="value")
            
            summary.write_excel(
                workbook=wb, 
                worksheet=col,
                position=(0, temp.width + 1),
                table_name=col + "_summary",
                table_style="Table Style Medium 26",
                hide_gridlines=True,
                autofit=True
            )
            
            additional.write_excel(
                workbook=wb, 
                worksheet=col,
                position=(summary.height + 2, temp.width + 1),
                table_name=col + "_additional",
                table_style="Table Style Medium 26",
                hide_gridlines=True,
            )  

            # 3. { ... }
            # don't provide graphs for variables with a high # of categories
            max_height_2 = 1000
            if temp.height > max_height_2:
                continue
            
            # don't include data labels in graph if exceeding 10 unique values
            max_height_3 = 10
            data_labels = temp.height <= max_height_3
            
            # Row count chart
            chart = wb.add_chart({"type": "column"})
            chart.set_title({"name": col})
            chart.set_legend({"none": True})
            chart.set_style(38)
            chart.add_series(
                {  # note the use of structured references
                    "values": "={}[{}]".format(col, "count"),
                    "categories": "={}[{}]".format(col, col),
                    "data_labels": {"value": data_labels},
                }
            )
            
            # add chart to the worksheet
            ws.insert_chart(0, temp.width + 1 + summary.width + 1, chart)
        
            # # Exposure and Freq chart
            # column_chart = wb.add_chart({"type": "column"})
            # column_chart.set_title({"name": col})
            # column_chart.set_legend({"none": False, "position": "bottom"})
            # column_chart.set_style(38)
            # column_chart.add_series(
            #     {  # note the use of structured reference
            #         "name": "Total_Exposure",
            #         "values": "={}[{}]".format(col, "Total_Exposure"),
            #         "categories": "={}[{}]".format(col, col),
            #         "data_labels": {"value": False},
            #     }
            # )

            # # Create a new line chart. This will use this as the secondary chart.
            # line_chart = wb.add_chart({"type": "line"})

            # # Configure the data series for the secondary chart. We also set a
            # # secondary Y axis via (y2_axis).
            # line_chart.add_series(
            #     {
            #         "name": "Claim Frequency",
            #         "values": "={}[{}]".format(col, "Claim_Freq"),
            #         "categories": "={}[{}]".format(col, col),
            #         "y2_axis": True,
            #         "line": {'width': 3, 'color': '#770737'}
            #     }
            # )
            
            # line_chart.add_series(
            #     {
            #         "name": "Average Claim Frequency",
            #         "values": "={}[{}]".format(col, "average_freq"),
            #         "categories": "={}[{}]".format(col, col),
            #         "y2_axis": True,
            #         "line": {'width': 1.5, 'dash_type': 'dash'}
            #     }
            # )

            # # Combine the charts.
            # column_chart.combine(line_chart)

            # # Add a chart title and some axis labels.
            # column_chart.set_title({"name": "Exposure and Claim Frequency"})
            # column_chart.set_x_axis({"name": col})
            # column_chart.set_y_axis({"name": "Exposure"})

            # # Note: the y2 properties are on the secondary chart.
            # line_chart.set_y2_axis({"name": "Claim Frequency"})
            
            # ws.insert_chart(18, temp.width + 1 + summary.width + 1, column_chart, 
            #                 options={'x_scale': 1.5, 'y_scale': 1.5}
            # )