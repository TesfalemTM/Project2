# This file contains relevant function for "Olist project".

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Functions for visualization-

def create_barplot_simple(df, col, title, x_label, y_label, flag, top):
  """ Function creates and displays barplot based on value_counts() of column.
  parm df: selected data frame.
  parm col: The column on which the func is performed.
  parm title: graph title
  parm x_label: X axis title
  parm y_label: Y axis title
  parm flag: bool value, which decides whether there will be different or identical colors for the bars.
  parm top: top values, not mandatory.
  return: none.
  """
  # use value_counts()
  if top is None:
    df_VC = df[col].value_counts() # show all values
  else:
    df_VC = df[col].value_counts().head(top) # show only the top values

  # choose color plate
  plt.figure(figsize=(10, 6))
  if flag:
    barplot= sns.barplot(x=df_VC.index, y=df_VC) # different colors for bars
  else:
     barplot = sns.barplot(x=df_VC.index, y=df_VC, color="steelblue") # same colors for bars

  # add other settings
  barplot.set_title(title)
  barplot.set_xlabel(x_label)
  barplot.set_ylabel(y_label)
  barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)
  plt.tight_layout()
  plt.show()


def create_barplot_complicated(df, col1, col2, title, x_label, y_label, flag):
  """ Function creates and displays barplot based on 2 cols.
  parm df: selected data frame.
  parm col1: categorical col.
  parm col2: numeric col.
  parm title: graph title
  parm x_label: X axis title
  parm y_label: Y axis title
  parm flag: bool value, which decides whether there will be different or identical colors for the bars.
  return: none.
  """

  # choose color plate
  plt.figure(figsize=(10, 6))
  if flag:
    barplot= sns.barplot(x=df[col1], y=df[col2], errorbar=None) # different colors for bars
  else:
    barplot= sns.barplot(x=df[col1], y=df[col2],errorbar=None, color= "steelblue") # same color for bars

  # add other settings
  barplot.set_title(title)
  barplot.set_xlabel(x_label)
  barplot.set_ylabel(y_label)
  barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)
  plt.tight_layout()
  plt.show()


def line_graph(df, col1, col2, my_label,  title, x_label, y_label):
    """Function creates and displays line graph.
    param df: selected data frame.
    param col1: selected column from the data frame (x).
    param col2: selected column from the data frame (y).
    param title: graph title.
    param x_label: X-axis title.
    param y_label: Y-axis title.
    param my_list: legend titles.
    return: none
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df[col1], df[col2], linewidth=2, label=my_label)
    plt.fill_between(df[col1], df[col2], alpha=0.1)
    plt.axhline(y=0)
    plt.title(title)
    plt.xlabel(x_label)
    plt.xticks(rotation=90)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def multiple_line_chart(df, col1, col2, colX, title, x_label, y_label, my_legends):
    """Function creates and displays multiple line chart.
    param df: selected data frame.
    param col1: selected column from the data frame for line1.
    param col2: selected column from the data frame for line2.
    param colX: selected column for x-axis.
    param title: graph title.
    param x_label: X-axis title.
    param y_label: Y-axis title.
    param my_legends: legend titles.
    return: none
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df[colX], df[col1], label=my_legends[0], linewidth=2)
    plt.plot(df[colX], df[col2], label=my_legends[1], color="darkgreen", linewidth=2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.xticks(rotation=90)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def grouped_bar_df(df, col1, col2, colX, title, x_label, y_label, my_labels):
  """Functions create group bar for df.
  param df: selected dataframe.
  param col1: selected col from df.
  param col2: selected col from df.
  param col3: selected col from df.
  parm title: graph title.
  parm x_label: X axis title.
  parm y_label: Y axis title.
  my_labels: bars names.
  return: none.
  """
  plt.figure(figsize=(12, 6))

  # set the positions of bars on X-axis
  r1 = [x * 1.5 for x in range(len(df))]
  r2 = [x + 0.55 for x in r1]

  # create the grouped bar chart
  plt.bar(r1, df[col1], label=my_labels[0])
  plt.bar(r2, df[col2], label=my_labels[1])
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.xticks([r + 0.55 for r in r1], df[colX], rotation=90)
  plt.legend()
  plt.show()


def grouped_bar_crosstab(my_crosstab, title, x_label, y_label, legend_label):
  """Functions creates and displayes grouped_bar_chart.
  param my_crosstab: selected crosstab.
  parm title: graph title.
  parm x_label: X axis title.
  parm y_label: Y axis title.
  legend_label: legend title.
  return: none.
  """
  # create plot
  my_crosstab.plot.bar( figsize=(12, 6), width= 0.8)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(title=legend_label, bbox_to_anchor=(1, 1))
  plt.tight_layout()
  plt.show()


# Functions for data preprocessing

def find_NA(df):
  """Function finds missing values and display them.
  param df: selected data frame.
  return: none.
  """
  # find missing values
  missing_count= df.isna().sum()
  missing_percentage= round((missing_count / len(df)) * 100, 2)

  # display only missing values and percentage
  missing_info = pd.DataFrame({"missing count": missing_count, "missing percentage":missing_percentage})
  df_display = missing_info[missing_info["missing count"] > 0]
  if df_display.empty:
    print("There are no missing values.")
  else:
    display(df_display)


def check_missing_sys (df, name, my_list, col):
  """Function finds whether the values are missing systematically or not.
  param df: selected data frame.
  param name: the name of "my_list"
  param my_list: list that contains names of cols with missing values.
  param col: one col from "my_list".
  return: none.
  """
  print("Missing values for:", name)
  dfNA = df[my_list]

  # find missing values by col
  dfNA = dfNA[dfNA[col].isna()]
  display(dfNA.isna().sum())


def calc_outliers(df, col):
  """"Function finds outliers using IQR.
  param df: selected data frame.
  col: the column in which outliers are searched, not mandatory.
  return: lower and upper thresholds for outliers.
  """
  # calc Q1 and Q3: for the all df
  if col is None:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

  # calc Q1 and Q3: for specific col
  else:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

  # calc manually: IQR, upper_threshold and lower_threshold
  IQR = Q3 - Q1
  lower_threshold = Q1 - 1.5 * IQR
  upper_threshold = Q3 + 1.5 * IQR
  return lower_threshold, upper_threshold


# Functions for data analysis

def customer_growth_rate(df, list_year, list_month, col):
  """Function calcs the customer growth rate using number of new and returned customers.
  param df: selected data frame.
  param list_year: list of years.
  param list_month: list of months (January - December).
  param col: The column for which the growth rate is being calculated.
  return: new dataframe contains values- "year", "month", "returned", "new", "total", "growth_rate".
  """
  df_res = pd.DataFrame(columns=["year", "month", "returned", "new", "total", "growth_rate"])
  skip_2016 = list_month[0:8]
  skip_2018 = list_month[9:12]
  df_prev = pd.Series(dtype=float)
  prev=0

  for year in list_year:
    for month in list_month:
      curr_df = df[(df["year"] == year) & (df["month"] == month)][col].unique()

      # calc values: "returned", "new", "total", "growth_rate"
      returned = len(set(curr_df) & set(df_prev))
      new = len(set(curr_df) - set(df_prev))
      if (((year == 2016) & (month in skip_2016))  | ((year == 2018) & (month in skip_2018))):
        continue
      elif (prev == 0):
        growth =  round(((new - 0.01) /0.01)*100, 2)
      else:
        growth = round(((new- prev) / prev) * 100, 2)
      total = new + returned

      # change values
      prev = new
      df_prev = pd.concat([pd.Series(df_prev), pd.Series(curr_df)], ignore_index=True).drop_duplicates()

       #add new record
      new_record = {"year": year, "month": month, "returned": returned, "new": new, "total": total, "growth_rate":growth}
      df_res = pd.concat([df_res, pd.DataFrame([new_record])], ignore_index=True)

  return df_res


def customer_distribution(df, col1, col2, col3, quarter):
  """Function creates a crosstab that displays the customer distribution per state.
  param df: selected data frame.
  param col1: selected col from df.
  param col2: selected col from df.
  param col3: selected col from df.
  param quarter: quarter to remove, not mandatory.
  return: crosstab with the top 5 states by customer per quarter.
  """
  # filter df
  df_res = df[[col1, col2, col3]]
  df_res = df_res.drop_duplicates(subset= col1)

  # group df and for every state in the quarter count the number of shows
  df_res = df_res.groupby([col2, col3]).size().reset_index(name="count")
  df_res = df_res.sort_values(by=[col3, "count"], ascending=[True, False])

  # create crosstab
  my_crosstab = pd.crosstab(index=df_res[col3], columns=df_res[col2], values=df_res["count"], aggfunc="sum", normalize="index")
  if quarter is not None:
    my_crosstab = my_crosstab.drop(index=quarter)

  # select the top 5 states with the highest number of customers, per quarter
  my_crosstab = my_crosstab.apply(lambda row: row.nlargest(5), axis=1)
  my_crosstab["other"] = 1 - my_crosstab.sum(axis=1)

  # sort crosstab
  my_crosstab = my_crosstab[my_crosstab.mean().sort_values(ascending=False).index]
  return my_crosstab


def order_growth(df, list_year, list_month, col):
  """Function calcs the order growth.
  param df: selected data frame.
  param list_year: list of years.
  param list_month: list of months (January - December).
  param col: The column for which the growth rate is being calculated.
  return: new dataframe contains values: "year", "month", "total", "growth_rate".
  """
  df_res = pd.DataFrame(columns=["year", "month", "total", "growth"])
  skip_2016 = list_month[0:8]
  skip_2018 = list_month[9:12]
  prev = 0

  for year in list_year:
    for month in list_month:
      curr_df = df[(df["year"] == year) & (df["month"] == month)]

      # calc values: "total", "growth"
      month_total = curr_df[col].nunique()
      if (((year == 2016) & (month in skip_2016))  | ((year == 2018) & (month in skip_2018))):
        continue
      elif prev == 0:
        growth =  round(((month_total - 0.01) /0.01) * 100, 2)
      else:
        growth = round(((month_total- prev) / prev) * 100, 2)

      # change values
      prev = month_total
      new_record = {"year": year, "month": month, "total": month_total, "growth":growth}
      df_res = pd.concat([df_res, pd.DataFrame([new_record])], ignore_index=True)

    #add new record
    df_res["growth"] = pd.to_numeric(df_res["growth"], errors='coerce')
    df_res["total"] = pd.to_numeric(df_res["total"], errors='coerce')

  return df_res


def revenue_growth(df, col1, col2,  my_list):
  """Function calcs the revenue per period.
  param df: selected dataframe.
  param col1: selected col from df ("price")
  param col2: selected col from df ("freight_value")
  my_list: cols from df to be grouped by.
  return: new data frame with revenue per month.
  """
  # calc order value per cols (month/quarter/year)
  df["total_price"] = df[col1] + df[col2]
  revenue_per_period = df.groupby(my_list)["total_price"].sum().reset_index()
  return revenue_per_period


def delivery_times(df, my_list):
  """Function calcs the changes in delivery time.
  param df: selected dataframe.
  param my_list: period of time.
  return: df that display the changes in delivery time.
  """
  # calc the time that passed for every record
  df["est_delivery_time"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"])
  df["delivery_time"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"])

  # calc the avg time that passed for every period of times
  df_res = df.groupby(my_list)[["delivery_time", "est_delivery_time"]].mean().reset_index()

  # convert to int
  df_res["est_delivery_time"] = df_res["est_delivery_time"].dt.days.astype(int)
  df_res["delivery_time"] = df_res["delivery_time"].dt.days.astype(int)
  return df_res


def ranking (df, col , my_list):
  """Function calcs the change in ranking over time.
  param df: selected dataframe.
  param col: selected col ("review_score")
  param my_list: period of time.
  return: df with ranks in each period of time.
  """
  # calc ranking per cols (month/quarter/year)
  df_rank = df.groupby(my_list)[col].mean().reset_index(name="avg_score")
  df_rank = df_rank.sort_values(by=my_list)
  return df_rank


def create_col_date(df):
  """ Function create col "date" for given df.
  param df: selected df.
  return: df with additional col "date".
  """
  df ["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%B")
  df["date"] = df["date"].dt.strftime("%Y-%m-%d")
  df = df.sort_values(by="date")
  return df


def best_performers(df, value, period, col1, col2, top, my_title, x_label, y_label, my_legend):
  """Function display the best seller and most profitable categories/products/sellers, per quarter.
  param df: selected df.
  param value: "count"/"sum", decides what mathematical operation will be performed on col2.
  param period: the time period according to which the calculation is performed.
  param col1: selected col form df.
  param col2: selected col form df.
  param top: the number of top values.
  parm title: graph title.
  parm x_label: X axis title.
  parm y_label: Y axis title.
  param my_legends: legend title.
  return: none.
  """
  if value:
    # group df and count values per group
    df_res = df.groupby([period, col1])[col2].count().reset_index()
  else:
    # group df and sum values per group
    df_res = df.groupby([period, col1])[col2].sum().reset_index()

  # rank categories for each period
  df_res["rank"] = df_res.groupby(period)[col2].rank(ascending=False, method="dense")

  # select top categories for each period and create a new df to display it
  df_res = df_res[df_res["rank"] <= top].sort_values([period, "rank"])
  df_res = df_res.drop_duplicates(subset= ["quarter", "rank"])
  df_res_new = df_res.reset_index().pivot(index=period, columns="rank", values=col1)
  df_res_new.columns = [f"{i}th" for i in range(1, top + 1)]
  df_res_new.reset_index(inplace=True)

  # pivot "df_res"
  pivot_df = df_res.pivot(index=period, columns=col1, values=col2)

  # create stacked bar and show "df_res_new"
  display(df_res_new)
  colors = sns.color_palette("husl", len(df_res[col1].unique()))
  ax = pivot_df.plot(kind="bar", stacked=True, figsize=(12, 6), color = colors)
  ax.set_title(my_title)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  if len(df_res[col1].unique()) > 10:
    plt.legend(title=my_legend, bbox_to_anchor=(1, 1), ncol=3)
  else:
    plt.legend(title=my_legend, bbox_to_anchor=(1, 1))
  plt.show()