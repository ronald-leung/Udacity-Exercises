# # In the USWNT data, the columns are not consistent and therefore needs some clean up.
# # One item that needs cleanup is that the column changed in Yellow / Red card in 2019. In 2019, there's one column each, but previously it's one
# # column named Y/R. Same happened for the Caps / Goals column. 
# # Another issue is the "Pts" column. It's only available in 2015 and 2016. I'm just ignoring that column for now.
# # And also the column "Name", became "Player" in 2019. I'm assuming 2019 is the right one to follow, and the column going forward will be that one.

import pandas as pd
import numpy as np

def load_uswntData():
    desired_columns = ['Year', 'Name', 'Pos.', 'GP', 'GS', 'Min', 'G', 'A', 'Y', 'R', 'Caps', 'Goals']
    totalYears = np.arange(2015,2020)
    yr_capsgoals_merged_Years = np.arange(2015,2019)
    pts_AvailableYears = np.arange(2015,2017)
    old_columnNames = np.arange(2015,2019)


    uswnt_df = pd.DataFrame()

    for year in totalYears:
        fileName = str(year) + " USWNT Stats.csv"
        df = pd.read_csv(fileName)
        print("Year ", year , " columns = ", df.columns)
        df["Year"] = year

        if year in yr_capsgoals_merged_Years:
            df["Y"] = df["Y/R"].apply(lambda x: x.split("/")[0])
            df["R"] = df["Y/R"].apply(lambda x: x.split("/")[1] if (len(x.split("/")) > 1)  else "0")
            df["Caps"] = df["Caps/Goals"].apply(lambda x: x.split("/")[0])
            df["Goals"] = df["Caps/Goals"].apply(lambda x: x.split("/")[1])  
            df = df.drop("Y/R", axis=1)
            df = df.drop("Caps/Goals", axis=1)

        if year in pts_AvailableYears:
            df = df.drop("Pts", axis=1)

        if year in old_columnNames:
            df.rename(columns={'Player':'Name',
                              'Pos':'Pos.',
                              'Min.':'Min'}, 
                     inplace=True)

        df = df[desired_columns]
        uswnt_df = pd.concat([uswnt_df, df])
    
    return uswnt_df

