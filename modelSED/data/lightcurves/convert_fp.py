#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
import pandas as pd

infile = "fp.csv"
infile_with_other = "full_lc_no_fp.csv"
infile_alert = "alert.csv"

df = pd.read_csv(infile)
df_other = pd.read_csv(infile_with_other)
df_alert = pd.read_csv(infile_alert)

df = df.query("magpsf != 99")
print(f"available FP datapoints: {len(df)}")
df = df.reset_index()
df.drop(columns = ["Unnamed: 0", "index"], inplace=True)
df.replace({"ZTF_r": "r", "ZTF_i": "i", "ZTF_g": "g"}, inplace=True)
df.rename(columns={"band": "filter"}, inplace=True)
df["alert"] = False

df_alert_fids = df_alert["fid"].values
filters = []
for fid in df_alert_fids:
    if fid == 1:
        filters.append("g")
    if fid == 2:
        filters.append("r")
    if fid == 3:
        filters.append("i")

df_alert["filter"] = filters
df_alert.drop(columns = ["fid"], inplace=True)
df_alert.rename(columns={"mag": "magpsf", "mag_err": "sigmamagpsf", "obsmjd": "mjd"}, inplace=True)
df_alert.drop(columns = ["Unnamed: 0"], inplace=True)
df_alert["instrument"] = "P48+ZTF"
df_alert["alert"] = True
print(f"available alert datapoints: {len(df_alert)}")


df_other.query("instrument != 'P48+ZTF'", inplace=True)
df_other.drop(columns = ["index"], inplace=True)
df_other["alert"] = True
# print(df_other)

df_final1 = pd.concat([df, df_alert])
df_final = pd.concat([df_final1, df_other])

df_final = df_final.reset_index()
df_final.drop(columns = ["index"], inplace=True)

# print(df_final)

df_final.to_csv("full_lc_fp.csv")