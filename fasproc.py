#!/usr/bin/env python
# coding: utf-8

N_MEDI = 1440 /4

import pandas as pd

from fastai.imports import *
from fastai.structured import *
from fasproc import *



def dropbad(df,threshold=2880):
    dfo=pd.DataFrame()
    bads = []
    for serial, ndf in df.groupby(level=0, as_index=False, group_keys=False):
        #print(serial, len(ndf))
        if len(ndf)>threshold:
            dfo = pd.concat((dfo, ndf), axis=0)
        else:
            bads.append([serial, len(ndf)])
    return dfo, bads

def check_angles(row):
    for angle in row:
        if not( \
               (angle > 115 and angle < 125) \
                or (angle < 235 and angle < 245) \
               ):
            return False
    return True

def dates2(serie):
    for i,date in enumerate(serie):
        if pd.isnull(date):
            #print(i, date)
            serie[i] = serie[i-1] + pd.Timedelta('15 min')
    return serie.copy()

def check_nan(df):
    d=df.dropna()
    return df.shape[0] - d.shape[0]

def check_days_5_serial(df, col, minimum=0):
    summary = df[col].droplevel(0).resample('D').min()
    c=0
    for i in summary:
        if i <= minimum:
            c+=1
    return c

def check_min(serie, minimum = 0, threshold = N_MEDI):
    c=0
    for i in serie:
        if i<=0:
            c+=1
    return c

def check_days_angle_serial(df):
    #daysangle[serial] = []
    summary = df['angcheck'].droplevel(0).resample('D').apply(all)
    c=0
    for i in summary.index:
        if not(summary[i]):
            c+=1
    return c

def check_angles_2(col):
    return len(col) - col.sum()