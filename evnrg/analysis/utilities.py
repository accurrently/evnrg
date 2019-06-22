import pandas as pd
import numba as nb
import numpy as np
import dask

def apply_lambda(
    df: pd.DataFrame, 
    input_col: str,
    output_col: str,
    f: callable):

    df[output_col] = df[input_col].apply(f)

    return df

def sum_cols(df: pd.DataFrame, sname: str):

    out = pd.DataFrame(
        index = df.index,
        data = df.values.sum(axis=1),
        columns = [sname]
    )

    return out

def get_col(df: pd.DataFrame, new_name: str, colname: str):

    out = df[[colname]]

    out.columns = [new_name]

    return out