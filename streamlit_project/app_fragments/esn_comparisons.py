"""A python file for functions that are used to compare different ESNs"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def compare_esn_parameters(different_esn_parameters: dict[str, Any],
                           mode: str = "difference"
                           ) -> None | pd.DataFrame:
    """Compare a dict of parameters dicts.

    Args:
        different_esn_parameters: A dictionary containing for each key (the esn name) a dictionary
                                  for its parameters.
        mode: Either "difference" or "all". If "difference" the output df contains only parameters
              that are not the same for all esns. If "all" all build args are outputted.

    Returns:
        A pandas DataFrame containing the index "Parameters", and for each esn_key a column.
    """
    if len(different_esn_parameters) == 0:
        return None
    dep = different_esn_parameters
    for i, (key, val) in enumerate(dep.items()):
        pd_dict = {"Parameters": list(val.keys()), key: [str(x) for x in list(val.values())]}
        df = pd.DataFrame.from_dict(pd_dict).set_index("Parameters")
        if i == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], axis=1, join="outer")

    if mode == "difference":
        x = df_all.index[~df_all.eq(df_all.iloc[:, 0], axis=0).all(1)]
        df_all = df_all.loc[x]
    elif mode == "all":
        pass
    else:
        raise ValueError("mode must be either \"difference\" or \"all\".")

    return df_all
