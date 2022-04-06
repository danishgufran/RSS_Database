"""
functions that wrap around seth and uji dataset

This module assumes there are seth and uji sub-modules available
"""

from uji import UJI
import logging as log
import pandas as pd


def fetch_uji(
    month: int,
    floor=3,
    keep=list(range(34)),
    uji_dir="uji/",
    extra_cols=["LABEL", "CORD_X", "CORD_Y"]
):
    """Get data for a select portion of the UJI dataset

    Parameters
    ----------
    month: int
        month in which data was collected

    floor : int, optional
        floor in which data was collected, by default 3

    keep : list, optional
        RPs on the floor to keep, by default list(range(34))

    uji_dir : str, optional
        path to uji directory
        default "uji/"

    extra_cols : list, optional
        additional columns to preserve from uji, 
        by default ["LABEL", "CORD_X", "CORD_Y"]

    Returns
    -------
    pd.DataFrame
        dataframe of fingerprints for UJI
    """

    # could have been as assert statement
    # check if month is valid
    if not 1 <= month <= 15:
        raise ValueError("month should be in range [1, 15]")

    try:

        # make the uji object
        uji = UJI.from_cache(
            "test",
            month,
            cache_dir=uji_dir + "/db_cache"
        )

        # get the dataframe for the required floor
        df = uji.filter_record(FLOOR=floor)
    except:
        uji = UJI(
            "test",
            month,
            DATA_DIR=uji_dir + "/db/"
        )

        # get the dataframe for the required floor
        df = uji.filter_record(FLOOR=floor)

    # only keep select RPs
    if len(keep) > 0:
        df = df[df["LABEL"].isin(keep)]

    # get the visible APs
    visible_aps = list(uji.get_visible_waps(df))

    # dataframe with visible APs and extra cols
    df = df[visible_aps + extra_cols]

    return df


def uji_label_to_coords(
    labels: list,
    month: int = 1,
    uji_dir="uji/"
):
    """Get coodinates from LABEL

    Parameters
    ----------
    labels : list
        list of integer labels
    month : int, optional
        month should be in valid range of [1, 15], default 1
        coordinates do not change by month
    uji_dir : str, optional
        path to uji directory, default "uji/"

    Returns
    -------
    2D numpy array
        coordinates of each label passed
    """
    # could have been as assert statement
    # check if month is valid
    if not 1 <= month <= 15:
        raise ValueError("month should be in range [1, 15]")

    # get UJI object
    try:
        # make the uji object
        uji = UJI.from_cache(
            "test",
            month,
            cache_dir=uji_dir + "/db_cache"
        )
    except:
        uji = UJI(
            "test",
            month,
            DATA_DIR=uji_dir + "/db/"
        )

    # drop floor column and return
    return uji.labels_to_coords(labels)[:, 1:]


def uji_get_aps(columns: list):
    """pick WAPs from column list

    Parameters
    ----------
    columns : list
        columns of dataframe

    Returns
    -------
    list
        list of Waps
    """
    macs = []
    for lbl in columns:
        if "WAP_" in lbl:
            macs.append(lbl)
    return macs


def get_aps_generic(columns):
    macs = []
    for lbl in columns:
        if "WAP_" in lbl or ":" in lbl:
            macs.append(lbl)
    return macs


# TODO:
# fix standard column names for coordinate and label
# Do this for both seth and uji
