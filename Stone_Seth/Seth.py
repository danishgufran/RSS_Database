"""
Class to access data
"""
try:
    from Mapping.Floorplan import Floorplan
except:
    from .Mapping.Floorplan import Floorplan
from typing import Iterable
from matplotlib import pyplot as plt
import os
import pandas as pd
import re
import numpy as np
from typing import List, Tuple


class Devices:
    """
    Device names as given under raw folder
    """
    blu = "BLU"
    htc = "HTC"
    lg = "LG"
    moto = "MOTO"
    op3 = "OP3"
    s7 = "S7"
    devices = ["S7", "BLU", "HTC", "LG", "MOTO", "OP3"]


MAC_RE = ("^([0-9A-Fa-f]{2}[:-])" + "{5}([0-9A-Fa-f]{2})|" + "([0-9a-fA-F]{4}\\." +
          "[0-9a-fA-F]{4}\\." + "[0-9a-fA-F]{4})$")


def is_valid_mac_id(str):

    # Regex to check valid
    # MAC address

    # Compile the ReGex
    p = re.compile(MAC_RE)

    # If the string is empty
    # return false
    if (str == None):
        return False

    # Return if the string
    # matched the ReGex
    if (re.search(p, str)):
        return True
    else:
        return False


def get_mac_ids(labels: list):
    macs = []
    for lbl in labels:
        if is_valid_mac_id(lbl):
            macs.append(lbl)
    return macs


def fetch_seth_original(device, floor, ci, base_path="seth/temp/clean"):

    if ci < 10:
        csv_path = os.path.join(base_path, device, f"{floor}{ci}")
        return pd.read_csv(csv_path + ".csv"), pd.read_csv(csv_path + "_meta.csv")
    else:
        try:
            if "seth" in base_path:
                base_path = "seth/RamLocSelect" + "/temp"
            else:
                base_path = "RamLocSelect" + "/temp"
            csv_path = os.path.join(base_path, device, f"{floor}{ci}")
            return pd.read_csv(csv_path + ".csv"), None
        except FileNotFoundError:
            # Maybe data was interpolated
            if "seth" in base_path:
                base_path = "seth/RamLocSelect" + "/interpolate"
            else:
                base_path = "RamLocSelect" + "/interpolate"
            csv_path = os.path.join(base_path, device, f"{floor}{ci}")
            return pd.read_csv(csv_path + ".csv"), None


def fetch_seth(device,
               floor,
               ci,
               base_path="temp/clean",
               ap_drop_rate=None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:

    # Create a mapping between CI and ap_drop_rate
    if ci == 12:
        ap_drop_rate = 0.05
    elif ci == 13:
        ap_drop_rate = 0.10
    elif ci == 14:
        ap_drop_rate = 0.15
    elif ci == 15:
        ap_drop_rate = 0.20
    elif ci < 12 and ci >= 0:
        ap_drop_rate = None
    else:
        print("Invalid ci value. Only allowed up to 15", ci, ap_drop_rate)
        exit("ERROR")

    # if collected data goes beyond
    if ci > 11:
        # we will augment this data to represent distinct effects
        ci = 9  # <= this is the last known stable collected data
        df, meta = fetch_seth_original(device, floor, ci, base_path=base_path)
    else:
        # for real collected data
        # current CI ranges from [0-11]
        df, meta = fetch_seth_original(device, floor, ci, base_path=base_path)

    if ap_drop_rate is None:
        return df, meta
    elif type(ap_drop_rate) == float:
        return drop_ap_random(df, ap_drop_rate), meta
    else:
        print(" Invalid ap_drop_rate provided")
        return None, None


def fetch_seth_heavy(device, floor, ci, base_path="seth/temp/clean", ap_drop_rate=None):

    # Create a mapping between CI and ap_drop_rate
    if 12 <= ci <= 14 and ap_drop_rate is None:
        ap_drop_rate = 0.10
    elif 15 <= ci <= 17 and ap_drop_rate is None:
        ap_drop_rate = 0.20
    elif 18 <= ci <= 20 and ap_drop_rate is None:
        ap_drop_rate = 0.40
    elif 21 <= ci <= 23 and ap_drop_rate is None:
        ap_drop_rate = 0.60
    elif ci < 12:
        pass
    else:
        print("Invalid ci value. Only allowed up to 23", ci, ap_drop_rate)
        exit("ERROR")

    # if collected data goes beyond
    if ci > 11:
        # we will augment this data to represent distinct effects
        ci = 9  # <= this is the last known stable collected data
        df, meta = fetch_seth_original(device, floor, ci, base_path=base_path)
    else:
        # for real collected data
        # current CI ranges from [0-11]
        df, meta = fetch_seth_original(device, floor, ci, base_path=base_path)

    if ap_drop_rate is None:
        return df, meta
    elif type(ap_drop_rate) == float:
        return drop_ap_random(df, ap_drop_rate), meta
    else:
        print(" Invalid ap_drop_rate provided")
        return None, None


def drop_ap_random(df, rate, missing=-100):
    """Drop a certain ratio of APs
    """

    macs = get_mac_ids(df.columns)

    drop_macs = np.random.choice(macs, replace=False, size=int(len(macs) * rate))

    df = df.drop(drop_macs, axis=1)

    return df


def make_ephimeral_df(device, floor, cis=range(10), missing_marker=-100):

    # get all macs observed on a path
    all_macs = []

    for i in cis:

        df, _ = fetch_seth_heavy(device, floor, i)

        this_macs = list(get_mac_ids(df.columns))

        all_macs.extend(this_macs)

    # keep only single copy of each mac
    all_macs = sorted(list(set(all_macs)))
    # empty array with all_macs columns
    eph_mat = np.zeros((0, len(all_macs)))

    # iter over data again
    for i in cis:

        # dataframe
        df, _ = fetch_seth_heavy(device, floor, i)

        # APs seen in this df
        this_macs = list(get_mac_ids(df.columns))

        # add any missing
        missing_macs = list(set(all_macs) - set(this_macs))
        df[missing_macs] = missing_marker

        # create a row to
        eph_mask = df[all_macs].max().values != -100

        # vstack mask
        eph_mat = np.vstack((eph_mat, eph_mask))

    return pd.DataFrame(eph_mat, columns=all_macs, dtype=bool)


def plot_ephimeral(device,
                   floor,
                   cis=range(10),
                   missing_marker=-100,
                   show=False,
                   eph_df=None,
                   figsize=None):

    if eph_df is None:
        eph_df = make_ephimeral_df(device, floor, cis=cis, missing_marker=missing_marker)

    plt.figure(figsize=figsize)
    plt.imshow(eph_df.values, cmap='binary_r', aspect='auto', interpolation='nearest')
    plt.title(f'AP Presence for {device} at {floor}', size=14)
    plt.xlabel('WiFi APs', size=14)
    plt.ylabel('Collection\nInstances', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)

    if show:
        plt.show()

    return plt


def set_rp_interval(df, interval=2, target='label'):

    uniq_lbls = df[target].drop_duplicates().values.reshape((-1, 1))

    kept_labels = list(range(0, uniq_lbls.shape[0], interval))

    kept_df = df[df[target].isin(kept_labels)]

    # make dict of old to new labels
    lbls_dict = dict(zip(kept_labels, list(range(len(kept_labels)))))

    return kept_df.replace({target: lbls_dict})


def aug_eph_df(eph_df, augment=[0.05, 0.10, 0.15, 0.20]):

    base = eph_df.iloc[eph_df.index[-1]]
    for p_aug in augment:

        aug = base.copy()

        seen_inds = np.where(aug.values == True)[0]
        num_off = int(p_aug * seen_inds.shape[0])

        off_inds = np.random.choice(seen_inds, size=num_off, replace=False)

        aug[off_inds] = False

        eph_df = eph_df.append(aug)

    return eph_df


def make_eph_plots():

    path_name = [Floorplan.BASEMENT, "Basement"]
    eph_df = make_ephimeral_df(Devices.lg, path_name[0], cis=range(12))
    eph_df_basement = aug_eph_df(eph_df)

    path_name = [Floorplan.OFFICE, "Office"]
    eph_df = make_ephimeral_df(Devices.lg, path_name[0], cis=range(12))
    eph_df_office = aug_eph_df(eph_df)

    fig, axs = plt.subplots(2, figsize=(8, 4))

    axs[0].imshow(eph_df_basement.values, cmap='binary_r', aspect='auto', interpolation='nearest')
    axs[0].set_xlabel('Basement WiFi APs', size=14)
    axs[0].set_ylabel('Collection\nInstances', size=14)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].tick_params(axis='y', labelsize=14)

    axs[1].imshow(eph_df_office.values, cmap='binary_r', aspect='auto', interpolation='nearest')
    axs[1].set_xlabel('Office WiFi APs', size=14)
    axs[1].set_ylabel('Collection\nInstances', size=14)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=14)

    # plt.title(f'AP Presence for {device} at {floor}', size=14)
    # plt.xlabel('WiFi APs', size=14)
    # plt.ylabel('Collection\nInstances', size=14)
    # plt.xticks(size=14)
    # plt.yticks(size=14)

    plt.tight_layout()
    plt.savefig(f"seth/plots/ephimeral/eph_lg_paths.png")


if __name__ == "__main__":

    make_eph_plots()
    # path_name = [Floorplan.BASEMENT, "Basement"]

    # eph_df = make_ephimeral_df(Devices.lg, path_name[0], cis=range(12))

    # eph_df = aug_eph_df(eph_df)

    # plot = plot_ephimeral(Devices.lg, path_name[1],
    #                      figsize=(8, 2.4),
    #                       show=False, eph_df=eph_df)
    # plot.tight_layout()
    # plot.savefig(f"seth/plots/ephimeral/eph_lg_{path_name[1]}.png")
