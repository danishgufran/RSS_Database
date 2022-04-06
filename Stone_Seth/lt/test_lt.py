"""
Tests for long term KNN
"""

import numpy as np
import pandas as pd
from lt import LTKNN
import logging as lg
from uji import UJI


def test_random():
    """test with random data
    intended to check if LTKNN executes without errors
    this is not a test for fucntional correctness

    :returns: passing
    :rtype: bool
    """

    num_samples = 100
    num_cols = 4
    x = np.random.rand(num_samples, num_cols) * 100
    y = np.array(range(num_samples)).reshape((-1, 1))

    waps = [f"WAP_{x}" for x in range(num_cols)]
    lbl = ["LABEL"]

    data = np.hstack((x, y))
    data = pd.DataFrame(data, columns=waps+lbl)

    # init LTKNN
    lt = LTKNN(data, waps, lbl)

    #
    lg.debug(lt.__dict__)

    # predict some labels
    fp = pd.DataFrame(np.random.rand(2, 4) * 100, columns=waps)
    pred_labels_o = lt.predict(fp)
    lg.info(pred_labels_o)

    # assume that some time has passed
    # some APs have disappeared
    new_waps = np.random.choice(waps, size=len(waps)-1, replace=False)
    lg.info(f"previous waps: {lt.original_waps}")
    lt.update(new_waps)
    lg.info(f"new waps:      {list(lt.current_waps)}")

    # predict again
    # predict some labels
    pred_labels_n = lt.predict(fp)
    lg.info(pred_labels_n)

    # as the fp has changed slightly after the update
    # the new prediction should be different
    return set(pred_labels_o) != set(pred_labels_n)


def test_uji():
    """test the LTKNN class with real data

    :returns: passing
    :rtype: bool

    """

    errors = []

    m1 = UJI("test", 1)
    walk_range = list(range(1, 35, 1))
    data = m1.filter_record(FLOOR=3,
                            LABEL=walk_range,
                            COLLECTION_INSTANCE=4)

    train, test = UJI.split_frame(data, split=(80, 20))

    train_waps = m1.get_visible_waps()
    target_label = ["LABEL"]

    # init LTKNN
    lt = LTKNN(train[[*train_waps, *target_label]],
               train_waps,
               target_label,
               nn=3)

    pred_labels = lt.predict(test[train_waps].values)
    pred_coords = m1.labels_to_coords(pred_labels)
    true_coords = m1.labels_to_coords(list(test[target_label].values))

    errors.append(np.mean(UJI.compute_distances(pred_coords, true_coords)))

    lg.info(f"""month 1 error: {errors[0]: .2f} 
    missing waps: {len(lt.missing_waps)}""")

    #
    lg.debug(lt.__dict__)

    for month in range(2, 16):

        # predict some labels
        m = UJI("test", month)
        data = m.filter_record(FLOOR=3,
                               LABEL=walk_range,
                               COLLECTION_INSTANCE=4)

        _, data = UJI.split_frame(data, split=(80, 20))

        vw = m.get_visible_waps()
        lt.update(vw)

        # predict again
        # predict some labels
        fp = data[train_waps]
        pred_labels = lt.predict(fp)
        pred_coords = m.labels_to_coords(pred_labels)
        true_coords = m.labels_to_coords(list(data[target_label].values))

        errors.append(np.mean(UJI.compute_distances(pred_coords, true_coords)))

        # error save
        lg.info(f"""month {month} error: {errors[-1]: .2f}
        missing waps: {len(lt.missing_waps)}""")

    return True


if __name__ == "__main__":
    lg.basicConfig(format="", level=lg.INFO)

    # # run a test with random data
    # try:
    #     is_passed = test_random()
    # except Exception as e:
    #     is_passed = False
    #     print("test failed due to exception\n", e.print_stack_trace())

    # print("random data test passing: ", is_passed)

    # run with real data
    try:
        is_passed = test_uji()
    except Exception as e:
        is_passed = False
        print(
            "test failed due to exception\n",
            e
        )
        raise e

    print("UJI dataset test passing: ", is_passed)
