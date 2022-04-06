"""
A common evaluation that works for any dataset

The data is set up to work for heterogeneity and temporal variation
"""
import matplotlib.pyplot as plt
from operator import index
import numpy as np
from tqdm.auto import tqdm
from helpers import compute_distances, label2coords_builder
from Seth import Devices, Floorplan, fetch_seth
from RecordKeeper import RecordKeeper
from frameworks import (knn_train, knn_predict, lt_knn_train, lt_knn_predict, stone_test,
                        stone_train, attention_train)


def evaluate_all_devices(
    floorplan,
    train_ci=0,
    test_cis=range(1, 12),
    checkpoint=True,
    checkpoint_path="saved_result_data/saved_records/heterogeneity_results",
    frameworks=["KNN", "LTKNN", "stone2", "paris"],
):
    # build label 2 coordinate dict for error computation
    rps = Floorplan().get_coords(floorplan)[["label", "x", "y"]].values
    lbl2cords = label2coords_builder(rps, scale=Floorplan().get_scale(floorplan))
    # setup record keeper
    checkpoint_path = checkpoint_path + f"_{floorplan}"
    record_keeper = RecordKeeper(columns=["framework", "train_dev", "test_dev", "ci", "error"],
                                 checkpoint_path=checkpoint_path,
                                 auto_checkpoint=checkpoint)

    # for each train device
    for train_device in tqdm(Devices.devices, desc="Train"):

        # fetch training data
        train_df, _ = fetch_seth(train_device, floorplan, train_ci)

        ################
        # train models #
        ################
        # KNN
        if "KNN" in frameworks:
            knn, knn_train_waps = knn_train(train_df)
        # LT-KNN
        if "LTKNN" in frameworks:
            ltknn = lt_knn_train(train_df)
        if "stone2" in frameworks:
            stone2_encoder, stone2_predictor, stone2_train_waps = stone_train(
                train_df,
                nn=2,
                encoder_path=f"stone/saved_encoders/stone2_seth_{floorplan}_{train_device}_ci0",
            )
        if "paris" in frameworks:
            paris_encoder, paris_predictor, paris_train_waps = stone_train(
                train_df,
                nn=2,
                encoder_path=f"stone/saved_encoders/paris_seth_{floorplan}_{train_device}_ci0",
            )
        ###############
        # for each test device
        for test_device in tqdm(Devices.devices, desc="Test", leave=False):

            # for each collection instance
            for ci in tqdm(test_cis, desc="CI", leave=False):

                # TODO: Augment
                test_df, _ = fetch_seth(test_device, floorplan, ci)

                ###############
                # test models #
                ###############
                # KNN
                if "KNN" in frameworks:
                    pred_y = knn_predict(knn, test_df, knn_train_waps)
                    pred_y = np.array(pred_y).flatten()
                    test_y = test_df["label"].values.flatten()
                    err = compute_distances(test_y, pred_y, lbl2cords=lbl2cords).mean()

                    record_keeper.insert_record(framework="KNN",
                                                train_dev=train_device,
                                                test_dev=test_device,
                                                ci=ci,
                                                error=err)

                # LT-KNN
                if "LTKNN" in frameworks:
                    pred_y = lt_knn_predict(ltknn, test_df)
                    pred_y = np.array(pred_y).flatten()
                    test_y = test_df["label"].values.flatten()
                    err = compute_distances(test_y, pred_y, lbl2cords=lbl2cords).mean()
                    record_keeper.insert_record(framework="LTKNN",
                                                train_dev=train_device,
                                                test_dev=test_device,
                                                ci=ci,
                                                error=err)

                # stone
                if "stone2" in frameworks:
                    pred_y = stone_test(test_df, stone2_encoder, stone2_predictor,
                                        stone2_train_waps)
                    pred_y = np.array(pred_y).flatten()
                    test_y = test_df["label"].values.flatten()
                    err = compute_distances(test_y, pred_y, lbl2cords=lbl2cords).mean()
                    record_keeper.insert_record(framework="stone2",
                                                train_dev=train_device,
                                                test_dev=test_device,
                                                ci=ci,
                                                error=err)
                # paris
                if "paris" in frameworks:
                    pred_y = stone_test(test_df, paris_encoder, paris_predictor, paris_train_waps)
                    pred_y = np.array(pred_y).flatten()
                    test_y = test_df["label"].values.flatten()
                    err = compute_distances(test_y, pred_y, lbl2cords=lbl2cords).mean()
                    record_keeper.insert_record(framework="paris",
                                                train_dev=train_device,
                                                test_dev=test_device,
                                                ci=ci,
                                                error=err)
                ###############

    return record_keeper


def train_all_paris(floor, train_device):
    from frameworks import paris_train_model

    for ci in range(16):

        print(train_device, ci)

        save_path = f"paris/saved_encoders/paris_mha_{floor}_{train_device}_ci{ci}"

        triplet_encoder, predictor, train_waps = paris_train_model(
            train_device,
            floor,
            ci=ci,
            model_save_path=save_path,
        )


def main():
    # nr = evaluate_all_devices(floorplan=Floorplan.BASEMENT,
    #                               test_cis=range(1, 12),
    #                               checkpoint=False,
    #                               frameworks=["stone2", "paris"]).record

    # load_path = f"saved_result_data/saved_records/heterogeneity_results_{Floorplan.BASEMENT}.csv"
    # saved_rk = RecordKeeper.load_RK(
    #     load_path,
    #     auto_checkpoint=False,
    # )

    # from plotter import plot_framework_heatmaps

    # plot_framework_heatmaps(saved_rk.record)
    train_df, _ = fetch_seth(Devices.lg, Floorplan.OFFICE, 0)
    #print(train_df.head())
    print("##########################")
    stone2_encoder, stone2_predictor, stone2_train_waps = attention_train(
                train_df,
                nn=2,
                encoder_path=None,
            )

        # build label 2 coordinate dict for error computation
    rps = Floorplan().get_coords(Floorplan.OFFICE)[["label", "x", "y"]].values
    lbl2cords = label2coords_builder(rps, scale=Floorplan().get_scale(Floorplan.OFFICE))
    
    
    final_lg = []
    final_blu = []
    # for each collection instance

    # Test for LG
    for ci in tqdm(range(0,15), desc="CI", leave=False):

        # TODO: Augment
        test_df, _ = fetch_seth(Devices.lg, Floorplan.OFFICE, ci)
        pred_y = stone_test(test_df, stone2_encoder, stone2_predictor,
                                            stone2_train_waps)

        pred_y = np.array(pred_y).flatten()
        test_y = test_df["label"].values.flatten()
        err = compute_distances(test_y, pred_y, lbl2cords=lbl2cords).mean()

        print(f"mean error :{err} for ci : {ci}")
        final_lg.append(err)
    

    print(f"Final list of error -------> {final_lg}")


    # test for BLU
    for ci in tqdm(range(0,15), desc="CI", leave=False):

        # TODO: Augment
        test_df, _ = fetch_seth(Devices.blu, Floorplan.OFFICE, ci)
        pred_y = stone_test(test_df, stone2_encoder, stone2_predictor,
                                            stone2_train_waps)

        pred_y = np.array(pred_y).flatten()
        test_y = test_df["label"].values.flatten()
        err = compute_distances(test_y, pred_y, lbl2cords=lbl2cords).mean()

        print(f"mean error :{err} for ci : {ci}")
        final_blu.append(err)


    print(f"Final list of error -------> {final_blu}")


    plt.plot(final_lg, color='r', label='LG OFFICE')
    plt.plot(final_blu, color='g', label='Blu OFFICE')
    plt.legend()
    plt.xlabel('CI Values',color="red", fontweight='bold')
    plt.ylabel('Meters',color="red")
    plt.title(" Attention Trained on LG - OFFICE",color="red", fontweight='bold')
    plt.savefig('Attention_LG_train_OFFICE_lg_blu.png')
    plt.show()



    # record_keeper.insert_record(framework="stone2",
    #                             train_dev=train_device,
    #                             test_dev=test_device,
    #                             ci=ci,
    #                             error=err)


if __name__ == "__main__":
    main()
    #train_all_paris(Floorplan.BASEMENT, Devices.lg)
