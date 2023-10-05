from src.data_utility import delete_directory, prepare_data, distribute_classes, train_submodels, ism_post_process, evaluate_ism, mms_post_process, evaluate_mms, find_best_thr
from src import utility_functions
import json
import os
from os.path import join
from datetime import datetime
import sqlite3

def main():
    """
    Main function for running the ISM (Independent Softmax Model) and MMS (MixMaxSim) pipelines.

    This function orchestrates the entire pipeline, including data preparation, submodel training,
    post-processing, and evaluation for both ISM and MMS methods.

    Returns:
        None
    """
    # Load the configuration file    # Load the configuration file
    with open("./config/config.json", "r") as config_file:
        config = json.load(config_file)

    dataset_name = config['dataset_name']
    overwrite = config['overwrite']
    m = config['distance_measure']

    # Loop through different combinations of iterations, n_classes, n_clusters, and methods
    for iter in range(7, 10):
        for n_classes, n_clusters in zip([5000, 7500, 7500, 7500, 8900, 8900, 8900], [4, 2, 3, 4, 2, 3, 4]):
            for method in ["ISM", "MMS"]:
                conn = sqlite3.connect("./results/results.db")
                cursor = conn.cursor()

                scenario = str(n_classes) + '_' + method + str(n_clusters)
                data_scenario_path = join(config[dataset_name]["scenario_embs"], scenario)
                model_scenario_path = join(config[dataset_name]["scenario_submodels"], scenario)
                super_scenario_path = join(config[dataset_name]["scenario_embs"], str(n_classes))

                # Delete existing data and model directories
                delete_directory(data_scenario_path)
                delete_directory(model_scenario_path)

                if method == "ISM":
                    cursor.execute("INSERT INTO results(iteration, n_classes, n_clusters, dataset_name, ism_start_timestamp) \
                                    VALUES (?, ?, ?, ?, ?)", (iter, n_classes, n_clusters, dataset_name, datetime.now()))
                else:
                    cursor.execute("UPDATE results set mms_start_timestamp = ? where iteration = ? and n_classes = ? and n_clusters = ? and dataset_name = ?", (datetime.now(), iter, n_classes, n_clusters, dataset_name))

                conn.commit()
                conn.close()

                utility_functions.pprint(("-------------------------------"), dataset_name)
                utility_functions.pprint(("dataset_name = ", dataset_name), dataset_name)
                utility_functions.pprint(("meth = ", method), dataset_name)
                utility_functions.pprint(("n_classes = ", n_classes), dataset_name)
                utility_functions.pprint(("n_clusters = ", n_clusters), dataset_name)

                # Create necessary directories
                os.makedirs(super_scenario_path, exist_ok=True)
                os.makedirs(data_scenario_path, exist_ok=True)
                os.makedirs(model_scenario_path, exist_ok=True)

                # Prepare data, distribute classes, and train submodels
                trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall = prepare_data(n_classes)
                parts = distribute_classes(method, n_classes, n_clusters, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall)
                train_submodels(method, n_classes, parts, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall)

                if method == "ISM":
                    # Perform ISM post-processing and evaluation on the test set
                    test_softmax_classes = ism_post_process(m, n_classes, parts, testx, 'test')
                    evaluate_ism(iter, m, n_classes, n_clusters, testl, test_softmax_classes)
                else:
                    # Perform MMS post-processing and evaluation on the test set
                    val_sim_classes, val_sim_values, val_sim_softmax, val_softmax_values, val_softmax_sims, val_softmax_classes = mms_post_process(m, n_classes, parts, traincenterx, valx, 'val')
                    thr = find_best_thr(n_classes, vall, val_sim_classes, val_sim_values, val_sim_softmax, val_softmax_values, val_softmax_sims, val_softmax_classes)

                    test_sim_classes, test_sim_values, test_sim_softmax, test_softmax_values, test_softmax_sims, test_softmax_classes = mms_post_process(m, n_classes, parts, traincenterx, testx, 'test')
                    evaluate_mms(iter, thr, testl, n_classes, n_clusters, test_sim_classes, test_sim_values, test_sim_softmax, test_softmax_values, test_softmax_sims, test_softmax_classes)


if __name__ == "__main__":
    main()









