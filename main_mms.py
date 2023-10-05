from src.data_utility import delete_directory, prepare_data, distribute_classes, train_submodels, mms_post_process, evaluate_mms, find_best_thr
from src import utility_functions
import json
import os
from os.path import join
import sqlite3
from datetime import datetime

def main():
    """
    Main function for running the MMS (MixMaxSim) pipeline.

    This function orchestrates the entire MMS pipeline, including data preparation,
    submodel training, post-processing, threshold finding, and evaluation.

    Returns:
        None
    """
    # Load the configuration file
    with open("./config/config.json", "r") as config_file:
        config = json.load(config_file)

    dataset_name = config['dataset_name']
    overwrite = config['overwrite']
    m = config['distance_measure']
    method = "MMS"

    # Loop through different combinations of n_classes and n_clusters
    # for n_classes, n_clusters in zip([5000, 5000, 7500, 7500, 7500, 8900, 8900, 8900], [3, 4, 2, 3, 4, 2, 3, 4]): #zip([1000, 5000, 6500, 7500, 8000, 8900] , [2, 2, 3, 3, 3, 4]): #, 10000, 20000, 50000], [3,5,10,20]): #zip([1000, 5000, 10000, 20000, 50000], [2,3,5,10,20]):        for iter in range(10):
    for n_classes, n_clusters in zip([5000], [3]): 
        for iter in range(10):
            conn = sqlite3.connect("./results/results.db")
            cursor = conn.cursor()

            cursor.execute("UPDATE results set mms_start_timestamp = ? where iteration = ? and n_classes = ? and n_clusters = ? and dataset_name = ?", (datetime.now(), iter, n_classes, n_clusters, dataset_name))

            conn.commit()
            conn.close()

            scenario = str(n_classes) + '_' + method + str(n_clusters)
            data_scenario_path = join(config[dataset_name]["scenario_embs"], scenario)
            model_scenario_path = join(config[dataset_name]["scenario_submodels"], scenario)
            super_scenario_path = join(config[dataset_name]["scenario_embs"], str(n_classes))

            # Delete existing data and model directories
            delete_directory(data_scenario_path)
            delete_directory(model_scenario_path)

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

            # Perform post-processing, threshold finding, and evaluation on the validation set
            val_sim_classes, val_sim_values, val_sim_softmax, val_softmax_values, val_softmax_sims, val_softmax_classes = mms_post_process(m, n_classes, parts, traincenterx, valx, 'val')
            thr = find_best_thr(n_classes, vall, val_sim_classes, val_sim_values, val_sim_softmax, val_softmax_values, val_softmax_sims, val_softmax_classes)

            # Perform post-processing and evaluation on the test set
            test_sim_classes, test_sim_values, test_sim_softmax, test_softmax_values, test_softmax_sims, test_softmax_classes = mms_post_process(m, n_classes, parts, traincenterx, testx, 'test')
            evaluate_mms(iter, thr, testl, n_classes, n_clusters, test_sim_classes, test_sim_values, test_sim_softmax, test_softmax_values, test_softmax_sims, test_softmax_classes)

if __name__ == "__main__":
    main()