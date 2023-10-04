from src import utility_functions, clustering, training
import shutil
import json
import numpy as np
import tensorflow as tf
import os
from os.path import join
from numpy.random import seed
from numpy import unique, where, savez_compressed
import random
import torch
import re
import keras
from tqdm import tqdm
from sklearn.utils.extmath import softmax
from datetime import datetime
import pickle
import itertools
from sklearn.metrics import roc_curve

from scipy.spatial.distance import cdist
from sklearn import metrics
import sqlite3

seed(1)
tf.random.set_seed(2)

def load_config():
    with open("./config/config.json", "r") as config_file:
        config = json.load(config_file)
    return config


def set_weights(testl, n_classes):
    weights = []
    for i in range(n_classes):
        w = (len(np.where(testl == i)[0]))
        for s in range(w):
            weights.append(1/w)
    weights = np.array(weights)
    return weights


def prepare_data(n_classes):
    config = load_config()
    dataset_name = config['dataset_name']
    super_scenario_path = join(config[dataset_name]["scenario_embs"], str(n_classes))

    train_embeddings_path = join(config[dataset_name]["features"], 'train')
    val_embeddings_path = join(config[dataset_name]["features"], 'val')
    test_embeddings_path = join(config[dataset_name]["features"], 'test')

    trainx = []
    trainy = []
    trainl = []

    traincenterx = []
    traincentery = []
    traincenterl = []

    testx = []
    testy = []
    testl = []

    valx = []
    valy = []
    vall = []

    if os.path.isfile(join(super_scenario_path, 'testx.npz')):
        trainx = np.load(join(super_scenario_path, 'trainx.npz'))['res']
        trainy = np.load(join(super_scenario_path, 'trainy.npz'))['res']
        trainl = np.load(join(super_scenario_path, 'trainl.npz'))['res']

        traincenterx = np.load(join(super_scenario_path, 'traincenterx.npz'))['res']
        traincentery = np.load(join(super_scenario_path, 'traincentery.npz'))['res']
        traincenterl = np.load(join(super_scenario_path, 'traincenterl.npz'))['res']

        testx = np.load(join(super_scenario_path, 'testx.npz'))['res']
        testy = np.load(join(super_scenario_path, 'testy.npz'))['res']
        testl = np.load(join(super_scenario_path, 'testl.npz'))['res']

        valx = np.load(join(super_scenario_path, 'valx.npz'))['res']
        valy = np.load(join(super_scenario_path, 'valy.npz'))['res']
        vall = np.load(join(super_scenario_path, 'vall.npz'))['res']
    else:
        all_id_files = dict()
        with open(join(config[dataset_name]['orig_dataset_dir'], dataset_name, 'all_id_files.json')) as jsonfile:
            all_id_files = json.load(jsonfile)

        keys = list(all_id_files.keys())[:n_classes]
        
        os.makedirs(join('..', dataset_name, 'data', str(n_classes)), exist_ok=True)

        idx = 0
        for class_name in tqdm(keys):
            tr_x = np.load(join(train_embeddings_path, class_name + '.npz'), allow_pickle=True)
            tr_f = 20 # len(tr_features)
            tr_features = tr_x[tr_x.files[0]][:tr_f]    
            trainx.extend(tr_features)
            trainy.extend([class_name] for t in range(tr_f))
            trainl.extend([idx] for t in range(tr_f))

            traincenterx.append(np.mean(tr_features, axis=0))
            traincenterl.append(idx)
            traincentery.append(class_name)

            te_x = np.load(join(test_embeddings_path, class_name + '.npz'), allow_pickle=True)
            te_f = 5 # len(te_features)
            te_features = te_x[te_x.files[0]][:te_f]
            testx.extend(te_features)
            testl.extend([idx] for t in range(te_f))
            testy.extend([class_name] for t in range(te_f))

            v_x = np.load(join(val_embeddings_path, class_name + '.npz'), allow_pickle=True)
            v_f = 5 # len(v_features)
            v_features = v_x[v_x.files[0]][:v_f]
            valx.extend(v_features)
            vall.extend([idx] for t in range(v_f))
            valy.extend([class_name] for t in range(v_f))

            idx+=1

        trainx = np.array(trainx)
        trainl = np.array(trainl)
        trainy = np.array(trainy)

        traincenterx = np.array(traincenterx)
        traincenterl = np.array(traincenterl)
        traincentery = np.array(traincentery)

        testx = np.array(testx)
        testl = np.array(testl)
        testy = np.array(testy)

        valx = np.array(valx)
        vall = np.array(vall)
        valy = np.array(valy)

        # TODO: remove later
        trainl = trainl.squeeze()
        testl = testl.squeeze()
        vall = vall.squeeze()
        trainy = trainy.squeeze()
        testy = testy.squeeze()
        valy = valy.squeeze()

        savez_compressed(join(super_scenario_path, 'trainx.npz'), res=trainx)
        savez_compressed(join(super_scenario_path, 'trainy.npz'), res=trainy)
        savez_compressed(join(super_scenario_path, 'trainl.npz'), res=trainl)

        savez_compressed(join(super_scenario_path, 'traincenterx.npz'), res=traincenterx)
        savez_compressed(join(super_scenario_path, 'traincentery.npz'), res=traincentery)
        savez_compressed(join(super_scenario_path, 'traincenterl.npz'), res=traincenterl)

        savez_compressed(join(super_scenario_path, 'testx.npz'), res=testx)
        savez_compressed(join(super_scenario_path, 'testy.npz'), res=testy)
        savez_compressed(join(super_scenario_path, 'testl.npz'), res=testl)

        savez_compressed(join(super_scenario_path, 'valx.npz'), res=valx)
        savez_compressed(join(super_scenario_path, 'valy.npz'), res=valy)
        savez_compressed(join(super_scenario_path, 'vall.npz'), res=vall)    

    return trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall


def distribute_classes(method, n_classes, n_clusters, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall):    
    config = load_config()
    dataset_name = config['dataset_name']
    scenario = str(n_classes) + '_' + method + str(n_clusters)
    data_scenario_path = join(config[dataset_name]["scenario_embs"], scenario)
    model_scenario_path = join(config[dataset_name]["scenario_submodels"], scenario)
    distribution_path = join(data_scenario_path, 'parts.npz')
    m = config['distance_measure']
    parts = dict()

    if method == 'ISM':   
        if config['overwrite'] == False and os.path.isfile(distribution_path):
            parts = (np.load(distribution_path, allow_pickle=True)['res']).item()
        else:
            # # random 
            lbls = np.zeros((n_classes), dtype='int')
            for i in range(n_classes):
                lbls[i] = random.randint(0, n_clusters-1)
            for i in range(n_clusters):
                # parts[i] = torch.Tensor((lbls == i).nonzero()).squeeze(0).unsqueeze(1).cuda().int().cpu().numpy().squeeze(1)
                parts[i] = torch.Tensor((lbls == i).nonzero()).flatten().int().numpy()
            savez_compressed(distribution_path, res = parts)
    
    elif method == "MMS":
        clustering_model_filename = join(model_scenario_path, 'kmeans.sav')

        if config['overwrite'] == False and os.path.isfile(distribution_path) and os.path.isfile(clustering_model_filename):
            kmeans_model = pickle.load(open(clustering_model_filename, 'rb'))
            parts = (np.load(distribution_path, allow_pickle=True)['res']).item()
        else:
            centers, labels = clustering.init_centers(trainx, n_clusters) # IMPORTANT TODO: dont forget to cite the reference
            kmeans_model = clustering.Fast_KMeans(n_clusters=n_clusters, max_iter=100, tol=0.0001, verbose=0, centroids=centers, mode=m, minibatch=None)
            lbls = kmeans_model.fit_predict(torch.Tensor(traincenterx).cuda())
            pickle.dump(kmeans_model, open(clustering_model_filename, 'wb'))
            for i in range(n_clusters):
                parts[i] = (lbls == i).nonzero().cpu().numpy().squeeze(1)
            savez_compressed(distribution_path, res = parts)

    return parts

# def cluster_data(method, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall):
#     distribution_path = join(data_scenario_path, 'parts.npz')
#     parts = dict()

#     if method == 'ISM':   
#         if config['overwrite'] == False and os.path.isfile(distribution_path):
#             parts = (np.load(distribution_path, allow_pickle=True)['res']).item()
#         else:
#             # # random 
#             lbls = np.zeros((n_classes), dtype='int')
#             for i in range(n_classes):
#                 lbls[i] = random.randint(0, n_clusters-1)
#             for i in range(n_clusters):
#                 # parts[i] = torch.Tensor((lbls == i).nonzero()).squeeze(0).unsqueeze(1).cuda().int().cpu().numpy().squeeze(1)
#                 parts[i] = torch.Tensor((lbls == i).nonzero()).flatten().int().numpy()
#             savez_compressed(distribution_path, res = parts)
    
#     elif method == "MMS":
#         clustering_model_filename = join(model_scenario_path, 'kmeans.sav')

#         if config['overwrite'] == False and os.path.isfile(distribution_path) and os.path.isfile(clustering_model_filename):
#             kmeans_model = pickle.load(open(clustering_model_filename, 'rb'))
#             parts = (np.load(distribution_path, allow_pickle=True)['res']).item()
#         else:
#             centers, labels = clustering.init_centers(trainx, n_clusters) # IMPORTANT TODO: dont forget to cite the reference
#             kmeans_model = clustering.Fast_KMeans(n_clusters=n_clusters, max_iter=100, tol=0.0001, verbose=0, centroids=centers, mode=m, minibatch=None)
#             lbls = kmeans_model.fit_predict(torch.Tensor(traincenterx).cuda())
#             pickle.dump(kmeans_model, open(clustering_model_filename, 'wb'))
#             for i in range(n_clusters):
#                 parts[i] = (lbls == i).nonzero().cpu().numpy().squeeze(1)
#             savez_compressed(distribution_path, res = parts)

#     return parts


def train_submodels(method, n_classes, parts, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall):
    config = load_config()
    dataset_name = config['dataset_name']
    dataset_path = config[dataset_name]["features"]
    
    n_clusters = len(parts)
    scenario = str(n_classes) + '_' + method + str(n_clusters)
    data_scenario_path = join(config[dataset_name]["scenario_embs"], scenario)
    model_scenario_path = join(config[dataset_name]["scenario_submodels"], scenario)

    for r in range(len(parts)):
        utility_functions.pprint(('cluster', str(r)), config[dataset_name])
        train_ids = utility_functions.train_calc_ids(parts[r], trainy, trainl, r)
        test_ids = utility_functions.test_calc_ids(parts[r], trainy, trainl, r)
        train_sample_count, test_sample_count = utility_functions.convert_emb_to_tfrecord(dataset_name, data_scenario_path, dataset_path, train_ids, test_ids, r, overwrite=False, all_samples=True, n_classes=len(parts[r]))
        train_dataset, test_dataset = utility_functions.prepare_data_sets(dataset_name, data_scenario_path, train_ids, test_ids, r)    
        model_scenario_path = join(config[dataset_name]["scenario_submodels"], scenario)
        training.softmax_train(config[dataset_name], model_scenario_path, train_dataset, parts[r], trainx, trainl, r, epochs=config['epochs'],train_overwrite=False, freq = train_sample_count // 100)

def evaluate_ism(iter, m, n_classes, n_clusters, testl, test_softmax_classes):
    weights = set_weights(testl, n_classes)
    config = load_config()
    dataset_name = config["dataset_name"]

    max_max_trues = 0
    max_max_falses = 0

    max_max_true_prec = np.zeros(n_classes)
    max_max_false_prec = np.zeros(n_classes)
    max_max_true_recall = np.zeros(n_classes)
    max_max_false_recall = np.zeros(n_classes)

    index = 0
    for test_sample in range(n_classes * 5):
        real_class = testl[test_sample]

        if test_softmax_classes[test_sample] == real_class:
            max_max_trues += weights[test_sample]
            max_max_true_prec[real_class] += weights[test_sample]
            max_max_true_recall[real_class] += weights[test_sample]
        else:
            max_max_falses += weights[test_sample]
            max_max_false_recall[real_class] += weights[test_sample]
            try:
                max_max_false_prec[test_softmax_classes[test_sample].int()] += weights[test_sample]
            except:
                max_max_false_prec[test_softmax_classes[test_sample]] += weights[test_sample]

    # print(max_max_trues / (max_max_trues + max_max_falses))
    max_max_precision_array = np.divide(max_max_true_prec, (max_max_true_prec + max_max_false_prec), out=np.zeros_like(max_max_true_prec), where=(max_max_false_prec + max_max_true_prec)!=0)
    max_max_recall_array = np.divide(max_max_true_recall, (max_max_true_recall + max_max_false_recall), out=np.zeros_like(max_max_true_recall), where=(max_max_false_recall + max_max_true_recall)!=0)

    max_max_f_score_x = 2 * max_max_precision_array * max_max_recall_array
    max_max_f_score_y = max_max_precision_array + max_max_recall_array
    max_max_fscore_array =  np.divide(max_max_f_score_x, max_max_f_score_y, out=np.zeros_like(max_max_f_score_x), where=(max_max_f_score_y)!=0)

    max_max_precision = np.sum(max_max_precision_array) / n_classes
    max_max_recall = np.sum(max_max_recall_array) / n_classes
    max_max_fscore = np.sum(max_max_fscore_array) / n_classes

    max_max_precision, max_max_recall, max_max_fscore

    max_max_report = metrics.classification_report(testl, test_softmax_classes, output_dict=True, zero_division=0)
    utility_functions.pprint((max_max_report['macro avg']), config[dataset_name])

    conn = sqlite3.connect("./results/results.db")
    cursor = conn.cursor()

    cursor.execute("UPDATE results set ism_end_timestamp = ?, ism_recall = ?, ism_precision =?, ism_fscore = ? where iteration = ? and dataset_name= ? and n_classes = ? and n_clusters = ?", (datetime.now(), max_max_report['macro avg']['recall'], max_max_report['macro avg']['precision'], max_max_report['macro avg']['f1-score'], iter, dataset_name, n_classes, n_clusters))

    conn.commit()
    conn.close()

    return max_max_report


# def calc_mms_acc(m):
#     val_sim_classes, val_sim_values, val_sim_softmax, val_softmax_values, val_softmax_sims, val_softmax_classes = mms_post_process(m, 'val')
#     thr = find_best_thr(val_sim_classes, val_sim_values, val_sim_softmax, val_softmax_values, val_softmax_sims, val_softmax_classes)

#     test_sim_classes, test_sim_values, test_sim_softmax, test_softmax_values, test_softmax_sims, test_softmax_classes = mms_post_process(m, 'test')
#     final_report = evaluate_mms(thr, test_sim_classes, test_sim_values, test_sim_softmax, test_softmax_values, test_softmax_sims, test_softmax_classes)


def evaluate_mms(iter, thr, testl, n_classes, n_clusters, sim_classes, sim_values, sim_softmax, softmax_values, softmax_sims, softmax_classes):
    weights = set_weights(testl, n_classes)
    config = load_config()
    dataset_name = config["dataset_name"]
    
    best_thr = thr
    main_preds = np.zeros(n_classes * 5)

    trues = 0
    falses = 0
    try:
        res = (sim_values * (thr / 10) * ((sim_softmax+1)/2)) > (softmax_values * ((np.array(softmax_sims)+1)/2))
    except:
        res = (sim_values * (thr / 10) * ((sim_softmax+1)/2)) > (softmax_values.numpy() * ((np.array(softmax_sims.numpy())+1)/2))
    for idx in range(n_classes * 5):
        if softmax_values[idx] > 0.5:
            main_preds[idx] = softmax_classes[idx]
            if softmax_classes[idx] == testl[idx]:
                trues += weights[idx]
            else:
                falses += weights[idx]
            continue
        if res[idx]:
            main_preds[idx] = sim_classes[idx]
            if sim_classes[idx] == testl[idx]:
                trues += weights[idx]
            else:
                falses += weights[idx]
        else:
            main_preds[idx] = softmax_classes[idx]
            if softmax_classes[idx] == testl[idx]:
                trues += weights[idx]
            else:
                falses += weights[idx]
    
    main_report = (metrics.classification_report(testl, main_preds, output_dict=True, zero_division=0))
    utility_functions.pprint((main_report['macro avg']), config[dataset_name])
    
    conn = sqlite3.connect("./results/results.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE results set mms_end_timestamp = ?, mms_recall = ?, mms_precision =?, mms_fscore = ?, mms_best_thr = ? where iteration = ? and dataset_name= ? and n_classes = ? and n_clusters = ?", (datetime.now(), main_report['macro avg']['recall'], main_report['macro avg']['precision'], main_report['macro avg']['f1-score'], best_thr, iter, dataset_name, n_classes, n_clusters))
    conn.commit()
    conn.close()

    return main_report

def find_best_thr(n_classes, vall, sim_classes, sim_values, sim_softmax, softmax_values, softmax_sims, softmax_classes):
    weights = set_weights(vall, n_classes)
    config = load_config()
    dataset_name = config["dataset_name"]

    true_prec = np.zeros(n_classes)
    false_prec = np.zeros(n_classes)

    true_recall = np.zeros(n_classes)
    false_recall = np.zeros(n_classes)

    best_thr = -1
    best_recall = 0

    main_preds = np.zeros((len(vall), 20))
    for th in range(0, 20):
        trues = 0
        falses = 0
        try:
            res = (sim_values * (th / 10) * ((sim_softmax+1)/2)) > (softmax_values * ((np.array(softmax_sims)+1)/2))
        except:
            res = (sim_values * (th / 10) * ((sim_softmax+1)/2)) > (softmax_values.numpy() * ((np.array(softmax_sims.numpy())+1)/2))
        for idx in range(len(vall)):
            if softmax_values[idx] > 0.5: #0.5:
                main_preds[idx, th] = softmax_classes[idx]
                if softmax_classes[idx] == vall[idx]:
                    true_prec[vall[idx]] += weights[idx]
                    true_recall[vall[idx]] += weights[idx]
                    trues += weights[idx]
                else:
                    falses += weights[idx]
                    try:
                        false_prec[softmax_classes[idx].int()] += weights[idx]
                    except:
                        false_prec[softmax_classes[idx]] += weights[idx]

                    false_recall[vall[idx]] += weights[idx]
                continue
            if res[idx]:
                main_preds[idx, th] = sim_classes[idx]
                if sim_classes[idx] == vall[idx]:
                    trues += weights[idx]
                    true_recall[vall[idx]] += weights[idx]
                    true_prec[vall[idx]] += weights[idx]
                else:
                    falses += weights[idx]
                    false_recall[vall[idx]] += weights[idx]
                    try:
                        false_prec[softmax_classes[idx].int()] += weights[idx]
                    except:
                        false_prec[softmax_classes[idx]] += weights[idx]

            else:
                main_preds[idx, th] = softmax_classes[idx]
                if softmax_classes[idx] == vall[idx]:
                    true_prec[vall[idx]] += weights[idx]
                    true_recall[vall[idx]] += weights[idx]
                    trues += weights[idx]
                else:
                    falses += weights[idx]
                    false_recall[vall[idx]] += weights[idx]
                    try:
                        false_prec[softmax_classes[idx].int()] += weights[idx]
                    except:
                        false_prec[softmax_classes[idx]] += weights[idx]                    
        main_recall = trues / (trues + falses)
        print((th, main_recall))
        if main_recall >= best_recall:
            best_thr = th
            best_recall = main_recall
        print(trues, falses)

    utility_functions.pprint(("best_thr", best_thr), config[dataset_name])
    return best_thr

def mms_post_process(m, n_classes, parts, traincenterx, valx, t = 'val'):
    config = load_config()

    models = dict()
    n_clusters = len(parts)
    scenario = str(n_classes) + '_MMS' + str(n_clusters)
    dataset_name = config["dataset_name"]
    data_scenario_path = join(config[dataset_name]["scenario_embs"], scenario)
    model_scenario_path = join(config[dataset_name]["scenario_submodels"], scenario)

    for idx in (range(n_clusters)):    
        model_path = join(model_scenario_path, str(idx), 'exported', 'hrnetv2')
        with tf.device('/cpu:0'):
            model = tf.keras.models.load_model(model_path)
            models[idx] = model

    model = None
    batch_softmax = None
    batch_argmax_softmax = None
    batch_max_softmax = None

    softmax_prediction = dict()
    batch_size = 1000

    batch_number = len(valx) // batch_size + (1 if (len(valx) % batch_size != 0) else 0)

    max_softmax = dict()
    argmax_softmax = dict()

    for idx in tqdm(range(n_clusters)):
        if config['overwrite'] == False and os.path.isfile(join(data_scenario_path, str(idx) + '_predicted_max.npz')) and os.path.isfile(join(data_scenario_path, str(idx) + '_predicted_argmax.npz')):
            max_softmax[idx] = np.load(join(data_scenario_path, str(idx) + '_' + t + '_predicted_max.npz'))['res']
            argmax_softmax[idx] = np.load(join(data_scenario_path, str(idx) + '_' + t + '_predicted_argmax.npz'))['res']
        else:
            model_path = join(model_scenario_path, str(idx), 'exported', 'hrnetv2')
            model = tf.keras.models.load_model(model_path)

            max_softmax[idx] = []
            argmax_softmax[idx] = []

            for batch_counter in range(batch_number):
                batch_softmax = model(np.array(valx[batch_counter * batch_size : np.min([len(valx), (batch_counter + 1) * batch_size])]))
                batch_max_softmax = np.max(batch_softmax, axis=1)
                batch_argmax_softmax = np.argmax(batch_softmax, axis=1)
                max_softmax[idx] += list(batch_max_softmax)
                argmax_softmax[idx] += list(batch_argmax_softmax)
            savez_compressed(join(data_scenario_path, str(idx) + '_' + t + '_predicted_max.npz'), res=max_softmax[idx])
            savez_compressed(join(data_scenario_path, str(idx) + '_' + t + '_predicted_argmax.npz'), res=argmax_softmax[idx])

    max_softmax = np.array(list(max_softmax.values())).transpose()
    argmax_softmax = np.array(list(argmax_softmax.values())).transpose()

    # ابتدا نزدیکترین دسته به هر نمونه آزمون را پیدا میکنیم
    # شماره دسته، شماره خوشه، شماره دسته در خوشه و مقدار شباهت کسینوسی و مقدار سافتمکس نزدیکترین دسته را در لیستهای مجزا ذخیره میکنیم
    batch_size = 5000
    batch_numbers = len(valx) // batch_size + (1 if (len(valx) % batch_size != 0) else 0)

    sim_clusters = []
    sim_classes = []
    sim_classes_in_clusters=[]
    sim_values = []

    pre_path = data_scenario_path

    if config['overwrite'] == False and os.path.isfile(join(pre_path, t + '_sim_clusters.pt')):
        sim_clusters = np.array(torch.load(join(pre_path, t + '_sim_clusters.pt')))
        sim_classes = np.array(torch.load(join(pre_path, t + '_sim_classes.pt')))
        sim_classes_in_clusters = np.array(torch.load(join(pre_path, t + '_sim_classes_in_clusters.pt')))
        sim_values = np.array(torch.load(join(pre_path, t + '_sim_values.pt')))
        sim_softmax = np.array(torch.load(join(pre_path, t + '_sim_softmax.pt')))
    else:
        for batch in tqdm(range(batch_numbers)):
            if batch == batch_numbers - 1 and (len(valx) % batch_size):
                batch_clusters = [0] * (len(valx) % batch_size)
            else:
                batch_clusters = [0] * batch_size
            batch_classes_in_clusters = []
            if m == 'euclidean':
                batch_sim = utility_functions.euc_sim(torch.Tensor(valx[batch*batch_size:np.min([len(valx), (batch+1)*batch_size])]), torch.Tensor(traincenterx)) 
            else:
                batch_sim = utility_functions.cos_sim(torch.Tensor(valx[batch*batch_size:np.min([len(valx), (batch+1)*batch_size])]), torch.Tensor(traincenterx))
            
            # just for euclidean
            if m == 'euclidean':
                v = batch_sim
                v_min, v_max = v.min(), v.max() #(dim=1)[0], v.max(dim=1)[0]
                new_min, new_max = 0, 0.9
                v_p = ((v.transpose(0,1) - v_min)/(v_max - v_min)*(new_max - new_min) + new_min).transpose(0,1)
                batch_sim = v_p

            batch_classes = (batch_sim.max(1)[1]).numpy()
            batch_values = (batch_sim.max(1)[0]).numpy()
            for r in parts:
                batch_clusters += (r * (np.in1d(batch_classes, parts[r])).astype(np.int32))
            for idx, b in enumerate(batch_classes):
                batch_classes_in_clusters.append(np.where(parts[batch_clusters[idx]] == batch_classes[idx].item())[0][0])
            
            sim_clusters.extend(list(batch_clusters))
            sim_classes.extend(list(batch_classes))
            sim_classes_in_clusters.extend(list(batch_classes_in_clusters))
            sim_values.extend(list(batch_values))

        torch.save(torch.Tensor(sim_clusters), join(pre_path, t + '_sim_clusters.pt'))
        torch.save(torch.Tensor(sim_classes), join(pre_path, t + '_sim_classes.pt'))
        torch.save(torch.Tensor(sim_classes_in_clusters), join(pre_path, t + '_sim_classes_in_clusters.pt'))
        torch.save(torch.Tensor(sim_values), join(pre_path, t + '_sim_values.pt'))

        sim_clusters = np.array(torch.load(join(pre_path, t + '_sim_clusters.pt')))
        sim_classes = np.array(torch.load(join(pre_path, t + '_sim_classes.pt')))
        sim_classes_in_clusters = np.array(torch.load(join(pre_path, t + '_sim_classes_in_clusters.pt')))
        sim_values = np.array(torch.load(join(pre_path, t + '_sim_values.pt')))

        # نمونه های مربوط به هر خوشه را جدا میکنیم
        ids = dict()
        for r in parts:
            ids[r] = np.where(sim_clusters == r)[0]

        # دار سافتمکس دسته ای که بیشترین شباهت کسینوسی به داده آزمون را دارد
        sim_softmax = np.zeros(len(valx))
        m = dict()
        batch_size = 1000

        print(len(parts))
        print(sim_classes_in_clusters.shape)
        for i in tqdm(ids):
            m[i] = []
            batch_numbers = len(ids[i]) // batch_size + (1 if len(ids[i]) % batch_size != 0 else 0)
            for batch in (range(batch_numbers)):
                pr = models[i](valx[ids[i][batch * batch_size : np.min([len(ids[i]), (batch + 1) * batch_size])]])#[0][sim_classes_in_clusters[idx].int().item()]
                for pidx, p in enumerate(pr):
                    m[i].append(pr[pidx][sim_classes_in_clusters[ids[i][pidx]].astype('int')])
            sim_softmax[ids[i]] = m[i]

        torch.save(torch.Tensor(sim_softmax), join(pre_path, t + '_sim_softmax.pt'))
        sim_softmax = np.array(torch.load(join(pre_path, t + '_sim_softmax.pt')))

    
    # در مرحله دوم، ابتدا بهترین دسته هایی که بیشترین مقدار سافتمکس را دریافت کرده اند، پیدا میکنیم
    # شماره دسته، شماره خوشه، شماره دسته در خوشه، مقدار سافتمکس و مقدار شباهت کسینوسی دسته با داده آزمون
    if config['overwrite'] == False and os.path.isfile(join(pre_path, t + '_softmax_values.pt')):
        softmax_clusters = torch.load(join(pre_path, t + '_softmax_clusters.pt'))
        softmax_classes = torch.load(join(pre_path, t + '_softmax_classes.pt'))
        softmax_classes_in_clusters = torch.load(join(pre_path, t + '_softmax_classes_in_clusters.pt'))
        softmax_values = torch.load(join(pre_path, t + '_softmax_values.pt'))
        softmax_sims = torch.load(join(pre_path, t + '_softmax_sims.pt'))
    else:
        softmax_values = max_softmax.max(1)
        softmax_clusters = max_softmax.argmax(1)
        softmax_classes = []
        softmax_classes_in_clusters = []
        for idx, cl in tqdm(enumerate(softmax_clusters)):
            softmax_classes.append(parts[cl][argmax_softmax[idx][cl]])
            softmax_classes_in_clusters.append(argmax_softmax[idx][cl])

        torch.save(torch.Tensor(softmax_clusters), join(pre_path, t + '_softmax_clusters.pt'))
        torch.save(torch.Tensor(softmax_classes), join(pre_path, t + '_softmax_classes.pt'))
        torch.save(torch.Tensor(softmax_classes_in_clusters), join(pre_path, t + '_softmax_classes_in_clusters.pt'))
        torch.save(torch.Tensor(softmax_values), join(pre_path, t + '_softmax_values.pt'))

        # مقدار شباهت کسینوسی دسته ای که بیشترین مقدار سافتمکس را دارد
        softmax_sims = []
        for idx in tqdm(range(len(valx))):
            if m == 'euclidean':
                softmax_sims.append(utility_functions.euc_sim(torch.Tensor(np.array([valx[idx]])), torch.Tensor(np.array([traincenterx[softmax_classes[idx]]])))[0][0].item())
            else:
                softmax_sims.append(utility_functions.cos_sim(torch.Tensor(np.array([valx[idx]])), torch.Tensor(np.array([traincenterx[softmax_classes[idx]]])))[0][0].item())

        # # just for euclidean
        if m == 'euclidean':
            v = torch.Tensor(softmax_sims)
            v_min, v_max = v.min(), v.max()
            new_min, new_max = 0, 0.9
            v_p = ((v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min)
            softmax_sims = v_p   

        torch.save(torch.Tensor(softmax_sims), join(pre_path, t + '_softmax_sims.pt'))
    return sim_classes, sim_values, sim_softmax, softmax_values, softmax_sims, softmax_classes


def ism_post_process(m, n_classes, parts, valx, t = 'val'):
    models = dict()
    n_clusters = len(parts)

    config = load_config()
    dataset_name = config['dataset_name']

    scenario = str(n_classes) + '_ISM' + str(n_clusters)
    data_scenario_path = join(config[dataset_name]["scenario_embs"], scenario)
    model_scenario_path = join(config[dataset_name]["scenario_submodels"], scenario)

    for idx in (range(n_clusters)):    
        model_path = join(model_scenario_path, str(idx), 'exported', 'hrnetv2')
        with tf.device('/cpu:0'):
            model = tf.keras.models.load_model(model_path)
            models[idx] = model

    model = None
    batch_softmax = None
    batch_argmax_softmax = None
    batch_max_softmax = None

    softmax_prediction = dict()
    batch_size = 1000

    batch_number = len(valx) // batch_size + (1 if (len(valx) % batch_size != 0) else 0)

    max_softmax = dict()
    argmax_softmax = dict()

    for idx in tqdm(range(n_clusters)):
        if config['overwrite'] == False and os.path.isfile(join(data_scenario_path, str(idx) + '_predicted_max.npz')) and os.path.isfile(join(data_scenario_path, str(idx) + '_predicted_argmax.npz')):
            max_softmax[idx] = np.load(join(data_scenario_path, str(idx) + '_' + t + '_predicted_max.npz'))['res']
            argmax_softmax[idx] = np.load(join(data_scenario_path, str(idx) + '_' + t + '_predicted_argmax.npz'))['res']
        else:
            model_path = join(model_scenario_path, str(idx), 'exported', 'hrnetv2')
            model = tf.keras.models.load_model(model_path)

            max_softmax[idx] = []
            argmax_softmax[idx] = []

            for batch_counter in range(batch_number):
                batch_softmax = model(np.array(valx[batch_counter * batch_size : np.min([len(valx), (batch_counter + 1) * batch_size])]))
                batch_max_softmax = np.max(batch_softmax, axis=1)
                batch_argmax_softmax = np.argmax(batch_softmax, axis=1)
                max_softmax[idx] += list(batch_max_softmax)
                argmax_softmax[idx] += list(batch_argmax_softmax)
            savez_compressed(join(data_scenario_path, str(idx) + '_' + t + '_predicted_max.npz'), res=max_softmax[idx])
            savez_compressed(join(data_scenario_path, str(idx) + '_' + t + '_predicted_argmax.npz'), res=argmax_softmax[idx])

    max_softmax = np.array(list(max_softmax.values())).transpose()
    argmax_softmax = np.array(list(argmax_softmax.values())).transpose()

    # ابتدا نزدیکترین دسته به هر نمونه آزمون را پیدا میکنیم
    # شماره دسته، شماره خوشه، شماره دسته در خوشه و مقدار شباهت کسینوسی و مقدار سافتمکس نزدیکترین دسته را در لیستهای مجزا ذخیره میکنیم
    batch_size = 5000
    batch_numbers = len(valx) // batch_size + (1 if (len(valx) % batch_size != 0) else 0)

    pre_path = data_scenario_path
   
    # در مرحله دوم، ابتدا بهترین دسته هایی که بیشترین مقدار سافتمکس را دریافت کرده اند، پیدا میکنیم
    # شماره دسته، شماره خوشه، شماره دسته در خوشه، مقدار سافتمکس و مقدار شباهت کسینوسی دسته با داده آزمون
    if config['overwrite'] == False and os.path.isfile(join(pre_path, t + '_softmax_classes.pt')):
        softmax_clusters = torch.load(join(pre_path, t + '_softmax_clusters.pt'))
        softmax_classes = torch.load(join(pre_path, t + '_softmax_classes.pt'))
        softmax_classes_in_clusters = torch.load(join(pre_path, t + '_softmax_classes_in_clusters.pt'))
        softmax_values = torch.load(join(pre_path, t + '_softmax_values.pt'))
        softmax_sims = torch.load(join(pre_path, t + '_softmax_sims.pt'))
    else:
        softmax_values = max_softmax.max(1)
        softmax_clusters = max_softmax.argmax(1)
        softmax_classes = []
        softmax_classes_in_clusters = []
        for idx, cl in tqdm(enumerate(softmax_clusters)):
            softmax_classes.append(parts[cl][argmax_softmax[idx][cl]])
            softmax_classes_in_clusters.append(argmax_softmax[idx][cl])

        torch.save(torch.Tensor(softmax_clusters), join(pre_path, t + '_softmax_clusters.pt'))
        torch.save(torch.Tensor(softmax_classes), join(pre_path, t + '_softmax_classes.pt'))
        torch.save(torch.Tensor(softmax_classes_in_clusters), join(pre_path, t + '_softmax_classes_in_clusters.pt'))
        torch.save(torch.Tensor(softmax_values), join(pre_path, t + '_softmax_values.pt'))

    return softmax_classes

def delete_directory(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print("Error deleting directory " + path + str(e))