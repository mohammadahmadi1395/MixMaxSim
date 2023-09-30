import utility_functions, clustering, training
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

seed(1)
tf.random.set_seed(2)


def prepare_data(dataset_name, config):
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

    if overwrite == False and os.path.isfile(join(super_scenario_path, 'testx.npz')):
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


def cluster_data(method, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall):
    distribution_path = join(data_scenario_path, 'parts.npz')
    parts = dict()

    if method == 'ISM':   
        if overwrite == False and os.path.isfile(distribution_path):
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

        if overwrite == False and os.path.isfile(distribution_path) and os.path.isfile(clustering_model_filename):
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

def cluster_data(method, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall):
    distribution_path = join(data_scenario_path, 'parts.npz')
    parts = dict()

    if method == 'ISM':   
        if overwrite == False and os.path.isfile(distribution_path):
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

        if overwrite == False and os.path.isfile(distribution_path) and os.path.isfile(clustering_model_filename):
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


def train_submodels(method, parts, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall):
    dataset_path = config[dataset_name]["features"]
    for r in (range(n_clusters)):
        utility_functions.pprint(('cluster', str(r)), dataset_name)
        train_ids = utility_functions.train_calc_ids(parts[r], trainy, trainl, r)
        test_ids = utility_functions.test_calc_ids(parts[r], trainy, trainl, r)
        train_sample_count, test_sample_count = utility_functions.convert_emb_to_tfrecord(dataset_name, data_scenario_path, dataset_path, train_ids, test_ids, r, overwrite=False, all_samples=True, n_classes=len(parts[r]))
        train_dataset, test_dataset = utility_functions.prepare_data_sets(dataset_name, data_scenario_path, train_ids, test_ids, r)    
        training.softmax_train(dataset_name, model_scenario_path, train_dataset, parts[r], trainx, trainl, r, epochs=50,train_overwrite=False, freq = train_sample_count // 100)