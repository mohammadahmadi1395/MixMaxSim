import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
import math
from os.path import join
from tensorflow.keras import Model
from tqdm import tqdm

def current_time():
    return datetime.now().strftime('%y/%m/%d-%H:%M:%S')

def cos_sim(a, b):
    a_norm = a.norm(dim=-1, keepdim=True)
    b_norm = b.norm(dim=-1, keepdim=True)
    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)
    return a @ b.transpose(-2, -1)

def euc_sim(a, b):
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]
    
def pprint(tpl, dataset_name, endline=False, doubleendline=False):
    with open('F:/test/deepface/mixmaxsim/results/results.txt', 'a') as f:
        print(current_time(), tpl)
        print(current_time(), tpl, file=f)
        if endline:
            print('-'*15)
            print('-'*15, file=f)
        if doubleendline:
            print('-'*15)
            print('-'*15)
            print('*'*15, file=f)
            print('*'*15, file=f)

def _parse_tfrecord(is_transform=True):
    def parse_tfrecord(tfrecord):
        features = {'image/class_num_in_all': tf.io.FixedLenFeature([], tf.int64),
                    'image/class_num_in_cluster': tf.io.FixedLenFeature([], tf.int64),
                    'image/filename': tf.io.FixedLenFeature([], tf.string),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)
        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        x_train = _transform_images(is_transform=is_transform)(x_train)

        y_train = tf.cast(x['image/class_num_in_cluster'], tf.int64)

        return x_train, y_train #(x_train, y_train), y_train
    return parse_tfrecord


def _transform_images(is_transform=True):
    def transform_images(x_train):
        if is_transform:
            # x_train = tf.image.decode_jpeg(x_train, channels=3)
            x_train = tf.image.resize(x_train, (128, 128), method="nearest")
            x_train = tf.image.random_crop(x_train, (112, 112, 3))
            x_train = tf.image.random_flip_left_right(x_train)
            # x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
            # x_train = tf.image.random_brightness(x_train, 0.4)

            # extra
            # x_train = tfa.image.rotate(x_train, np.random.randint(90))
            # x_train = tf.image.random_contrast(x_train, 0.2, 1.5)
            # augmented_image = adjust_color_img(augmented_image)
            # augmented_image = translate_img(augmented_image)
            # # augmented_image = random_crop(image)
            # # augmented_image = shear_img(augmented_image)
            # # augmented_image = rotate_img(augmented_image)
            # # augmented_image = adjust_contrast_img(augmented_image)
            # # augmented_image = adjust_brightness_img(augmented_image)
            # # augmented_image = img_equalize(augmented_image)

        x_train = tf.image.resize(x_train, (112, 112), method="nearest")
        # x_train = x_train / 255
        x_train = (tf.cast(x_train, tf.float32) - 127.5) / 128.
        return x_train
    return transform_images


# def load_tfrecord_dataset(tfrecord_name, batch_size,
#                           binary_img=True, shuffle=True, buffer_size=10240,
#                           is_ccrop=False, is_repeat=True, is_transform=True):
#     """load dataset from tfrecord"""
#     raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
#     if is_repeat:
#         raw_dataset = raw_dataset.repeat()
#     if shuffle:
#         raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
#     dataset = raw_dataset.map(
#         _parse_tfrecord(binary_img=binary_img, is_ccrop=is_ccrop, is_transform=is_transform),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(
#         buffer_size=tf.data.experimental.AUTOTUNE)
#     return dataset

# def prepare_dataset(n_classes, batch_size=64, train_transform=True, test_transform=False):
#     train_dataset = load_tfrecord_dataset('./data/' + str(n_classes) + '/glint_bin_train.tfrecord', batch_size, shuffle=True, binary_img=True, is_ccrop=True, is_transform=train_transform)
#     val_dataset = load_tfrecord_dataset('./data/' + str(n_classes) + '/glint_bin_test.tfrecord', batch_size, binary_img=True, shuffle=False, is_repeat=False, is_transform=test_transform)
#     return train_dataset, val_dataset


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        # self.w = self.add_variable(
        #     "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists

def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head

def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        # x = Dense(num_classes, activation='softmax')(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head

def SoftmaxLoss():
    """softmax loss"""
    def softmax_loss(y_true, y_pred):
        # y_true: sparse target
        # y_pred: logist
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)
        return tf.reduce_mean(ce)
    return softmax_loss
























# # old:
# # فقط داده های درون هر خوشه، در آموزش مدل مشارکت دارند
# # def train_calc_ids(res, y, label, sub):
# #     y_result = y[res].squeeze()
# #     idx = 0
# #     counter = 0
# #     train_ids = dict()
# #     train_ids[idx] = []
# #     for ind, id in tqdm(enumerate(y_result)):
# #         class_label = np.where(label[res[ind]] == unique_result[sub])[0][0]
# #         # train_ids[idx] += [[id, sub, class_label, trainl[res[ind]], res[ind]]] # total_class_name, cluster_number, class_label, total_class_label, sample_number
# #         train_ids[idx] += [[id, class_label, res[ind]]] 
# #         counter += 1
# #         if (counter != 0) and (counter % 50000 == 0):
# #             idx += 1
# #             train_ids[idx] = []
# #     return train_ids

# # new:
# # همه داده های همکلاس با داده های درون خوشه، در آموزش مدل مشارکت دارند
def train_calc_ids(res, trainy, trainl, sub):    
    indexes = np.unique(trainy, return_index=True)[1]
    all_unique_ids = np.array([trainy[index] for index in sorted(indexes)])
    idx = 0
    counter = 0
    train_ids = dict()
    train_ids[idx] = []
    for ind, id in tqdm(enumerate(all_unique_ids[res])):
        total_class_number = res[ind]
        class_label = np.where(total_class_number == res)[0][0]
        train_ids[idx] += [[id, class_label, ind]]
        counter += 1
        if (counter != 0) and (counter % 1000 == 0):
            idx += 1
            train_ids[idx] = []

    return train_ids

def test_calc_ids(res, trainy, trainl, sub):
    indexes = np.unique(trainy, return_index=True)[1]
    all_unique_ids = np.array([trainy[index] for index in sorted(indexes)])
    idx = 0
    counter = 0
    test_ids = dict()
    test_ids[idx] = []
    for ind, id in tqdm(enumerate(all_unique_ids[res])):
        total_class_number = res[ind]
        class_label = np.where(total_class_number == res)[0][0]
        test_ids[idx] += [[id, class_label, ind]]
        # test_ids[idx] += [[id, unique_result[sub][ind], ind]]
        counter += 1
        if (counter != 0) and (counter % 4000 == 0):
            idx += 1
            test_ids[idx] = []

    return test_ids

# # مورد استفاده قرار نمیگیرد. برای زمانی است که میخواستم برای خوشه بندی هم دسته بندی انجام دهیم
# def clustering_train_calc_ids(trainl, trainy, result):        
#     counter = 0
#     unique_trainl = np.unique(trainl)
#     clustering_train_ids = dict()
#     clustering_train_ids[counter] = []
#     for idx in range(len(unique_trainl)):
#         first_trainl = np.where(trainl == unique_trainl[idx])[0][0]
#         for cluster in result:
#             if unique_trainl[idx] in result[cluster]:
#                 cluster_idx = cluster
#                 break
#         clustering_train_ids[counter].append([trainy[first_trainl], cluster_idx, trainl[first_trainl]])
#         if ((idx + 1) % 2000 == 0):
#             counter += 1
#             clustering_train_ids[counter] = []
#     return clustering_train_ids

# # مورد استفاده قرار نمیگیرد. برای زمانی است که میخواستم برای خوشه بندی هم دسته بندی انجام دهیم
# def clustering_test_calc_ids(testl, testy, result):        
#     counter = 0
#     unique_testl = np.unique(testl)
#     clustering_test_ids = dict()
#     clustering_test_ids[counter] = []
#     for idx in range(len(unique_testl)):
#         first_testl = np.where(testl == unique_testl[idx])[0][0]
#         for cluster in result:
#             if unique_testl[idx] in result[cluster]:
#                 cluster_idx = cluster
#                 break
#         clustering_test_ids[counter].append([testy[first_testl], cluster_idx, testl[first_testl]])
#         if ((idx + 1) % 8000 == 0):
#             counter += 1
#             clustering_test_ids[counter] = []
#     return clustering_test_ids



def parse_record(record, arc):
    # if arc == True:
    #     name_to_feature = {
    #         'emb': tf.io.FixedLenFeature([], tf.string),
    #         'label': tf.io.FixedLenFeature([], tf.string), # for arc
    #         # 'label': tf.io.FixedLenFeature([],tf.int64), # for arcfacelosssimple: tf.string), 
    #     }
    # else:
    name_to_feature = {
        'emb': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([],tf.int64), # for arcfacelosssimple: tf.string),
    }
    return tf.io.parse_single_example(record, name_to_feature)


def arc_decode_record(record):
    emb = tf.io.decode_raw(record['emb'], out_type=np.float32,little_endian=True,fixed_length=None, name=None)
    # label = tf.io.decode_raw(record['label'], out_type=np.float32,little_endian=True,fixed_length=None, name=None)
    label = record['label'] # for sparse
    return emb, label#, id

def normal_decode_record(record):
    emb = tf.io.decode_raw(record['emb'], out_type=np.float32,little_endian=True,fixed_length=None, name=None)
    label = record['label']
    return emb, label#, id

# # برای زمانی است که میخواستیم خوشه بندی را هم طبقه بندی کنیم
# # def clustering_convert_emb_to_tfrecord(dataset_name, scenario_number, tr_ids, te_ids, overwrite=True, arc=True):
# #     test_sample_count = 0
# #     train_sample_count = 0
# #     n_shards = len(tr_ids)

# #     path = join('.', dataset_name, 'data', str(scenario_number), 'tfrecords', 'clustering')
# #     os.makedirs(path, exist_ok=True)

# #     for shard_counter in tqdm(range(n_shards)):        
# #         # pprint('train_%.3d-of-%.3d' % (shard_counter + 1, n_shards))
# #         x_train = []
# #         y_train = []
# #         id_train = []
# #         for id in (tr_ids[shard_counter]):
# #             cl_train_emb = np.load(join('.', dataset_name, 'embeddings', 'common', 'all', 'train', id[0] + '.npz'))['res']
# #             x_train += list(cl_train_emb)
# #             y_train +=  ([id[2]] * len(cl_train_emb))
        
# #         train_sample_count += len(x_train)

# #         if overwrite==False and os.path.isfile(join(path,'train_%.3d-of-%.3d.tfrecord' % (shard_counter + 1, n_shards))):
# #             continue

# #         train_features = x_train
# #         train_labels = y_train

# #         with tf.io.TFRecordWriter(join(path,'train_%.3d-of-%.3d.tfrecord' % (shard_counter + 1, n_shards))) as writer:
# #             # train_ids = id_train
# #             for j in range(len(train_features)):
# #                 emb = train_features[j]
# #                 # label = (train_labels[j] == np.array(range(n_classes))).astype(np.float32) # for arc
# #                 label= train_labels[j] # for simple
# #                 # id = train_ids[j]
# #                 with tf.device('/cpu'):
# #                     tf_example = sample_example(emb, label, arc) #, id)
# #                     writer.write(tf_example.SerializeToString())

# #     n_shards = len(te_ids) #(n_classes // n_identity_shard) + (1 if (n_classes % n_identity_shard) != 0 else 0)
# #     for shard_counter in tqdm(range(n_shards)):
# #         # pprint('test_%.3d-of-%.3d' % (shard_counter + 1, n_shards))
# #         x_test = []
# #         y_test = []
# #         id_test = []
# #         for id in (te_ids[shard_counter]):           
# #             cl_val_emb = np.load(join('.', dataset_name, 'embeddings', 'common', 'all', 'val', id[0] + '.npz'))['res']
# #             x_test += list(cl_val_emb)
# #             y_test +=  ([id[2]] * len(cl_val_emb)) #(list(np.where(cluster_ids == id[2])[0]) * len(cl_val_emb)) # ([100] * len(cl_val_emb))

# #         test_sample_count += len(x_test)

# #         if overwrite == False and os.path.isfile(join(path,'test_%.3d-of-%.3d.tfrecord' % (shard_counter + 1, n_shards))):
# #             continue

# #         test_features = x_test
# #         test_labels = y_test

# #         with tf.io.TFRecordWriter(join(path,'test_%.3d-of-%.3d.tfrecord' % (shard_counter + 1, n_shards))) as writer:
# #             for j in range(len(test_features)):
# #                 emb = test_features[j]
# #                 # label = (test_labels[j] == np.array(range(n_classes))).astype(np.float32) #test_labels[j] #idx # for arcfacelosssimple: 
# #                 label = test_labels[j] #idx # for arcfacelosssimple: 
# #                 with tf.device('/cpu'):
# #                     tf_example = sample_example(emb, label, arc) #, id)
# #                     writer.write(tf_example.SerializeToString())

# #     return train_sample_count , test_sample_count


def convert_emb_to_tfrecord(dataset_name, data_scenario_path, dataset_path, tr_ids, te_ids, sub, n_classes, overwrite=True, all_samples=True, arc=True):
    test_sample_count = 0
    train_sample_count = 0
    n_shards = len(tr_ids)

    path = join(data_scenario_path, 'tfrecords', str(sub))
    os.makedirs(path, exist_ok=True)

    for shard_counter in tqdm(range(n_shards)):        
        # pprint('train_%.3d-of-%.3d' % (shard_counter + 1, n_shards))
        x_train = []
        y_train = []
        id_train = []
        for id in (tr_ids[shard_counter]):
            if all_samples == False:
                cl_train_emb = trainx[id[2]]
                x_train.append(cl_train_emb)
                y_train += [id[1]]
            else:
                cl_train_emb = None
                # if dataset_name == 'vggface2':
                #     cl_train_emb = np.load(join("I:/balanced_vggface2/embeddings/train", id[0]+ '.npz'))['res']
                # elif dataset_name == 'ms1m':
                #     cl_train_emb = np.load(join("E:/balanced_ms1m/embeddings/train", id[0]+ '.npz'))['res']
                # else:
                cl_train_emb = np.load(join(dataset_path, 'train', id[0] + '.npz'))['res']
                x_train += list(cl_train_emb)
                y_train +=  ([id[1]] * len(cl_train_emb))
        
        train_sample_count += len(x_train)

        if overwrite==False and os.path.isfile(join(path,'train_%.3d-of-%.3d.tfrecord' % (shard_counter + 1, n_shards))):
            continue

        train_features = x_train
        train_labels = y_train

        with tf.io.TFRecordWriter(join(path,'train_%.3d-of-%.3d.tfrecord' % (shard_counter + 1, n_shards))) as writer:
            # train_ids = id_train
            for j in range(len(train_features)):
                emb = train_features[j]
                # label = (train_labels[j] == np.array(range(n_classes))).astype(np.float32) # for arc
                label= train_labels[j] # for simple
                # id = train_ids[j]
                with tf.device('/cpu'):
                    tf_example = sample_example(emb, label, arc) #, id)
                    writer.write(tf_example.SerializeToString())

    n_shards = len(te_ids) #(n_classes // n_identity_shard) + (1 if (n_classes % n_identity_shard) != 0 else 0)
    for shard_counter in tqdm(range(n_shards)):
        # pprint('test_%.3d-of-%.3d' % (shard_counter + 1, n_shards))
        x_test = []
        y_test = []
        id_test = []
        for id in (te_ids[shard_counter]):           
            cl_test_emb = None
            # if dataset_name == 'vggface2':
            #     cl_test_emb = np.load(join("I:/balanced_vggface2/embeddings/val", id[0]+ '.npz'))['res']
            # elif dataset_name == 'ms1m':
            #     cl_test_emb = np.load(join("E:/balanced_ms1m/embeddings/val", id[0]+ '.npz'))['res']
            # else:
            cl_test_emb = np.load(join(dataset_path, 'val', id[0] + '.npz'))['res']
            x_test += list(cl_test_emb)
            y_test +=  ([id[1]] * len(cl_test_emb)) #(list(np.where(cluster_ids == id[2])[0]) * len(cl_val_emb)) # ([100] * len(cl_val_emb))

        test_sample_count += len(x_test)

        if overwrite == False and os.path.isfile(join(path,'test_%.3d-of-%.3d.tfrecord' % (shard_counter + 1, n_shards))):
            continue

        test_features = x_test
        test_labels = y_test

        with tf.io.TFRecordWriter(join(path,'test_%.3d-of-%.3d.tfrecord' % (shard_counter + 1, n_shards))) as writer:
            for j in range(len(test_features)):
                emb = test_features[j]
                # label = (test_labels[j] == np.array(range(n_classes))).astype(np.float32) #test_labels[j] #idx # for arcfacelosssimple: 
                label = test_labels[j] #idx # for arcfacelosssimple: 
                with tf.device('/cpu'):
                    tf_example = sample_example(emb, label, arc) #, id)
                    writer.write(tf_example.SerializeToString())

    return train_sample_count , test_sample_count

def prepare_data_sets(dataset_name, data_scenario_path, train_ids, test_ids, sub, arc=True):    
    n_shards = len(train_ids)
    path = join(data_scenario_path, 'tfrecords', str(sub))

    batch_size = 50
    files = tf.io.matching_files(join(path, 'train_*-of-' + '{:>03}'.format(len(train_ids)) + '.tfrecord'))
    files = tf.random.shuffle(files)
    shards = tf.data.Dataset.from_tensor_slices(files)
    train_dataset = shards.interleave(tf.data.TFRecordDataset)
    train_dataset = train_dataset.shuffle(buffer_size=100000) #l * n_identity_shard)
    # if arc == True:
    #     train_dataset = train_dataset.map(
    #         lambda x: arc_decode_record(parse_record(x, arc))#,
    #     )
    # else:
    train_dataset = train_dataset.map(
        lambda x: normal_decode_record(parse_record(x, arc))#,
    )

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
    files = tf.io.matching_files(join(path, 'test_*-of-' + '{:>03}'.format(len(test_ids)) + '.tfrecord'))
    shards = tf.data.Dataset.from_tensor_slices(files)
    test_dataset = shards.interleave(tf.data.TFRecordDataset)
    # if arc == True:
    #     test_dataset = test_dataset.map(
    #         lambda x: arc_decode_record(parse_record(x, arc))
    #     )
    # else:
    test_dataset = test_dataset.map(
        lambda x: normal_decode_record(parse_record(x, arc))
    )

    test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def test_acc_calc(sub, test_dataset, dataset_name, model_scenario_path):
    model = tf.keras.models.load_model(os.path.join(model_scenario_path, str(sub), 'exported', 'hrnetv2'))

    lbls = []
    preds = []
    for test in test_dataset:
        for t in test[1]:
            # lbls.append(np.where(t == 1)[0][0])        
            lbls.append(t)
        # preds += list(np.argmax(model.predict(test[0]), axis=1))
        preds += list(np.argmax(model(test[0]), axis=1))

    trues = 0
    falses = 0
    for idx in range(len(lbls)):
        if lbls[idx] == preds[idx]:
            trues += 1
        else:
            falses += 1
    val_acc = trues / (trues + falses)
    pprint(('val_acc for cluster ', sub, ' is : ', val_acc ), dataset_name)
    return val_acc

def _float32_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_features(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def sample_example(emb, label, arc=True): #, id):#, dimension):
    # if arc == True:
    #     feature = {
    #         # 'label': _int64_feature(label), # for sparse
    #         'label': _bytes_features(label.tobytes()), # for arcfacelosssimple: 
    #         'emb': _bytes_features(emb.tobytes()),
    #     }
    # else:
    feature = {
        'label': _int64_feature(label),
        'emb': _bytes_features(emb.tobytes()),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))
