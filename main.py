from __future__ import division
from __future__ import print_function
from time import strftime, localtime
import tensorflow as tf
import argparse
import numpy as np

from util import load_data
from models import BaseModel
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed
seed = 2019
np.random.seed(seed)
tf.set_random_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DEMO-Net.")
    parser.add_argument('--dataset', nargs='?', default='brazil',
                        help='Choose a dataset: brazil, europe or usa')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--dropout', type=int, default=0.6,
                        help='dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience to update the parameters.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight for l2 loss on embedding matrix')
    parser.add_argument('--hash_dim', type=int, default=256,
                        help='Feature hashing dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden units')
    parser.add_argument('--n_hash_kernel', type=int, default=1,
                        help='Number of hash kernels')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of hidden layers')
    return parser.parse_args()


def construct_placeholder(num_nodes, fea_size, num_classes):
    with tf.name_scope('input'):
        placeholders = {
            'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
            'features': tf.placeholder(tf.float32, shape=(num_nodes, fea_size), name='features'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'masks': tf.placeholder(dtype=tf.int32, shape=(num_nodes,), name='masks'),
        }
        return placeholders


def train(args, data):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, degreeTasks, neighbor_list = data
    features = features.todense()
    num_nodes, fea_size = features.shape
    num_classes = y_train.shape[1]

    placeholders = construct_placeholder(num_nodes, fea_size, num_classes)

    model = BaseModel(placeholders, degreeTasks, neighbor_list, num_classes, fea_size, hash_dim=args.hash_dim,
                      hidden_dim=args.hidden_dim, num_hash=args.n_hash_kernel, num_layers=args.n_layers)

    logits = model.inference()
    log_resh = tf.reshape(logits, [-1, num_classes])
    lab_resh = tf.reshape(placeholders['labels'], [-1, num_classes])
    msk_resh = tf.reshape(placeholders['masks'], [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr=args.lr, l2_coef=args.weight_decay)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())

    vloss_min = np.inf
    vacc_max = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)
        vacc_early_model = 0.0
        vlss_early_model = 0.0

        for epoch in range(args.epochs):
            train_feed_dict = {}
            train_feed_dict.update({placeholders['labels']: y_train})
            train_feed_dict.update({placeholders['features']: features})
            train_feed_dict.update({placeholders['dropout']: args.dropout})
            train_feed_dict.update({placeholders['masks']: train_mask})
            _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy], feed_dict=train_feed_dict)

            val_feed_dict = {}
            val_feed_dict.update({placeholders['labels']: y_val})
            val_feed_dict.update({placeholders['features']: features})
            val_feed_dict.update({placeholders['dropout']: 0.0})
            val_feed_dict.update({placeholders['masks']: val_mask})
            loss_value_val, acc_val = sess.run([loss, accuracy], feed_dict=val_feed_dict)

            print('Training epoch %d-th: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                  (epoch + 1, loss_value_tr, acc_tr, loss_value_val, acc_val))

            if acc_val >= vacc_max or loss_value_val <= vloss_min:
                if acc_val >= vacc_max and loss_value_val <= vloss_min:
                    vacc_early_model = acc_val
                    vlss_early_model = loss_value_val
                vacc_max = np.max((acc_val, vacc_max))
                vloss_min = np.min((loss_value_val, vloss_min))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == args.patience:
                    print('Early stop! Min loss: ', vloss_min, ', Max accuracy: ', vacc_max)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

        test_feed_dict = {}
        test_feed_dict.update({placeholders['labels']: y_test})
        test_feed_dict.update({placeholders['features']: features})
        test_feed_dict.update({placeholders['dropout']: 0.0})
        test_feed_dict.update({placeholders['masks']: test_mask})
        loss_value_test, acc_test = sess.run([loss, accuracy], feed_dict=test_feed_dict)
        print('Test loss:', loss_value_test, '; Test accuracy:', acc_test)
        sess.close()


if __name__ == '__main__':
    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    print("The time of running the codes: ", time_stamp)
    args = parse_args()
    data = load_data(args.dataset)
    train(args, data)
