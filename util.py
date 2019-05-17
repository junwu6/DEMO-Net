import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing

###############################################
# Some code adapted from tkipf/gcn            #
# https://github.com/tkipf/gcn                #
###############################################


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Read the data and preprocess the task information."""
    dataset_G = "data/{}-airports.edgelist".format(dataset_str)
    dataset_L = "data/labels-{}-airports.txt".format(dataset_str)
    label_raw, nodes = [], []
    with open(dataset_L, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            node, label = lines.split()
            if label == 'label': continue
            label_raw.append(int(label))
            nodes.append(int(node))
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(label_raw)
    G = nx.read_edgelist(open(dataset_G, 'rb'), nodetype=int)
    adj = nx.adjacency_matrix(G, nodelist=nodes)
    features = sp.csr_matrix(adj)

    # Randomly split the train/validation/test set
    indices = np.arange(adj.shape[0]).astype('int32')
    np.random.shuffle(indices)
    idx_train = indices[:adj.shape[0] // 3]
    idx_val = indices[adj.shape[0] // 3: (2 * adj.shape[0]) // 3]
    idx_test = indices[(2 * adj.shape[0]) // 3:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # task information
    degreeNode = np.sum(adj, axis=1).A1
    degreeNode = degreeNode.astype(np.int32)
    degreeValues = set(degreeNode)

    neighbor_list = []
    degreeTasks = []
    adj = adj.todense()
    for value in degreeValues:
        degreePosition = [int(i) for i, v in enumerate(degreeNode) if v == value]
        degreeTasks.append((value, degreePosition))

        d_list = []
        for idx in degreePosition:
            neighs = [int(i) for i in range(adj.shape[0]) if adj[idx, i] > 0]
            d_list += neighs
        neighbor_list.append(d_list)
        assert len(d_list) == value * len(degreePosition), 'The neighbor lists are wrong!'
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, degreeTasks, neighbor_list


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx
