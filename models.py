import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from util import sparse_to_tuple


class BaseModel(object):
    def __init__(self, placeholders, degreeTasks, neighbor_list, num_class, fea_size, hash_dim, hidden_dim, num_hash, num_layers, activation=tf.nn.elu, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.features = placeholders['features']
        self.labels = placeholders['labels']
        self.masks = placeholders['masks']
        self.dropout = placeholders['dropout']
        self.degreeTasks = degreeTasks
        self.neighbor_list = neighbor_list
        self.hash_dim = hash_dim
        self.act = activation
        self.hid_units = []
        self.num_hash = num_hash
        self.num_class = num_class
        self.fea_size = fea_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def hashmap(self, fea_dim):
        """Create Hash Kernel Function."""
        rdn = -1 + 2 * np.random.random((fea_dim, self.hash_dim))
        row = np.arange(fea_dim)
        col = np.argmax(abs(rdn), axis=1)
        data = [1.0] * fea_dim
        coords = np.vstack((row, col)).transpose()
        tensor = tf.sparse_to_dense(sparse_indices=coords, output_shape=[fea_dim, self.hash_dim], sparse_values=data)

        sign = sp.diags(np.sign(-1 + 2 * np.random.random((fea_dim,))), dtype=np.float32)
        coords, values, shape = sparse_to_tuple(sign)
        sign = tf.SparseTensor(indices=coords, values=values, dense_shape=shape)
        tensor = tf.sparse_tensor_dense_matmul(sign, tensor)
        return tensor

    def global_hashing_layer(self, id_layer, from_self, out_sz, act=tf.nn.elu):
        """ Hash Kernel based Multi-task Function """
        fea_dim = from_self.get_shape().as_list()[1]
        with tf.name_scope('global_hash_layer' + str(id_layer)):
            hashed_feas = []
            for _ in range(self.num_hash):
                from_neighs = []
                global_hashMap = self.hashmap(fea_dim)
                for i, (nodeDegree, nodeID) in enumerate(self.degreeTasks):
                    neighID = self.neighbor_list[i]
                    if nodeDegree == 0:
                        neigh_fea = tf.gather(from_self, nodeID)
                        neigh_hashMap = self.hashmap(fea_dim)
                        hashed_neigh = tf.matmul(neigh_fea, neigh_hashMap)
                        hashed_global = tf.matmul(neigh_fea, global_hashMap)
                        from_neighs.append(tf.add_n([hashed_neigh, hashed_global]))
                    else:
                        neigh_fea = tf.gather(from_self, neighID)
                        neigh_hashMap = self.hashmap(fea_dim)
                        hashed_neigh = tf.matmul(neigh_fea, neigh_hashMap)
                        hashed_global = tf.matmul(neigh_fea, global_hashMap)
                        h = tf.reshape(tf.add_n([hashed_neigh, hashed_global]), [len(nodeID), nodeDegree, self.hash_dim])
                        h = tf.reduce_mean(h, axis=1)
                        from_neighs.append(h)
                from_neighs = tf.concat(from_neighs, axis=0)
                all_list = []
                for i, (nodeDegree, nodeID) in enumerate(self.degreeTasks):
                    all_list = all_list + nodeID
                id_list = np.argsort(all_list)
                nodeTFID = tf.Variable(tf.constant(id_list), trainable=False)
                seq_neighs = tf.nn.embedding_lookup(from_neighs, nodeTFID)
                hashed_feas.append(seq_neighs)
            hashed_feas = tf.concat(hashed_feas, axis=-1)

            hidden_neigh = tf.layers.conv1d(tf.expand_dims(hashed_feas, axis=0), out_sz, 1, use_bias=False)
            hideen_self = tf.layers.conv1d(tf.expand_dims(from_self, axis=0), out_sz, 1, use_bias=False)
            hidden_neigh = tf.nn.dropout(hidden_neigh, 1.0 - self.dropout)
            hideen_self = tf.nn.dropout(hideen_self, 1.0 - self.dropout)

            self_neigh = tf.add_n([hidden_neigh, hideen_self])
            self_neigh = tf.squeeze(self_neigh)
            ret = tf.contrib.layers.bias_add(self_neigh)
        return act(ret)

    def global_weight_layer(self, id_layer, inputs, out_sz, act=tf.nn.elu):
        """ Weight-based Multi-task Function """
        with tf.name_scope('global_hash_layer' + str(id_layer)):
            global_maps = tf.layers.conv1d(tf.expand_dims(inputs, axis=0), out_sz, 1, padding='valid',
                                               use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            global_maps = tf.squeeze(global_maps)
            from_neighs = []
            for i, (nodeDegree, nodeID) in enumerate(self.degreeTasks):
                neighID = self.neighbor_list[i]
                if nodeDegree == 0:
                    neigh_fea = tf.gather(global_maps, nodeID)
                    from_neighs.append(neigh_fea)
                else:
                    neigh_inputs = tf.gather(inputs, neighID)
                    neigh_local = tf.layers.conv1d(tf.expand_dims(neigh_inputs, axis=0), out_sz, 1, padding='valid',
                                                   use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
                    neigh_local = tf.squeeze(neigh_local)

                    neigh_global = tf.nn.embedding_lookup(global_maps, neighID)
                    h = tf.reshape(tf.add(neigh_local, neigh_global), [len(nodeID), nodeDegree, out_sz])

                    h = tf.reduce_mean(h, axis=1)
                    from_neighs.append(h)

            from_neighs = tf.concat(from_neighs, axis=0)
            all_list = []
            for i, (nodeDegree, nodeID) in enumerate(self.degreeTasks):
                all_list = all_list + nodeID
            id_list = np.argsort(all_list)
            nodeTFID = tf.Variable(tf.constant(id_list), trainable=False)
            hidden_neigh = tf.nn.embedding_lookup(from_neighs, nodeTFID)

            hideen_self = tf.layers.conv1d(tf.expand_dims(inputs, axis=0), out_sz, 1, padding='valid',
                                           use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            hideen_self = tf.squeeze(hideen_self)

            hideen_self = tf.nn.dropout(hideen_self, 1.0 - self.dropout)
            hidden_neigh = tf.nn.dropout(hidden_neigh, 1.0 - self.dropout)
            ret = tf.add_n([hidden_neigh, hideen_self])
            ret = tf.contrib.layers.bias_add(ret)
        return act(ret)

    def inference(self):
        """Create DEMO-Net With Weight-based Multi-task Function"""
        with tf.name_scope('model'):
            inputs = self.features
            for i in range(self.num_layers):
                inputs = self.global_weight_layer(i, inputs, out_sz=self.hidden_dim, act=self.act)

            logits = self.global_weight_layer(self.num_layers, inputs, out_sz=self.num_class, act=lambda x: x)
        return logits

    def training(self, loss, lr, l2_coef):
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss + lossL2)
        return train_op

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(self, logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)
