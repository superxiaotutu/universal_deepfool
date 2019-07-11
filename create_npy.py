from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import numpy as np
from targeted_universal_pert import targeted_perturbation
from util_univ import *
import tensorflow as tf
from tensorflow.python.platform import gfile

target = 3
path_train_imagenet = 'datasets2/ILSVRC2012/train'
device = '/gpu:0'

def jacobian(y_flat, x, inds):
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=2),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < 2,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()


with tf.device(device):
    persisted_sess = tf.Session()
    inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb')


    model = os.path.join(inception_model_path)

    # Load the Inception model
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    persisted_sess.graph.get_operations()

    persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")
    persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")

    print('>> Computing feedforward function...')


    def f(image_inp): return persisted_sess.run(persisted_output,
                                                feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})


    print('>> Compiling the gradient tensorflow functions. This might take some time...')
    y_flat = tf.reshape(persisted_output, (-1,))
    inds = tf.placeholder(tf.int32, shape=(2,))
    dydx = jacobian(y_flat, persisted_input, inds)

    print('>> Computing gradient function...')


    def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp,

                                                                                inds: indices}).squeeze(axis=1)

    for i in range(100):
        npy_img = 'data/npy_img/10classes1000imgs/' + str(i) +'.npy'
        npy_per = 'data/npy_per/10classes1000imgs/' + str(i) + '.npy'

        if os.path.isfile(npy_img) == 0:
            X = create_imagenet_npy(path_train_imagenet, len_batch=1100, num_class=10, p=0, q=0, r=i*10)
            np.save(npy_img, X)
        else:
            X = np.load(npy_img)
        if os.path.isfile(npy_per) == 0:
            v = targeted_perturbation(X, f, grad_fs, delta=0.25, max_iter_uni=10, target=target)
            np.save((npy_per), v)
