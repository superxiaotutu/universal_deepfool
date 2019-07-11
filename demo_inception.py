import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


from targeted_universal_pert import targeted_perturbation
from util_univ import *

# if you want using cpu. change for device='/cpu:0'
device = '/gpu:0'

# choose your target
target = 3

npy_data = 'top-imagenet_data.npy'
#npy_data = 'bottom-imagenet_data.npy'
#npy_data = 'middle-imagenet_data.npy'
npy_perturbations ='modif-top-universal-target-'+str(target).zfill(5)+'.npy'
#npy_perturbations ='modif-bottom-universal-target-'+str(target).zfill(5)+'.npy'


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

if __name__ == '__main__':

    # Parse arguments
    argv = sys.argv[1:]

    # Default values
    path_train_imagenet = 'datasets2/ILSVRC2012/train'
    path_test_image = 'data/test_img.jpg'
    
    try:
        opts, args = getopt.getopt(argv,"i:t:",["test_image=","training_path="])
    except getopt.GetoptError:
        print ('python ' + sys.argv[0] + ' -i <test image> -t <imagenet training path>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-t':
            path_train_imagenet = arg
        if opt == '-i':
            path_test_image = arg

    with tf.device(device):
        persisted_sess = tf.Session()
        inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb')

        if os.path.isfile(inception_model_path) == 0:
            print('Downloading Inception model...')
            urlretrieve ("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip", os.path.join('data', 'inception5h.zip'))
            # Unzipping the file
            zip_ref = zipfile.ZipFile(os.path.join('data', 'inception5h.zip'), 'r')
            zip_ref.extract('tensorflow_inception_graph.pb', 'data')
            zip_ref.close()

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
        def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})
        if target==None:
            file_perturbation = os.path.join('data','precomputing_perturbations', 'modif-top-universal-nontarget.npy')
        else:
            file_perturbation = os.path.join('data','precomputing_perturbations', npy_perturbations)
            
        # TODO: Optimize this construction part!
        print('>> Compiling the gradient tensorflow functions. This might take some time...')
        y_flat = tf.reshape(persisted_output, (-1,))
        inds = tf.placeholder(tf.int32, shape=(2,))
        dydx = jacobian(y_flat,persisted_input,inds)

        print('>> Computing gradient function...')
        def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

        if os.path.isfile(file_perturbation) == 0:

            # Load/Create data
            datafile = os.path.join('data', npy_data)
            if os.path.isfile(datafile) == 0:
                print('>> Creating pre-processed imagenet data...')
                X = create_imagenet_npy(path_train_imagenet)

                print('>> Saving the pre-processed imagenet data')
                if not os.path.exists('data'):
                    os.makedirs('data')

                # Save the pre-processed images
                # Caution: This can take take a lot of space. Comment this part to discard saving.
                np.save(os.path.join('data', npy_data), X)

            else:
                print('>> Pre-processed imagenet data detected')
                X = np.load(datafile)
            # Running universal perturbation
            for x in range(995):
                v = targeted_perturbation(X, f, grad_fs, delta=0.25,max_iter_uni=10,target=target)

                # Saving the universal perturbation
                file_perturbation = os.path.join('data','precomputing_perturbations', npy_perturbations)
                np.save(os.path.join(file_perturbation), v)
                target+=1

        else:
            print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
            v = np.load(file_perturbation)

        print('>> Testing the targeted universal perturbation on an image')

        # Test the perturbation on the image
        labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')

        image_original = preprocess_image_batch([path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb")
        str_label_original =img2str(f=f,img=image_original)

        # Clip the perturbation to make sure images fit in uint8

        image_perturbed = avg_add_clip_pert(image_original,v)
        label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
        str_label_perturbed = img2str(f=f,img=image_perturbed)

        # Additional
        pre_file_perturbation = os.path.join('data','precomputing_perturbations', npy_perturbations)
        pre_v = np.load(file_perturbation)
        # import matplotlib
        # noise = pre_v/255.0
        # matplotlib.image.imsave('noise.png', noise.reshape(224,224,3))

        X = np.load(os.path.join('data', npy_data))
        target_fooling_rate = target_fooling_rate_calc(v=pre_v, dataset=X, f=f, target=target)
        print("")
        print('TARGET FOOLING RATE = ', target_fooling_rate)

        # Show original and perturbed image
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(undo_image_avg(image_original[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_original)

        # plt.subplot(1, 2, 2)
        # plt.imshow(undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
        # plt.title(str_label_perturbed)

        str_label_ = img2str(f=f, img=pre_v.reshape(224,224,3))
        plt.subplot(1, 2, 2)
        plt.imshow(undo_image_avg(pre_v.reshape(224,224,3)).astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_)

        plt.show()
