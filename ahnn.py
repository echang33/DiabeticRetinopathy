#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:54:14 2022

@author: r2gawa
"""

import numpy as np 
from tensorflow import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output
import datetime
import os
import tensorflow as tf
import random
import ahutils as au



class printLearningRate(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr.read_value()
        print("\nEpoch: {}. Learning Rate is {}".format(epoch, lr))
        

class plotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def __init__(self, basename, workspace = '.'):
        self.basename = basename
        if workspace[-1] == '/':
            self.workdir = workspace
        else:
            self.workdir = workspace+'/'
        
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            # print('KOKO: mtric = ', metric)
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        # if epoch % 100 == 0:
        if True:
            metrics = [x for x in logs if 'val' not in x]
            
            f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
            clear_output(wait=True)
    
            for i, metric in enumerate(metrics):
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics[metric], 
                            label=metric)
                if 'val_' + metric in logs and logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2), 
                                self.metrics['val_' + metric], 
                                label='val_' + metric)
                    
                axs[i].legend()
                axs[i].grid()
           
            plt.tight_layout()
            print("saved",self.workdir+self.basename+'.png')
            plt.savefig(str(self.workdir)+str(self.basename), dpi=300)
            plt.show()
            

class SaveModelWeights(keras.callbacks.Callback):
    def __init__(self, basename, workspace='.', everyepoch=10):
        self.epochs_gap = everyepoch
        self.basename = basename
        if workspace[-1] == '/':
            self.workdir = workspace
        else:
            self.workdir = workspace+'/'
        self.lastweight = ''
        
    def on_training_begin(self, logs={}):
        pass
        # savedname = self.basename+"_%05d" % (0)+"-"+datetime.datetime.now().strftime("%y-%m-%d-%H.%M.%S")+".h5"
        # self.model.save(savedname)
        # print("Saved model to disk: ",savedname)
        
    def on_epoch_end(self, epoch, longs={}):
        if epoch==0 or (epoch+1) % self.epochs_gap == 0:
            savedname = self.workdir+self.basename+"_%05d" % (epoch+1)+"-"+datetime.datetime.now().strftime("%y-%m-%d-%H.%M.%S")+".h5"
            self.model.save(savedname)
            print("Saved model to disk: ",savedname)
        else:
            savedname = self.workdir+self.basename+"_%05d" % (epoch+1)+"-"+datetime.datetime.now().strftime("%y-%m-%d-%H.%M.%S")+".h5"
            self.model.save(savedname)
            print("Saved model to disk: ",savedname)
            
            # remove last weight file
            if os.path.exists(self.lastweight):
                os.remove(self.lastweight)
            
            # update last weight file
            self.lastweight = savedname
            
            
            
import pathlib


def no_augment(images, labels):
    return (images, labels)


def small_changes(images):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_contrast(images, 0.95, 1.0)
    images = tf.image.random_brightness(images, 0.05)
    return images


class ManageData:
    def __init__(self, datadir, ext='png', validation_split = 0.2, seed=None): #, exts=['png','jpg']):
        self.valid_split = validation_split
        self.data_dir = pathlib.Path(datadir)
        
        # self.image_list = []
        # for ext in exts:
        #     self.image_list += list(self.data_dir.glob('*/*.'+ext))
        self.image_list = list(self.data_dir.glob('*/*.'+ext))
        
        if not seed:
            random.shuffle(self.image_list)
        else:
            random.Random(seed).shuffle(self.image_list)
        self.clanames = np.array(sorted([item.name for item in self.data_dir.glob('*') if os.path.isdir(datadir+'/'+item.name)]))
        self.count = 0
        self.ext = ext
        self.seed = seed
        
    def samples(self,num=5):
        """
        print out examples
        
        Parameters
        ----------
        num : integer, optional
            sample size. The default is 5.

        Returns
        -------
        None.

        """
        for f in self.image_list.take(num):
            print(f.numpy())
    
    def classnames(self):
        """
        return a list of class names

        Returns
        -------
        list
            a list of class names.

        """
        return self.clanames
    
    def select_class(self, aclassname):
        """
        extract dataset belonging to a class

        Parameters
        ----------
        aclassname : string
            a class name.

        Returns
        -------
        list
            a list of image file names.

        """
        sdata = []
        for item in self.image_list:
            if aclassname in str(item):
                sdata.append(str(item))
        return tf.data.Dataset.list_files(sdata, shuffle=True)
    
    def select_ds(self, subset, validation_split=None):
        """
        select dataset --- all, training, or validation

        Parameters
        ----------
        subset : string
            'all', 'training', or 'validation'.

        Returns
        -------
        list
            a list of image file names.

        """
        if validation_split:
            self.valid_split = validation_split
            
        image_count = len(self.image_list)
        val_size = int(image_count * self.valid_split)
        train_size = image_count - val_size
        print("training: %d, validation: %d" % (train_size, val_size))

        if subset == 'training':
            lst = [str(x) for x in self.image_list[val_size:]]
            train_ds = tf.data.Dataset.list_files(lst, shuffle=True)
            return train_ds
        elif subset == 'validation':
            lst = [str(x) for x in self.image_list[:val_size]]
            val_ds = tf.data.Dataset.list_files(lst, shuffle=True)
            return val_ds
        else:
            lst = [str(x) for x in self.image_list]
            all_ds = tf.data.Dataset.list_files(lst, shuffle=True)
            return all_ds
        
    def breakdown(self, lst, printout = True):
        """
        print breakdown of dataset

        Parameters
        ----------
        lst : list
            a list of image file names.

        Returns
        -------
        None.

        """
        total = tf.data.experimental.cardinality(lst).numpy()
        # print("total number of images: ", total)
        distribution = []
        for aclass in self.clanames:
            brkdwn = []
            for one in lst:
                # print(one.numpy())
                if str(one.numpy()).split('/')[-2] == aclass:
                # if aclass in str(one.numpy()):
                    brkdwn = brkdwn + [str(one.numpy())]
            if printout:
                print("%15s ---- %5d (%4.1f%%)" % (aclass, len(brkdwn), float(len(brkdwn)*100)/total))
            distribution.append(len(brkdwn))
        return distribution

    
    def class_weights(self, lst, k=1.0, max_weight = 1000.0):
        total = tf.data.experimental.cardinality(lst).numpy()
        dist = self.breakdown(lst, False)
        numclasses = len(self.clanames)
        
        classweights = {}
        for i, classname in enumerate(self.clanames):
            print("%d : %s" % (i, classname))
            weight = (1.0/float(numclasses)) * (float(total)/float(dist[i]))
            weight *= k
            if weight > max_weight:
                weight = max_weight
            classweights[i] = weight
        return classweights


    def balance_list(self, lst, workdir='.', conv_func=small_changes):
        dist = self.breakdown(lst, printout=True)
        print(dist)
        max_num = max(dist)
        
        add_data = []
        output = [str(x.numpy(), 'utf-8') for x in lst]
        for i,aclass in enumerate(self.clanames):
            brkdwn = []
            for one in lst:
                one = str(one.numpy(), 'utf-8')
                # print(item.numpy())
                if aclass in one:
                    brkdwn.append(one)
            tobeadded = max_num - dist[i]
            print("%15s" % self.clanames[i], '---- add %5d,' % tobeadded, " goal: %5d" % max_num, "(%5d)" % dist[i])
            if tobeadded > 0:
                add_data = add_data + random.choices(brkdwn, k=tobeadded)
        
        print("data to be added : ", len(add_data))
        for datum in add_data:
            output.append(str(self.generate_dupfile(datum, workdir, conv_func)))
        # for x in output:
        #     print(":::",x)
        return tf.data.Dataset.list_files(output, shuffle=True)


    def generate_dupfile(self, file, destination, conv_func):
        (tmp, basename, ext) = au.parsefilename(file, ['jpg', 'png'])
        # print("parsing filename: %s" % file)
        # print("  ==> %s, %s, %s" % (tmp, basename, ext))
        
        parts = tf.strings.split(file, os.path.sep)
        # The second to last is the class-directory
        classname = parts[-2]
        # filename  = parts[-1]
        
        # print("input is '%s'" % file)
        img = tf.io.read_file(file)
        img = self._decode_img(img, ext)
        img = conv_func(img)
        img = self._encode_img(img, ext)
            
        if destination[-1]!='/':
            destination += '/'
            
        outname = destination + classname + '/' + basename + '-%d' % self.count + '.' + ext
        outname = str(outname.numpy(), 'utf-8')
        # print(file, ' ---> ', outname)
        self.count += 1
        
        tf.io.write_file(outname, img)
        return outname
        

    def _get_label(self, file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.clanames
        # Integer encode the label
        return tf.argmax(one_hot)
    
    def _encode_img(self, img, ext):
        # Convert a 3D uint8 tensor to the compressed image 
        if ext == 'jpg':
            img = tf.io.encode_jpeg(img)
        elif ext == 'png':
            img = tf.io.encode_png(img)
        return img
    
    def _decode_img(self, img, ext):
        # Convert the compressed string to a 3D uint8 tensor
        if ext == 'jpg':
            img = tf.io.decode_jpeg(img, channels=3)
        elif ext == 'png':
            img = tf.io.decode_png(img, channels=3)
        return img
    
    def _process_path(self, file_path):
        # (tmp, basename, ext) = au.parsefilename(str(file_path.numpy(), 'utf-8'), ['jpg', 'png'])
        # print(basename, ext)
        
        label = self._get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        
        img = self._decode_img(img, ext=self.ext)
        return img, label

    def dpmap(self, ds, batch_size=32, augment_func=no_augment):
        """
        convert a list of image file names to dataset

        Parameters
        ----------
        ds : list
            a list of image file names.
        batch_size : int, optional
            batch size. The default is 32.
        augment_func : function for augmentation, optional
            DESCRIPTION. The default is no_augment.

        Returns
        -------
        ds : TYPE
            DESCRIPTION.

        """
        AUTOTUNE = tf.data.AUTOTUNE
        ds = ds.map(self._process_path, num_parallel_calls=AUTOTUNE)
        
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.map(augment_func, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds


""" 
# EXAMPLES 


ext = 'png'
a = ManageData('/Volumes/ExtremeSSD/DeepLearning/Diabetic_Retinopathy/colored/colored_images', ext, 0.3)

# a.samples(10)

print(a.classnames())


# for cname in a.classnames():
#     ds = a.select_class(cname)
#     print("%s ---- %5d" % (cname, len(ds)))
#     for f in ds.take(5):
#         print(f.numpy())


# all_ds = a.select_ds('all')
# print("%s ---- %5d" % ('validation', len(all_ds)))
# for f in all_ds.take(5):
#     print(f.numpy())
# a.breakdown(all_ds)
# a.breakdown(all_ds)
# a.breakdown(all_ds)

# all_ds = a.dpmap(all_ds, 32)

# for image, label in all_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())
    

tr_ds = a.select_ds('training', 0.99)
print("%s ---- %5d" % ('training', len(tr_ds)))
for f in tr_ds.take(5):
    print(f.numpy())
    
print(a.class_weights(tr_ds))
# print break down of ds
# print(a.breakdown(tr_ds))
# print("%s ---- %5d" % ('training', len(tr_ds)))
# for f in tr_ds.take(5):
#     print(f.numpy())
# print(a.breakdown(tr_ds))
# print("%s ---- %5d" % ('training', len(tr_ds)))
# for f in tr_ds.take(5):
#     print(f.numpy())
# print(a.breakdown(tr_ds))
# print("%s ---- %5d" % ('training', len(tr_ds)))
# for f in tr_ds.take(5):
#     print(f.numpy())

# convert to paired dataset
# tr_ds = a.dpmap(tr_ds)

# # print('koko', tr_ds.class_names)


# for image, label in tr_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())


# val_ds = a.select_ds('validation')
# print("%s ---- %5d" % ('validation', len(val_ds)))
# for f in val_ds.take(5):
#     print(f.numpy())
# a.breakdown(val_ds)

# val_ds = a.dpmap(val_ds)

# for image, label in val_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())
    

# bal_train_lst = a.balance_list(tr_ds)
# print("%s ---- %5d" % ('balanced training', len(bal_train_lst)))
# a.breakdown(bal_train_lst)
    

"""

