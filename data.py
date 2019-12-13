import numpy as np
import os

import binvox_rw_py as binvox

newdirs=["modelnet10xz90","modelnet10xz150","modelnet10xz240"]
# newdirs=["modelnet10xz30","modelnet10xz60","modelnet10xz90"]

def _read_file(fp):
    with open(fp, 'rb') as f:
        return binvox.read_as_3d_array(f).data


def _train_test_split_paths(dp, sub_path:str):
    path = os.path.join(dp, sub_path)
    file_paths = [os.path.join(path, i)
                  for i in os.listdir(path)]
    binvox_paths = list(filter(lambda x: '.binvox' in x, file_paths))
    return binvox_paths


def load_data(dp):
    """
    Args:
        dp: path to ModelNetXX folder which contains train and test folders

    Return:
        (x_train, y_train), (x_test, y_test), target_names
    """
    assert 'ModelNet' in dp, "directory should be ModelNet!!!"

    label_dirs = list(os.scandir(dp))
    # label_dirs = [i for i in os.scandir(dp) if os.path.isdir(i)]   不注释掉无法运行
    target_names = [i.name for i in label_dirs]

    # associating the filepaths with their respective targets
    # for both train and test sets
    train_paths = []
    test_paths = []
    y_train = []
    y_test = []
    # for i, dir_path in enumerate(label_dirs):
    #     for path in _train_test_split_paths(dir_path.path, 'train'):
    #         train_paths.append(path)
    #         y_train.append(i)
    #     for path in _train_test_split_paths(dir_path.path, 'test'):
    #         test_paths.append(path)
    #         y_test.append(i)

    label_dirs1 = list(os.scandir("modelnet10r//ModelNet10b"))
    target_names = [i.name for i in label_dirs1]
    for name in target_names:
        for i, dir_path in enumerate(label_dirs1):
            if name in str(dir_path):
                for path in _train_test_split_paths(dir_path.path, 'train'):
                    train_paths.append(path)
                    y_train.append(i)
                for path in _train_test_split_paths(dir_path.path, 'test'):
                    test_paths.append(path)
                    y_test.append(i)
        for p in newdirs:
            labeldirs = list(os.scandir("modelnet10r//{}".format(p)))
            for i, dir_path in enumerate(labeldirs):
                if name in str(dir_path):  # if name in dirname,do it.if not,go to next dir
                    for path in _train_test_split_paths(dir_path.path, 'train'):
                        train_paths.append(path)
                        y_train.append(i)
    
    # converting binvox to numpy and reshape
    x_train = [_read_file(i) for i in train_paths]
    x_train = np.array(x_train).reshape(-1, 30, 30, 30, 1)

    x_test = [_read_file(i) for i in test_paths]
    x_test = np.array(x_test).reshape(-1, 30, 30, 30, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test), target_names


def load_custom_model(model_path):
    from keras.utils import CustomObjectScope
    from keras.models import load_model
    from capsulenet import margin_loss
    from capsulelayers import CapsuleLayer, Mask, Length

    with CustomObjectScope({'CapsuleLayer': CapsuleLayer,
                            'Mask': Mask, 'Length': Length,
                            'margin_loss': margin_loss}):
        return load_model(model_path)
