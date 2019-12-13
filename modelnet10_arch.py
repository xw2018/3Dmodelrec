"""
This file is supposed to contain the architecture in each function

If there are new architectures to try a new function should be started for each

e.g. if you're going to try with more capsule layers or conv layers

The motivation is to keep kinds of changes separate from one another

e.g. once you've settled on an architecture, it should be simple to gridsearch/hillclimb to the optimal hyperparamters like learning rate or number of capsules
"""
import numpy as np
import os
import sys
from keras import backend as K
from keras.callbacks import (TensorBoard,
                             EarlyStopping,
                             ReduceLROnPlateau,
                             CSVLogger)
from keras.layers import Conv3D, Dense, Reshape, Add, Input,MaxPooling3D
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.utils import to_categorical, multi_gpu_model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf

from capsulenet import margin_loss
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

from data import load_data
from utils import upsample_classes, stratified_shuffle
from results import process_results, _accuracy


NUM_EPOCHS = 5
NAME="ModelNet10"



def base_model(x_train, y_train, x_test, y_test, target_names,
               model_name='1c1p_true4_1cDRDL48dim10epoch_nodensemodel',
               dim_sub_capsule=48,
               dim_primary_capsule=8,
               n_channels=4,
               primary_cap_kernel_size=9,
               first_layer_kernel_size=9,
               conv_layer_filters=256,
               gpus=1, cv=False):
    model_name = "ModelNet10" + '_' + model_name
    n_class = y_test.shape[1]
    input_shape = (30, 30, 30, 1)


    ##### If using multiple GPUS ##########
    # with tf.device("/cpu:0"):全图版本
    # def make_model():
    #     x = Input(shape=input_shape)
    #     conv1 = Conv3D(filters=conv_layer_filters, kernel_size=first_layer_kernel_size, strides=1,
    #                                   padding='valid', activation='relu', name='conv1')(x)
    #     conv2=Conv3D(filters=96,kernel_size=5,strides=1,padding='valid',activation='relu',name='conv2')(conv1)
    #     primarycaps = PrimaryCap(conv2, dim_capsule=dim_primary_capsule, n_channels=n_channels,
    #                                                       kernel_size=primary_cap_kernel_size, strides=2, padding='valid',
    #                                                       name='primarycap_conv3d')
    #     sub_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_sub_capsule,
    #                                                      routings=3, name='sub_caps')(primarycaps)
    #     out_caps = Length(name='capsnet')(sub_caps)
    #
    #     # Decoder network
    #     y = Input(shape=(n_class,))
    #     masked_by_y = Mask()([sub_caps, y])
    #     masked = Mask()(sub_caps)
    #
    #     # shared decoder model in training and prediction
    #     decoder = Sequential(name='decoder')
    #     decoder.add(Dense(512, activation='relu',
    #                       input_dim=dim_sub_capsule*n_class))
    #     decoder.add(Dense(1024, activation='relu'))
    #     decoder.add(Dense(np.prod(input_shape), activation='sigmoid'))
    #     decoder.add(Reshape(target_shape=input_shape, name='out_recon'))
    #
    #
    #     ### Models for training and evaluation (prediction and actually using)
    #     train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
    #     eval_model = Model(x, [out_caps, decoder(masked)])
    #
    #     ### manipulate model can be used to visualize activation maps for specific classes
    #     noise = Input(shape=(n_class, dim_sub_capsule))
    #     noised_sub_caps = Add()([sub_caps, noise])
    #     masked_noised_y = Mask()([noised_sub_caps, y])
    #     manipulate_model = Model([x, y, noise], decoder(masked_noised_y))
    #     return train_model, eval_model, manipulate_model


    #没有全链接层版本
    def make_model():
        x = Input(shape=input_shape)

        # conv1 = Conv3D(filters=48, kernel_size=3, strides=1,
        #                padding='valid', activation='relu', name='conv1')(x)
        #
        # conv2 = Conv3D(filters=55, kernel_size=3, strides=1,
        #                padding='valid', activation='relu', name='conv2')(conv1)
        # m1=MaxPooling3D(pool_size=(2, 2, 2), strides=2)(conv2)
        # conv3 = Conv3D(filters=62, kernel_size=3, strides=1,
        #                padding='valid', activation='relu', name='conv3')(m1)
        # conv4 = Conv3D(filters=69, kernel_size=3, strides=1,
        #                padding='valid', activation='relu', name='conv4')(conv3)
        # m2=MaxPooling3D(pool_size=(2,2,2),strides=2)(conv4)
        conv1 = Conv3D(filters=conv_layer_filters, kernel_size=first_layer_kernel_size, strides=1,
                                      padding='valid', activation='relu', name='conv1')(x)
        m=MaxPooling3D(pool_size=(4,4,4), strides=1)(conv1)
        conv2=Conv3D(filters=96,kernel_size=5,strides=1,padding='valid',activation='relu',name='conv2')(m)
        primarycaps = PrimaryCap(conv2, dim_capsule=dim_primary_capsule, n_channels=n_channels,
                                                          kernel_size=primary_cap_kernel_size, strides=2, padding='valid',
                                                          name='primarycap_conv3d')
        sub_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_sub_capsule,
                                                         routings=3, name='sub_caps')(primarycaps)
        out_caps = Length(name='capsnet')(sub_caps)

        # Decoder network
        y = Input(shape=(n_class,))
        # masked_by_y = Mask()([sub_caps, y])
        # masked = Mask()(sub_caps)

        # # shared decoder model in training and prediction
        # decoder = Sequential(name='decoder')
        # decoder.add(Dense(512, activation='relu',
        #                   input_dim=dim_sub_capsule*n_class))
        # decoder.add(Dense(1024, activation='relu'))
        # decoder.add(Dense(np.prod(input_shape), activation='sigmoid'))
        # decoder.add(Reshape(target_shape=input_shape, name='out_recon'))


        ### Models for training and evaluation (prediction and actually using)
        # train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
        # eval_model = Model(x, [out_caps, decoder(masked)])
        train_model = Model([x, y], [out_caps])
        eval_model = Model(x, [out_caps])

        ### manipulate model can be used to visualize activation maps for specific classes
        noise = Input(shape=(n_class, dim_sub_capsule))
        noised_sub_caps = Add()([sub_caps, noise])
        masked_noised_y = Mask()([noised_sub_caps, y])
        manipulate_model = Model([x, y, noise])
        return train_model, eval_model, manipulate_model


    if gpus > 1:
        with tf.device("/cpu:0"):
            train_model, eval_model, manipulate_model = make_model()
    else:
        train_model, eval_model, manipulate_model = make_model()


    ################################ Compile and Train ###############################
    ##### IF USING MULTIPLE GPUS APPLY JUST BEFORE COMPILING ######
    if gpus > 1:
        train_model = multi_gpu_model(train_model, gpus=gpus) #### Adjust for number of gpus
    # train_model = multi_gpu_model(train_model, gpus=2) #### Adjust for number of gpus
    ##### IF USING MULTIPLE GPUS ######


#全图版本
    # lam_recon = .04
    # INIT_LR = .003
    # optimizer = Adam(lr=INIT_LR)
    # train_model.compile(optimizer,
    #                     loss=[margin_loss, 'mse'],
    #                     loss_weights=[1., lam_recon],
    #                     metrics={'capsnet': 'accuracy'})
    # call_back_path = 'logs/{}.log'.format(model_name)
    # tb = TensorBoard(log_dir=call_back_path)
    # csv = CSVLogger(os.path.join(call_back_path, 'training.log'))
    # early_stop = EarlyStopping(monitor='val_capsnet_acc',
    #                            min_delta=0,
    #                            patience=12,
    #                            verbose=1,
    #                            mode='auto')
    # reduce_lr = ReduceLROnPlateau(monitor='val_capsnet_acc', factor=0.5,
    #                               patience=3, min_lr=0.0001)

    lam_recon = .04
    INIT_LR = .003
    optimizer = Adam(lr=INIT_LR)
    train_model.compile(optimizer,
                        loss=[margin_loss],
                        loss_weights=[1.],
                        metrics={'capsnet': 'accuracy'})
    call_back_path = 'logs/{}.log'.format(model_name)
    tb = TensorBoard(log_dir=call_back_path)
    csv = CSVLogger(os.path.join(call_back_path, 'training.log'))
    early_stop = EarlyStopping(monitor='val_capsnet_acc',
                               min_delta=0,
                               patience=12,
                               verbose=1,
                               mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_capsnet_acc', factor=0.5,
                                  patience=3, min_lr=0.0001)
    def reset_weights(model):
        session = K.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
    if cv:
        if 'ModelNet40' in NAME:
            for i in range(10):
                train_model.fit([x_train[:500], y_train[:500]], [y_train[:500], x_train[:500]],
                                                batch_size=32, epochs=1)
                # this is only to get the network warmed up.
                restart_acc = _accuracy(eval_model, x_train[:500], y_train[:500])
                if not restart_acc > .4:
                    print("\n\n\n ### Doing random restart! ### \n\n\n")
                    reset_weights(train_model)
                    continue
                else:
                    print("\n\n\n ### not in local minimum! ### \n\n\n")
                    break
            if not restart_acc > .1:
                print("\n\n\n ### Couldn't get out of local minimum ### \n\n\n")


        train_model.fit([x_train, y_train], [y_train],
                        batch_size=10, epochs=NUM_EPOCHS,callbacks=[tb, csv, reduce_lr, early_stop])  # 原来是256
        # train_model.fit([x_train, y_train], [y_train, x_train],
        #                 batch_size=10, epochs=NUM_EPOCHS)  #原来是256
        # y_pred, x_recon = eval_model.predict(x_test, y_test)
        # return y_pred, x_recon
    else:
        train_model.fit([x_train, y_train], [y_train],
                        batch_size=10, epochs=NUM_EPOCHS,  # 原来是256
                        # validation_data=[[x_val, y_val], [y_val, x_val]],#没找到x_val,y_val
                        #                 callbacks=[tb, checkpointer])
                        callbacks=[tb, csv, reduce_lr, early_stop])

        # train_model.fit([x_train, y_train], [y_train, x_train],
        #                                 batch_size=10, epochs=NUM_EPOCHS,  #原来是256
        #                                 # validation_data=[[x_val, y_val], [y_val, x_val]],#没找到x_val,y_val
        #                 #                 callbacks=[tb, checkpointer])
        #                                 callbacks=[tb, csv, reduce_lr, early_stop])


    ################################ Process the results ###############################
    process_results(model_name, eval_model,
                    manipulate_model, x_test, y_test, target_names,
                    INIT_LR=INIT_LR,
                    lam_recon=lam_recon,
                    NUM_EPOCHS=NUM_EPOCHS,
                    dim_sub_capsule=dim_sub_capsule,
                    dim_primary_capsule=dim_primary_capsule,
                    n_channels=n_channels,
                    primary_cap_kernel_size=primary_cap_kernel_size,
                    first_layer_kernel_size=first_layer_kernel_size,
                    conv_layer_filters=conv_layer_filters)

def main():
    # # Load the data
    # (x_train, y_train), (x_test, y_test), target_names = load_data(NAME)
    # x_train, y_train, x_val, y_val = stratified_shuffle(x_train, y_train, test_size=.1)
    # x_train, y_train = upsample_classes(x_train, y_train)
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # y_val = to_categorical(y_val)
    # if 'test' in sys.argv:
    #     print('RUNNING IN TEST MODE')
    #     x_train, y_train, x_val, y_val, x_test, y_test = \
    #         x_train[:512], y_train[:512], x_val[:100], y_val[:100], x_test[:100], y_test[:100]

    # NUM_EPOCHS = 2
    from sklearn.model_selection import ParameterGrid
    # initial shot
    # param_grid = {
    #     "first_layer_kernel_size": [9],
    #     "conv_layer_filters": [24, 48],
    #     "primary_cap_kernel_size": [9, 7],
    #     "dim_primary_capsule": [4, 8],
    #     "n_channels": [4],
    #     "dim_sub_capsule": [8, 16],
    # }
    # param_grid = ParameterGrid(param_grid)
    # for params in param_grid:
    #     try:
            # base_model(x_train, y_train, x_test, y_test,
            #            'base_model',
            #            gpus=6,
            #            **params)
    #     except:
    #         print('whoops')
    # hey maybe 1 channel? Never thought to try it til now
    # base_model(x_train, y_train, x_test, y_test,
    #            'out_of_box',
    #            gpus=6, conv_layer_filters=256, dim_primary_capsule=8,
    #            dim_sub_capsule=8, n_channels=1)
    # param_grid = {
    #     "first_layer_kernel_size": [9],
    #     "conv_layer_filters": [128],
    #     "primary_cap_kernel_size": [9],
    #     "dim_primary_capsule": [4, 8],
    #     "n_channels": [2, 3],
    #     "dim_sub_capsule": [16],
    # }
    # param_grid = ParameterGrid(param_grid)
    # for params in param_grid:
    #     try:
            # base_model(x_train, y_train, x_test, y_test,
            #            'base_model',
            #            gpus=8,
            #            **params)
    #     except:
    #         print('whoops')

    # base_model(x_train, y_train, x_test, y_test,
    #            'out_of_box_same_settings',
    #            gpus=8, conv_layer_filters=256, dim_primary_capsule=8,
    #            dim_sub_capsule=16, n_channels=32)

    # this needs to get cleaned up, but I don't have the time right now sorry
    from sklearn.model_selection import StratifiedKFold
    from results import _accuracy
    #### Cross validated
    #### was also used against modelnet40
    (x_train, y_train), (x_test, y_test), target_names = load_data("/home/mushan/code/3d_model_retriever-master/modelnet10r/ModelNet10b")
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    # x_train, y_train = upsample_classes(x_train, y_train)
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # base_model(x_train, y_train, x_test, y_test, target_names,
    #            # model_name='best_cv',
    #            gpus=1, conv_layer_filters=256, dim_primary_capsule=8,  # gpu改为1个
    #            dim_sub_capsule=8, n_channels=1,
    #            cv=True)
    cvscores = []
    for train, test in kfold.split(x, y):
        x_train, y_train = x[train], y[train]
        x_train, y_train = upsample_classes(x_train, y_train)
        x_test, y_test = x[test], y[test]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        base_model(x_train, y_train, x_test, y_test, target_names,
                   # model_name='best_cv',
                   gpus=1, conv_layer_filters=256, dim_primary_capsule=8,#gpu改为1个
                   dim_sub_capsule=8, n_channels=1,
                   cv=True)
        K.clear_session()
        tf.reset_default_graph()

if __name__ == '__main__':
    main()







"""
original

This file is supposed to contain the architecture in each function

If there are new architectures to try a new function should be started for each

e.g. if you're going to try with more capsule layers or conv layers

The motivation is to keep kinds of changes separate from one another

e.g. once you've settled on an architecture, it should be simple to gridsearch/hillclimb to the optimal hyperparamters like learning rate or number of capsules
"""
# import numpy as np
# import os
# import sys
#
# from keras.callbacks import (TensorBoard,
#                              EarlyStopping,
#                              ReduceLROnPlateau,
#                              CSVLogger)
# from keras.layers import Conv3D, Dense, Reshape, Add, Input
# from keras.models import Sequential, Model
# from keras.optimizers import SGD, Adam
# from keras import backend as K
# from keras.utils import to_categorical, multi_gpu_model
# from keras.utils.vis_utils import plot_model
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# from capsulenet import margin_loss
# from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
#
# from data import load_data
# from utils import upsample_classes, stratified_shuffle
# from results import process_results, _accuracy
#
#
# NAME = 'ModelNet10'
# NUM_EPOCHS = 5
#
#
#
#
# def base_model(x_train, y_train, x_test, y_test, target_names,
#                model_name='ori_model_306090',
#                dim_sub_capsule=16,
#                dim_primary_capsule=8,
#                n_channels=4,
#                primary_cap_kernel_size=9,
#                first_layer_kernel_size=9,
#                conv_layer_filters=256,
#                gpus=1, cv=False):
#     model_name = NAME + '_' + model_name
#     n_class = y_test.shape[1]
#     input_shape = (30, 30, 30, 1)
#
#
#     ##### If using multiple GPUS ##########
#     # with tf.device("/cpu:0"):
#     def make_model():
#         x = Input(shape=input_shape)
#         conv1 = Conv3D(filters=conv_layer_filters, kernel_size=first_layer_kernel_size, strides=1,
#                                       padding='valid', activation='relu', name='conv1')(x)
#         primarycaps = PrimaryCap(conv1, dim_capsule=dim_primary_capsule, n_channels=n_channels,
#                                                           kernel_size=primary_cap_kernel_size, strides=2, padding='valid',
#                                                           name='primarycap_conv3d')
#         sub_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_sub_capsule,
#                                                          routings=3, name='sub_caps')(primarycaps)
#         out_caps = Length(name='capsnet')(sub_caps)
#
#         # Decoder network
#         y = Input(shape=(n_class,))
#         masked_by_y = Mask()([sub_caps, y])
#         masked = Mask()(sub_caps)
#
#         # shared decoder model in training and prediction
#         decoder = Sequential(name='decoder')
#         decoder.add(Dense(512, activation='relu',
#                           input_dim=dim_sub_capsule*n_class))
#         decoder.add(Dense(1024, activation='relu'))
#         decoder.add(Dense(np.prod(input_shape), activation='sigmoid'))
#         decoder.add(Reshape(target_shape=input_shape, name='out_recon'))
#
#
#         ### Models for training and evaluation (prediction and actually using)
#         train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
#         eval_model = Model(x, [out_caps, decoder(masked)])
#
#         ### manipulate model can be used to visualize activation maps for specific classes
#         noise = Input(shape=(n_class, dim_sub_capsule))
#         noised_sub_caps = Add()([sub_caps, noise])
#         masked_noised_y = Mask()([noised_sub_caps, y])
#         manipulate_model = Model([x, y, noise], decoder(masked_noised_y))
#         return train_model, eval_model, manipulate_model
#
#     if gpus > 1:
#         with tf.device("/cpu:0"):
#             train_model, eval_model, manipulate_model = make_model()
#     else:
#         train_model, eval_model, manipulate_model = make_model()
#
#
#     ################################ Compile and Train ###############################
#     ##### IF USING MULTIPLE GPUS APPLY JUST BEFORE COMPILING ######
#     if gpus > 1:
#         train_model = multi_gpu_model(train_model, gpus=gpus) #### Adjust for number of gpus
#     # train_model = multi_gpu_model(train_model, gpus=2) #### Adjust for number of gpus
#     ##### IF USING MULTIPLE GPUS ######
#
#     lam_recon = .04
#     INIT_LR = .003
#     optimizer = Adam(lr=INIT_LR)
#     train_model.compile(optimizer,
#                         loss=[margin_loss, 'mse'],
#                         loss_weights=[1., lam_recon],
#                         metrics={'capsnet': 'accuracy'})
#
#     call_back_path = 'logs/{}.log'.format(model_name)
#     tb = TensorBoard(log_dir=call_back_path)
#     csv = CSVLogger(os.path.join(call_back_path, 'training.log'))
#     early_stop = EarlyStopping(monitor='val_capsnet_acc',
#                                min_delta=0,
#                                patience=12,
#                                verbose=1,
#                                mode='auto')
#     reduce_lr = ReduceLROnPlateau(monitor='val_capsnet_acc', factor=0.5,
#                                   patience=3, min_lr=0.0001)
#     def reset_weights(model):
#         session = K.get_session()
#         for layer in model.layers:
#             if hasattr(layer, 'kernel_initializer'):
#                 layer.kernel.initializer.run(session=session)
#     if cv:
#         if 'ModelNet40' in NAME:
#             for i in range(10):
#                 train_model.fit([x_train[:500], y_train[:500]], [y_train[:500], x_train[:500]],
#                                                 batch_size=32, epochs=1)
#                 # this is only to get the network warmed up.
#                 restart_acc = _accuracy(eval_model, x_train[:500], y_train[:500])
#                 if not restart_acc > .4:
#                     print("\n\n\n ### Doing random restart! ### \n\n\n")
#                     reset_weights(train_model)
#                     continue
#                 else:
#                     print("\n\n\n ### not in local minimum! ### \n\n\n")
#                     break
#             if not restart_acc > .1:
#                 print("\n\n\n ### Couldn't get out of local minimum ### \n\n\n")
#         train_model.fit([x_train, y_train], [y_train, x_train],
#                         batch_size=10, epochs=NUM_EPOCHS,callbacks=[tb, csv, reduce_lr, early_stop])
#         # y_pred, x_recon = eval_model.predict(x_test, y_test)
#         # return y_pred, x_recon
#     else:
#         train_model.fit([x_train, y_train], [y_train, x_train],
#                                         batch_size=10, epochs=NUM_EPOCHS,
#                         #                 callbacks=[tb, checkpointer])
#                                         callbacks=[tb, csv, reduce_lr, early_stop])
#
#
#     ################################ Process the results ###############################
#     process_results(model_name, eval_model,
#                     manipulate_model, x_test, y_test, target_names,
#                     INIT_LR=INIT_LR,
#                     lam_recon=lam_recon,
#                     NUM_EPOCHS=NUM_EPOCHS,
#                     dim_sub_capsule=dim_sub_capsule,
#                     dim_primary_capsule=dim_primary_capsule,
#                     n_channels=n_channels,
#                     primary_cap_kernel_size=primary_cap_kernel_size,
#                     first_layer_kernel_size=first_layer_kernel_size,
#                     conv_layer_filters=conv_layer_filters)
#
# def main():
#     # # Load the data
#     # (x_train, y_train), (x_test, y_test), target_names = load_data(NAME)
#     # x_train, y_train, x_val, y_val = stratified_shuffle(x_train, y_train, test_size=.1)
#     # x_train, y_train = upsample_classes(x_train, y_train)
#     # y_train = to_categorical(y_train)
#     # y_test = to_categorical(y_test)
#     # y_val = to_categorical(y_val)
#     # if 'test' in sys.argv:
#     #     print('RUNNING IN TEST MODE')
#     #     x_train, y_train, x_val, y_val, x_test, y_test = \
#     #         x_train[:512], y_train[:512], x_val[:100], y_val[:100], x_test[:100], y_test[:100]
#
#     # NUM_EPOCHS = 2
#     from sklearn.model_selection import ParameterGrid
#     # initial shot
#     # param_grid = {
#     #     "first_layer_kernel_size": [9],
#     #     "conv_layer_filters": [24, 48],
#     #     "primary_cap_kernel_size": [9, 7],
#     #     "dim_primary_capsule": [4, 8],
#     #     "n_channels": [4],
#     #     "dim_sub_capsule": [8, 16],
#     # }
#     # param_grid = ParameterGrid(param_grid)
#     # for params in param_grid:
#     #     try:
#             # base_model(x_train, y_train, x_test, y_test,
#             #            'base_model',
#             #            gpus=6,
#             #            **params)
#     #     except:
#     #         print('whoops')
#     # hey maybe 1 channel? Never thought to try it til now
#     # base_model(x_train, y_train, x_test, y_test,
#     #            'out_of_box',
#     #            gpus=6, conv_layer_filters=256, dim_primary_capsule=8,
#     #            dim_sub_capsule=8, n_channels=1)
#     # param_grid = {
#     #     "first_layer_kernel_size": [9],
#     #     "conv_layer_filters": [128],
#     #     "primary_cap_kernel_size": [9],
#     #     "dim_primary_capsule": [4, 8],
#     #     "n_channels": [2, 3],
#     #     "dim_sub_capsule": [16],
#     # }
#     # param_grid = ParameterGrid(param_grid)
#     # for params in param_grid:
#     #     try:
#             # base_model(x_train, y_train, x_test, y_test,
#             #            'base_model',
#             #            gpus=8,
#             #            **params)
#     #     except:
#     #         print('whoops')
#
#     # base_model(x_train, y_train, x_test, y_test,
#     #            'out_of_box_same_settings',
#     #            gpus=8, conv_layer_filters=256, dim_primary_capsule=8,
#     #            dim_sub_capsule=16, n_channels=32)
#
#     # this needs to get cleaned up, but I don't have the time right now sorry
#     from sklearn.model_selection import StratifiedKFold
#     from results import _accuracy
#     #### Cross validated
#     #### was also used against modelnet40
#     (x_train, y_train), (x_test, y_test), target_names = load_data("/home/mushan/code/3d_model_retriever-master/modelnet10r/ModelNet10b")
#     x = np.concatenate((x_train, x_test))
#     y = np.concatenate((y_train, y_test))
#     kfold = StratifiedKFold(n_splits=5, shuffle=True)
#     # cvscores = []
#     for train, test in kfold.split(x, y):
#         x_train, y_train = x[train], y[train]
#         x_train, y_train = upsample_classes(x_train, y_train)
#         x_test, y_test = x[test], y[test]
#         y_train = to_categorical(y_train)
#         y_test = to_categorical(y_test)
#         base_model(x_train, y_train, x_test, y_test, target_names,
#                    gpus=1, conv_layer_filters=256, dim_primary_capsule=8,
#                    dim_sub_capsule=8, n_channels=1,
#                    cv=True)
#         K.clear_session()
#         tf.reset_default_graph()
#
# if __name__ == '__main__':
#     main()
