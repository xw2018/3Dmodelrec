from capsulelayers import CapsuleLayer,Mask,Length
import os
import numpy as np
from data import _read_file,_train_test_split_paths
from keras.models import load_model
from keras.utils import to_categorical
from results import save_confusion_matrix
def mreload(modelname):
    # newdirs=["ModelNet10b","modelnet10z30","modelnet10z60","modelnet10z90","modelnet10z120","modelnet10z150","modelnet10z180","modelnet10z210","modelnet10z240","modelnet10z270","modelnet10z300","modelnet10z330"]
    newdirs=["ModelNet10b","modelnet10z30","modelnet10z60","modelnet10z90",
             "modelnet10z120","modelnet10z150","modelnet10z180","modelnet10z210",
             "modelnet10z240","modelnet10z270","modelnet10z300","modelnet10z330"]
    # newdirs=["ModelNet10b","modelnet10z5","modelnet10z10","modelnet10z15",
    #          "modelnet10z20","modelnet10z25","modelnet10z30", "modelnet10z60", "modelnet10z90",
    #              "modelnet10z120","modelnet10z150","modelnet10z180","modelnet10z210",
    #              "modelnet10z240","modelnet10z270","modelnet10z300","modelnet10z330"]
    labels = list(os.scandir("modelnet10r/ModelNet10b"))
    target_names = [i.name for i in labels]

    #y_test初始化
    y_test = []
    for i, dir_path in enumerate(labels):
        for path in _train_test_split_paths(dir_path.path, 'test'):
            y_test.append(i)
    y_test = np.array(y_test)
    y_test = to_categorical(y_test)

    #读取模型
    modelpath="/home/mushan/code/ori/3d_model_retriever-master/results/"+modelname+"/models/eval_model.hdf5"
    evalmodel = load_model(modelpath,custom_objects={"CapsuleLayer": CapsuleLayer, "Mask": Mask, "Length": Length})

    #结果目录
    dirpath="results//"+modelname+"//"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    #保存模型结构信息
    with open(dirpath+"struinfo.txt","w")as f:
        f.write(evalmodel.to_yaml())


    #保存旋转信息
    with open(dirpath+"rotation.txt","w") as f:
           for newdir in newdirs:
                test_paths=[]
                label_dirs = list(os.scandir("./modelnet10r/{}".format(newdir)))
                for i, dir_path in enumerate(label_dirs):
                    for path in _train_test_split_paths(dir_path.path, 'test'):
                        test_paths.append(path)
                x_test1 = [_read_file(i) for i in test_paths]
                x_test = np.array(x_test1).reshape(-1, 30, 30, 30, 1)
                # y_pred,x_recon=evalmodel.predict(x_test,batch_size=5,verbose=0)
                y_pred=evalmodel.predict(x_test,batch_size=5,verbose=0)
                test_accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
                confusion_matrix_png_path=dirpath+"{}_{}.png".format(newdir,str(round(test_accuracy,5)).replace('.',''))
                save_confusion_matrix(y_test,y_pred,target_names,confusion_matrix_png_path)
                print("{}_accuracy:{}\n".format(newdir,test_accuracy))
                f.write("{}_accuracy:{}\n".format(newdir,test_accuracy))
    os.system("xdg-open {}".format(dirpath+"rotation.txt"))
    os.system("cp {} {}".format(dirpath+"rotation.txt","myreload/"+modelname+".txt"))

    print("end of rotation evalution")


def main():
    mreload("ModelNet10_1convDRDL5epoch_model_acc_093699_map_088667")

if __name__=="__main__":
    main()


#查看并保存模型的详细信息
# from keras.models import load_model
# from capsulelayers import CapsuleLayer,Mask,Length
# modelname="ModelNet10_2conv_acc_088436"
# modelpath="/home/xw/code/3d_model_retriever-master/3d_model_retriever-master/results/"+modelname+"/models/eval_model.hdf5"
# evalmodel = load_model(modelpath,custom_objects={"CapsuleLayer": CapsuleLayer, "Mask": Mask, "Length": Length})
# print(evalmodel.to_yaml())
# with open("results/{}".format(modelname)+"/"+modelname+"struinfo.txt","w") as f:
#     f.write(evalmodel.to_yaml())
