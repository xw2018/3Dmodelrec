import os
from data import _read_file,_train_test_split_paths






# M="../ModelNet10"
# labels=os.walk(M)
# mlx=["modelnet10z90.mlx","modelnet10z180.mlx","modelnet10z270.mlx"]
# m=["xz90","xz180","xz270"]
# for i in labels:
#     # print(i[0],i[1],i[2])
#     if i[1]==[]:
#         indirs=i[0]
#         if indirs.endswith("train"):
#             initems=i[2]
#             for j,k in zip(m,mlx):
#                 outdirs=indirs.replace("ModelNet10","modelnet10"+j)
#                 print(outdirs)
#                 if not os.path.exists(outdirs):
#                     os.makedirs(outdirs)
#                 for initem in initems:
#                     print("meshlabserver -i {} -o {} -s {}".format(indirs+'//'+initem,outdirs+'//'+initem,k))
#                     # os.system("meshlabserver -i {} -o {} -s {}".format(indirs+'//'+initem,outdirs+'//'+initem,k))
# print("end")

# os.system("meshlabserver -i {} -o {} -s {}".format("D:\\Program\\meshlab\\try\\input\\bathtub_0107.off","D:\\Program\\meshlab\\try\\output\\bathtub_0107.off","D:\\Program\\meshlab\\try\\rotatez60.mlx"))






#除非换了文件夹，否则这段代码运行一次就行
# os.system(" wget http://www.patrickmin.com/binvox/linux64/binvox?rnd=1520896952313989 -O binvox")
# os.system("chmod 755 binvox")
# os.system("python -m venv .env")
# os.system("source .env/bin/activate")
# os.system("pip install -r mac_requirements.txt")


for i in ["90","180","270"]:
    # print("python binvox_converter.py modelnet10xz{}/ --remove-all-dupes".format(i))
    os.system("python binvox_converter.py modelnet10xz{}/ --remove-all-dupes".format(i))


# y_test=[]
# for j in [5,10,15,20,25]:
#     path="modelnet10z{}".format(j)
#     for i,dir_path in enumerate(list(os.scandir(path))):
#         path = os.path.join(dir_path.path, 'test')
#         file_paths = [os.path.join(path, i)
#                       for i in os.listdir(path)]
#         for k in file_paths:
#             if k.endswith(".off"):
#                 os.system("rm {}".format(k))

