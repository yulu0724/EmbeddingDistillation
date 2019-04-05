import os
import shutil

root = 'CUB_200_2011'
image_folder = 'CUB_200_2011/images'
dirs = os.listdir(image_folder)

sets = ['train', 'test']

for x in sets:
    if not os.path.exists(os.path.join(root, x)):
        os.mkdir(os.path.join(root, x))

paths = [os.path.join(root, x) for x in sets]


for img in dirs:
    index = (int(img.split('.')[0]))
    img_path = os.path.join(image_folder, img)
    if index < 101:
        print(index)
        shutil.move(img_path, paths[0])
    else:
        shutil.move(img_path, paths[1])
