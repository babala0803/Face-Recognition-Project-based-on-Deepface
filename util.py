import os
from deepface import DeepFace
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_cls():
    cls = []
    for root, dirs, files in os.walk('./vggface'):
        for d in dirs:
            cls.append(d)
    return cls


def get_img_paths(cls, type='test'):
    train_image_paths = []
    test_image_paths = []
    train_labels = []
    test_labels = []
    image_paths = []
    image_labels = []

    if type == 'veri_test':
        return image_paths, image_labels
    return train_image_paths, test_image_paths, train_labels, test_labels


def ground_true_list():
    folder_path = "testset1"
    file_list = os.listdir(folder_path)  # 获取文件夹中所有文件的列表

    name_list = []
    for file_name in file_list:
        base_name = os.path.splitext(file_name)[0]  # 获取文件名的部分，不包括扩展名
        name_parts = base_name.split("_")  # 使用"_"分割文件名的部分
        name = name_parts[0]  # 获取第一部分作为名称
        name_list.append(name)

    print("ground_true_list:")
    print(name_list)
    return name_list


def get_all_path():
    folder_path = "testset1"
    file_list = os.listdir(folder_path)  # 获取文件夹中所有文件的列表

    relative_paths = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)  # 获取文件的完整路径
        relative_path = os.path.join(".", file_path)  # 添加 "./" 前缀作为相对路径
        relative_paths.append(relative_path)

    return relative_paths


def get_predict(model='VGG-Face',distance_metric='cosine'):
    db_path = "./testset2/"  # 数据库路径
    relative_path = get_all_path()
    predict_list = []
    for path in relative_path:
        result = DeepFace.find(img_path=path, db_path=db_path, enforce_detection=False,model_name=model
                               ,distance_metric=distance_metric)
        if len(result) != 0 and not result[0].empty:
            name = result[0].iloc[0, 0]
            file_name = os.path.basename(name)  # 获取文件名
            name = file_name.split("_")[0]  # 使用下划线分割文件名并获取第一部分
        else:
            name = "n000082"
        # print("name:")
        # print(name)
        predict_list.append(name)
    return predict_list


def get_y_sim():
    db_path = "./testset2/"  # 数据库路径
    relative_path = get_all_path()
    real_name_list = ground_true_list()
    y_true = []
    y_score = []
    for index, path in enumerate(relative_path):
        real_name = real_name_list[index]

        result = DeepFace.find_all(img_path=path, db_path=db_path, enforce_detection=False)
        for row in result[0].itertuples(index=False):
            score = row[5]
            file_name = row[0].split("/")[-1].split("_")[0]
            if real_name == file_name:
                y_true.append(1)
            else:
                y_true.append(0)
            y_score.append(1-score)
    return y_true,y_score

def show_results(name, test_labels,
                 categories, abbr_categories, predicted_categories):
    """
    shows the results
    :param train_image_paths:
    :param test_image_paths:
    :param train_labels:
    :param test_labels:
    :param categories:
    :param abbr_categories:
    :param predicted_categories:
    :return:
    """
    from sklearn.metrics import confusion_matrix
    print(categories)
    cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    # confusion matrix
    y_true = [cat2idx[cat] for cat in test_labels]
    y_pred = [cat2idx[cat] for cat in predicted_categories]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(np.float64) / cm.sum(axis=1)[:, np.newaxis]
    acc = np.mean(np.diag(cm))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('jet'))
    plt.title(name+' Confusion matrix. Mean of diagonal = {:4.2f}%'.format(acc * 100))
    tick_marks = np.arange(len(categories))
    plt.tight_layout()
    plt.xticks(tick_marks, abbr_categories, rotation=45)
    plt.yticks(tick_marks, categories)


if __name__ == '__main__':
    # df = DeepFace.find_all(img_path='./testset1/n000009_1.jpg', db_path='./testset2/', enforce_detection=False)
    # print(df[0].iloc[0, 5])
    # print(df[0].iloc[0, 0])
    # file_name = df[0].iloc[0, 0].split("/")[-1].split("_")[0]
    #
    # print(file_name)  # 输出：n000009

    # for row in df[0].itertuples(index=False):
    #     print(row[0])
    #     print(row[5])
    check=get_cls()
    print(check)
