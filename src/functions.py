import os
from collections import OrderedDict

# from pytorch_metric_learning import losses, miners, distances, reducers, testers
from sklearn.metrics import precision_recall_fscore_support

from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity
# from pytorch_metric_learning.utils import common_functions as c_f

from PIL import Image
from tqdm import tqdm


from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


@lru_cache(maxsize=5)
def get_all_embeddings(dataset, model, epoch):
    print(f'calc emb for epoch {epoch}')
    model.eval()
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def calculate_cosine_sim(train_emb, test_emb):
    calculated_matrix = cosine_similarity(train_emb, test_emb)
    pred_index_of_labels = np.argmax(calculated_matrix, axis=0)
    pred_dist = np.max(calculated_matrix, axis=0)

    return pred_index_of_labels, pred_dist


def calculate_top_n_cosine_sim(train_emb, test_emb, top_n=10):
    calculated_matrix = cosine_similarity(train_emb, test_emb)

    predicted_indexes = []
    predicted_distances = []

    for iteration in range(top_n):
        pred_index_of_labels = np.argmax(calculated_matrix, axis=0)
        pred_dist = np.max(calculated_matrix, axis=0)

        predicted_indexes.append(list(pred_index_of_labels))
        predicted_distances.append([float(distance) for distance in pred_dist])

        calculated_matrix[pred_index_of_labels] = 0

    predicted_distances = np.array(predicted_distances).T.tolist()
    predicted_indexes = np.array(predicted_indexes).T.tolist()

    return predicted_indexes, predicted_distances


@lru_cache(maxsize=5)
def ids_to_class_names(train_dataset, train_labels):
    train_class_names = []
    cls_to_idx_dict = train_dataset.dataset.class_to_idx

    for train_index in tqdm(train_labels, desc='converting labels to class names'):
        for cls_name, idx in cls_to_idx_dict.items():
            if train_index == idx:
                train_class_names.append(cls_name)
                break

    return np.array(train_class_names)


def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def frame_image(img, frame_width, color='green'):
    b = frame_width  # border size in pixel
    ny, nx = img.shape[0], img.shape[1]  # resolution / number of pixels in x and y
    if img.ndim == 3:  # rgb or rgba array
        framed_img = np.ones((b + ny + b, b + nx + b, img.shape[2]))
    if color == 'green':
        framed_img = framed_img * np.array([1, 255, 1])
    elif color == 'red':
        framed_img = framed_img * np.array([255, 1, 1])
    elif color == 'yellow':
        framed_img = framed_img * np.array([255, 255, 1])

    framed_img[b:-b, b:-b] = img
    return framed_img


def save_tensors_by_indexes(query_images, train_dataset, pred_index_of_labels, pred_dist, output_path):
    for query_index in tqdm(range(len(query_images)), total=len(query_images), desc='writing visualizations'):
        curr_tensors_list = []

        curr_pred_index_of_labels = pred_index_of_labels[query_index]
        curr_pred_dist = [f'[{index + 1} place] — ' \
                          f't: {int(train_dataset.get_original_item(int(image_index_in_dataset))["target"])},' \
                          f'\nscore: {round(float(curr_score), 4)}' for index, (curr_score, image_index_in_dataset) in
                          enumerate(zip(pred_dist[query_index], curr_pred_index_of_labels))]

        for current_index in curr_pred_index_of_labels:
            curr_tensors_list.append(
                train_dataset.get_original_item(int(current_index))['input'].permute(1, 2, 0).numpy())

        curr_tensors_list.insert(0, query_images[query_index]['image'])
        curr_pred_dist.insert(0, f'query image, t: {query_images[query_index]["target"]}')

        save_path = os.path.join(output_path, str(query_index))

        save_image_list(list_images=curr_tensors_list,
                        list_titles=curr_pred_dist,
                        num_cols=3,
                        figsize=(15, 11),
                        grid=False,
                        title_fontsize=20,
                        save_path=save_path)


def sort_sum_dist_labels(pred_labels, pred_distances):
    """
        returns labels sorted by sum of dists
        e.g:
            # pred_labels = np.array([0,0,1,1,3,3,4,1])
            # pred_distances = np.array([0.5,0.4,0.9,0.9,0.1,0.1,0.5,0.9])
            # unique_labels == [0, 1, 3, 4]
            # sorted_confs == [0.9, 2.7, 0.2, 0.5] # sum for class 0, sum for class 1, et.c.
            # returns array([1, 0, 4, 3]), array([2.7, 0.9, 0.5, 0.2])

    """
    unique_labels = list(set(pred_labels))
    confs = [sum(np.array(pred_distances)[np.array(pred_labels) == x]) for x in unique_labels]
    sort_ind = np.argsort(confs)[::-1]
    sorted_confs = np.array(confs)[sort_ind]
    sorted_labels = np.array(unique_labels)[sort_ind]
    return sorted_labels, sorted_confs.tolist()


def save_tensors_unique(query_images, train_dataset, pred_index_of_labels, pred_dist, output_path, num_cols=5):
    num_rows = len(query_images)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 4.5 * num_rows))
    fig.tight_layout(h_pad=10)
    fig.subplots_adjust(top=0.95)
    TP = 0
    TP_in_top = 0
    for query_index, q in tqdm(enumerate(query_images), total=len(query_images), desc="Calc metrics & plot"):
        query_image, target_label = q['image'], q['target']
        pred_labels = np.array([
            int(train_dataset.get_original_item(int(x))["target"]) for x in pred_index_of_labels[query_index]
        ])
        pred_distances = pred_dist[query_index]
        pred_labels_unique, pred_distances_unique = sort_sum_dist_labels(pred_labels, pred_distances)

        if target_label == pred_labels_unique[0]:
            TP += 1
            TP_in_top += 1
        elif target_label in pred_labels_unique:
            TP_in_top += 1

        imgs = []
        for l in pred_labels_unique:  # TODO: Map to reference, not train!
            train_img_index = np.array(pred_index_of_labels)[query_index][
                np.argwhere(pred_labels == l)[0]]  # just first match in train
            imgs.append(train_dataset.get_original_item(int(train_img_index))['input'].permute(1, 2, 0).numpy())

        # setup query image with target label class and "query image" string as distance
        imgs.insert(0, query_image)

        pred_distances_unique.insert(0, 1)
        pred_labels_unique = np.insert(pred_labels_unique.astype("object"), 0, target_label)
        plot_query_result(query_index,
                          axes,
                          imgs[:num_cols],
                          pred_distances_unique[:num_cols],
                          pred_labels_unique[:num_cols])

        h = axes[query_index][0].get_position(True).y0 - 0.005
        line = plt.Line2D((0, 1), (h, h), color="k", linewidth=1)
        fig.add_artist(line)

    plot_title = f"Acc: {TP/num_rows:.2f}, Top {num_cols} Acc: {TP_in_top/num_rows:.2f}"
    fig.suptitle(plot_title, fontsize=20)
    save_path = os.path.join(output_path, "resplot.png")
    print(f"{plot_title}, Plot saved to {save_path}")
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.close(fig)


def plot_query_result(query_index, axes, list_images, list_confs, list_classes, title_fontsize=10,  border_width=10):
    list_axes = axes[query_index]
    for i in range(len(list_axes)):
        list_axes[i].axis('off')
        list_axes[i].grid(False)

        try:
            img = list_images[i]
        except IndexError:
            continue

        title = f"ID:{list_classes[i]} CONF:{list_confs[i]:.1f}" if i > 0 else f"Target ID:{list_classes[i]}"
        if i == 1:
            if list_classes[i] == list_classes[0]:
                img = frame_image(img, border_width, color='green')
            else:
                img = frame_image(img, border_width, color='red')
        elif i > 1 and not list_classes[1] == list_classes[0]:
            if list_classes[i] == list_classes[0]:
                img = frame_image(img, border_width, color='yellow')

        list_axes[i].imshow(img.astype(np.uint8))
        list_axes[i].set_title(title, fontsize=title_fontsize)



def save_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
                    title_fontsize=30, save_path=None):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img.astype(np.uint8), cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def calc_metrics(predict, gt):
    f1 = f1_score(gt, predict, average='macro')
    precision = precision_score(gt, predict, average='macro')
    recall = recall_score(gt, predict, average='macro')
    #     roc_auc = roc_auc_score(gt, predict, average='macro')
    accuracy = accuracy_score(gt, predict)

    return f1, precision, recall, accuracy


def preprocess_weights(model_weights):
    return OrderedDict([(k.replace('model.', ''), v) for k, v in model_weights.items() if k != 'model.head.weight'])

