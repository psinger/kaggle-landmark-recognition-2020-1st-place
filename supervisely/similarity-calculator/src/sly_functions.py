import copy

import torch

import os
import pickle
import tempfile
import time
import urllib.request
import cv2
import requests
import numpy as np
from tqdm import tqdm
from functools import lru_cache

import supervisely_lib as sly

import sly_globals as g
import sly_progress


def init_fields(state, data):
    state['checkpointSelected'] = False
    state['loadingEmbeddingsList'] = False

    state['weightsPath'] = None

    state["modelWeightsOptions"] = "pretrained"
    state["selectedModel"] = "retail [medium]"
    state["device"] = "cuda:0"
    state["weightsPath"] = ""
    state["models"] = [
        {
        "config": "",
        "weightsUrl": "https://github.com/supervisely-ecosystem/gl-metric-learning/releases/download/v0.0.1/r2pk_and_10k.ckpt",
        "Model": "retail [medium]",
        "Classes": "12075"
      },
      {
        "config": "",
        "weightsUrl": "https://github.com/supervisely-ecosystem/gl-metric-learning/releases/download/v0.0.1/r2pk.ckpt",
        "Model": "retail [small]",
        "Classes": "2384"
      },
      {
        "config": "",
        "weightsUrl": "https://github.com/supervisely-ecosystem/gl-metric-learning/releases/download/v0.0.1/snacks_v1.ckpt",
        "Model": "retail [nano]",
        "Classes": "83"
      },
      {
        "config": "",
        "weightsUrl": "https://github.com/supervisely-ecosystem/gl-metric-learning/releases/download/v0.0.1/landmarks.ckpt",
        "Model": "landmarks [medium]",
        "Classes": "10752"
      },
      {
        "config": "",
        "weightsUrl": "https://github.com/supervisely-ecosystem/gl-metric-learning/releases/download/v0.0.1/pictures_v1.ckpt",
        "Model": "pictures [nano]",
        "Classes": "83"
      }
    ]
    state["modelColumns"] = [
        {
            "key": "Model",
            "title": "Model",
            "subtitle": None
        },
        {
            "key": "Classes",
            "title": "Classes",
            "subtitle": None
        }
    ]

    data['embeddingsInfo'] = {}

    state['selectAllEmbeddings'] = False
    state['selectedEmbeddings'] = [f'{g.project_info.name}_{g.project_id}']  # HARDCODE
    # state['selectedEmbeddings'] = []

    state['embeddingsLoaded'] = False
    state['loadingEmbeddings'] = False

    data['embeddingsStats'] = {}


def init_progress_bars(data, state):
    progress_names = ['LoadingEmbeddings']

    for progress_name in progress_names:
        data[f"progress{progress_name}"] = None
        data[f"progress{progress_name}Message"] = "-"
        data[f"progress{progress_name}Current"] = None
        data[f"progress{progress_name}Total"] = None
        data[f"progress{progress_name}Percent"] = None


def filter_paths_by_workspace_id(paths):
    filtered_paths = []
    for current_path in paths:
        try:
            if int(current_path.split('/')[4].split('_')[-1]) == g.workspace_id:
                filtered_paths.append(current_path)

        except Exception as ex:
            g.logger.warn(f'Cannot filter path: {current_path}\n'
                          f'ex: {ex}')
    return filtered_paths


def group_paths_by_project_ids(paths):
    paths_by_projects = {}

    for current_path in paths:
        current_project_name = current_path.split('/')[5]
        added_paths = paths_by_projects.get(current_project_name, [])
        added_paths.append(current_path)
        paths_by_projects[current_project_name] = added_paths

    return paths_by_projects


def download_embeddings(embeddings_paths):
    local_pickle_paths = []

    sly_progress_embeddings = sly_progress.SlyProgress(g.api, g.task_id, 'progressLoadingEmbeddings')
    sly_progress_embeddings.refresh_params('Downloading embeddings', len(embeddings_paths))

    for embedding_path in embeddings_paths:
        full_path = embedding_path
        relative_path = '/GL-MetricLearning/embeddings/'
        local_embedding_path = os.path.join(g.local_embeddings_dir, os.path.relpath(full_path, relative_path))

        if os.path.exists(local_embedding_path):
            os.remove(local_embedding_path)

        g.api.file.download(g.team_id, embedding_path, local_embedding_path)
        local_pickle_paths.append(local_embedding_path)

        sly_progress_embeddings.next_step()
    return local_pickle_paths


def load_embeddings_to_memory(pickles_files):
    sly_progress_pickles = sly_progress.SlyProgress(g.api, g.task_id, 'progressLoadingEmbeddings')
    sly_progress_pickles.refresh_params('Loading embeddings to memory', len(pickles_files))

    embeddings_data = {
        'label': []
    }

    placeholders_data = {
        'label': []
    }

    for pickle_file_path in pickles_files:
        with open(pickle_file_path, 'rb') as pickle_file:
            current_data = pickle.load(pickle_file)

            for current_label, current_embeddings_data in current_data.items():
                for current_embedding_data in current_embeddings_data:
                    if current_embedding_data['embedding'] is None:
                        current_storage = placeholders_data
                    else:
                        current_storage = embeddings_data

                    if len(current_storage) == 1:
                        for new_key in current_embedding_data.keys():
                            embeddings_data[new_key] = []
                            placeholders_data[new_key] = []

                    for key, value in current_embedding_data.items():
                        current_storage[key].append(value)

                    current_storage['label'].append(current_label)

        sly_progress_pickles.next_step()
    sly_progress_pickles.reset_params()
    return embeddings_data, placeholders_data


def cos_similarity_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_topn_cossim(test_emb, tr_emb, batchsize=64, n=10, device='cuda:0', verbose=True):
    tr_emb = torch.tensor(tr_emb, dtype=torch.float32, device=torch.device(device))
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=torch.device(device))
    vals = []
    inds = []

    n = tr_emb.shape[0]
    for test_batch in test_emb.split(batchsize):
        sim_mat = cos_similarity_matrix(test_batch, tr_emb)
        vals_batch, inds_batch = torch.topk(sim_mat, k=n, dim=1)
        vals += [vals_batch.detach().cpu()]
        inds += [inds_batch.detach().cpu()]
    vals = torch.cat(vals)
    inds = torch.cat(inds)
    return vals, inds


def add_indexes_to_database(embeddings_data_in_list):
    for index, row in enumerate(embeddings_data_in_list):
        row.update({'index': index})


def prepare_database(embeddings_in_memory):
    labels_list = embeddings_in_memory['label']
    items_database = {current_label: {} for current_label in labels_list}

    for index, current_url in enumerate(embeddings_in_memory['url']):
        current_label = labels_list[index]
        urls_in_memory = items_database[current_label].get('url', [])
        urls_in_memory.append(current_url)
        items_database[current_label]['url'] = urls_in_memory

    for index, current_description in enumerate(embeddings_in_memory['description']):
        current_label = labels_list[index]
        description_in_memory = items_database[current_label].get('description', None)
        if description_in_memory is None:
            items_database[current_label].update(current_description)

    return items_database


def get_custom_project_id(project_name):
    projects_list = g.api.project.get_list(g.workspace_id)
    for current_project in projects_list:
        if current_project.name == project_name:
            return current_project.id
    return None


def get_custom_dataset_id(project_id, items_count):
    datasets_list = g.api.dataset.get_list(project_id)

    if items_count > g.custom_dataset_images_max_count or len(datasets_list) == 0:
        return None

    if datasets_list[-1].images_count and ('custom_data' in str(datasets_list[-1].name)):
        if datasets_list[-1].images_count + items_count < g.custom_dataset_images_max_count:
            return datasets_list[-1].id

    return None


def update_class_list(project_id):
    project_meta = g.api.project.get_meta(project_id)
    if g.custom_label_title not in [class_info['title'] for class_info in project_meta['classes']]:
        objects_classes = sly.ObjClassCollection([sly.ObjClass(g.custom_label_title, sly.Rectangle)])
        meta = sly.ProjectMeta(obj_classes=objects_classes)
        meta = meta.merge(sly.ProjectMeta.from_json(project_meta))
        g.api.project.update_meta(project_id, meta.to_json())


def init_project_remotely(project_id=None, items_count=0, project_name='custom_reference_data'):

    if not project_id:
        project_id = get_custom_project_id(project_name)
        project = g.api.project.create(g.workspace_id, project_name, type=sly.ProjectType.IMAGES,
                                       change_name_if_conflict=True)
    else:
        project = g.api.project.get_info_by_id(project_id)

    update_class_list(project.id)
    ds_id = get_custom_dataset_id(project.id, items_count)

    if not ds_id:
        ds_name = 'custom_data'
        dataset = g.api.dataset.create(project.id, f'{ds_name}',
                                       change_name_if_conflict=True)
    else:
        dataset = None
        for dataset in g.api.dataset.get_list(project_id):
            if dataset.name == ds_id:
                dataset = dataset
                break

    return project, dataset


@lru_cache(maxsize=32)
def url_to_image(url):
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # return the image
        return image


def cache_images(data):
    """
     FOR EACH url in data
     download image to RAM by url
    """
    for row in tqdm(data, desc='⏬ downloading images'):
        try:
            image_in_memory = url_to_image(row['url'])
            row['cached_image'] = image_in_memory
        except Exception as ex:
            g.logger.warn(f'image not downloaded: {row["url"]}\n'
                          f'reason: {ex}')

            row['cached_image'] = None


def crop_images(data):
    """
     FOR EACH image in data
     crop image if bbox is not None
    """
    for row in data:
        if row['cached_image'] is not None and row['bbox']:
            top, left, height, width = row['bbox'][0], row['bbox'][1], row['bbox'][2], row['bbox'][3]
            crop = row['cached_image'][top:top + height, left:left + width]

            if crop.shape[0] > 0 and crop.shape[1] > 0:
                row['cached_image'] = crop
            else:
                g.logger.warn(f'image not cropped: {row["url"]}\n'
                              f'reason: {crop.shape}')
                row['cached_image'] = None


def get_data_to_upload(sly_dataset, data_to_process):
    img_names = []
    images_nps = []

    if sly_dataset.images_count:
        start_image_number = sly_dataset.images_count + 1
    else:
        start_image_number = 1
    for index, row in enumerate(data_to_process):
        img_names.append(f"{start_image_number + index:04d}_{time.time_ns()}.jpg")
        images_nps.append(row['cached_image'])

    return img_names, images_nps


def get_project_tags_names(project_meta):
    tags_names = []
    for curr_tag in project_meta.tag_metas:
        if curr_tag.name:
            tags_names.append(curr_tag.name)
    return tags_names


def generate_annotations(project_id, data_to_process, new_images_info):
    project_meta = g.api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta)
    project_tags_names = get_project_tags_names(project_meta)

    annotations = []
    for row, image_info in zip(data_to_process, new_images_info):
        tag_meta = sly.TagMeta(row['label'], sly.TagValueType.NONE)

        if row['label'] not in project_tags_names:
            project_meta = project_meta.add_tag_meta(tag_meta)
            project_tags_names.append(row['label'])

        tag_collection = sly.TagCollection([sly.Tag(tag_meta)])

        label = sly.Label(sly.Rectangle(top=0, left=0, bottom=image_info.height - 1, right=image_info.width - 1),
                          sly.ObjClass(g.custom_label_title, sly.Rectangle), tag_collection)
        annotations.append(sly.Annotation((image_info.height, image_info.width), [label]))

    g.api.project.update_meta(project_id, project_meta.to_json())
    return annotations


def add_images_to_project(data_to_process):
    data_to_process = [dict(zip(data_to_process, t)) for t in zip(*data_to_process.values())]

    sly_project, sly_dataset = init_project_remotely(project_id=g.project_id, items_count=len(data_to_process))

    cache_images(data_to_process)
    crop_images(data_to_process)

    images_names, images_nps = get_data_to_upload(sly_dataset, data_to_process)

    new_image_infos = g.api.image.upload_nps(sly_dataset.id, images_names, images_nps, metas=None, progress_cb=None)

    annotations = generate_annotations(sly_project.id, data_to_process, new_image_infos)

    g.api.annotation.upload_anns(
        img_ids=[current_image.id for current_image in new_image_infos],
        anns=annotations
    )

    new_images_urls = [current_image.path_original for current_image in new_image_infos]
    new_bboxes = [[0, 0, current_image.height - 1, current_image.width - 1] for current_image in new_image_infos]

    return new_images_urls, new_bboxes
