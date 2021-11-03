
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

import sly_globals as g
import sly_progress


def init_fields(state, data):
    state['checkpointSelected'] = False
    state['loadingEmbeddingsList'] = False

    state['weightsPath'] = None

    state["modelWeightsOptions"] = "pretrained"
    state["selectedModel"] = "landmarks"
    state["device"] = "cuda:0"
    state["weightsPath"] = ""
    state["models"] = [
        {
            "config": "",
            "weightsUrl": "https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/44467/T/5/Y7/3kHqfajvBP9Ry8TSc2LvNVjmZpaAtOlzCyqwwfb0TmxQfrjtrIDQz09Occgetu3OlT6QwHYm0DvwbNMXlCOaBDapXuYtPPbXyNAccvm342HUy8yCVlKmAKaNr8F7.bin",
            "Model": "landmarks",
            "Classes": "10752"
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "pictures v1",
            "Classes": "83"
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "snacks v1",
            "Classes": "83"
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "rp2k",
            "Classes": "2384"
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "rp2k and 10k",
            "Classes": "12075"
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

    state['selectAllEmbeddings'] = True
    state['selectedEmbeddings'] = []

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

    for pickle_file_path in pickles_files:
        with open(pickle_file_path, 'rb') as pickle_file:
            current_data = pickle.load(pickle_file)

            for current_label, current_embeddings_data in current_data.items():
                for current_embedding_data in current_embeddings_data:

                    if len(embeddings_data) == 1:
                        for new_key in current_embedding_data.keys():
                            embeddings_data[new_key] = []

                    for key, value in current_embedding_data.items():
                        embeddings_data[key].append(value)

                    embeddings_data['label'].append(current_label)

        sly_progress_pickles.next_step()
    sly_progress_pickles.reset_params()
    return embeddings_data


def cos_similarity_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_topk_cossim(test_emb, tr_emb, batchsize=64, k=10, device='cuda:0', verbose=True):
    tr_emb = torch.tensor(tr_emb, dtype=torch.float32, device=torch.device(device))
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=torch.device(device))
    vals = []
    inds = []
    for test_batch in test_emb.split(batchsize):
        sim_mat = cos_similarity_matrix(test_batch, tr_emb)
        vals_batch, inds_batch = torch.topk(sim_mat, k=k, dim=1)
        vals += [vals_batch.detach().cpu()]
        inds += [inds_batch.detach().cpu()]
    vals = torch.cat(vals)
    inds = torch.cat(inds)
    return vals, inds

