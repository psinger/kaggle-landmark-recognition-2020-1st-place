import supervisely as sly
import numpy as np
import sly_globals as g
import torch
import os
from collections import OrderedDict
from sly_visualization_progress import init_progress
from model_functions import initialize_network, load_weights, calculate_embeddings_for_nps_batch
from supervisely.calculator import main
from pathlib import Path
import pickle

from urllib.parse import urlparse



local_weights_path = None


def init(data, state):
    state["collapsed4"] = True
    state["disabled4"] = True
    state["modelLoading"] = False
    state['inferProject'] = False
    init_progress(4, data)

    state["weightsPath"] = ""
    data["done4"] = False


def restart(data, state):
    data["done4"] = False


def cos_similarity_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_resized_image(image_storage_url, height):
    parsed_link = urlparse(image_storage_url)

    return f'{parsed_link.scheme}://{parsed_link.netloc}' \
           f'/previews/q/ext:jpeg/resize:fill:0:{height}:0/q:0/plain{parsed_link.path}'


def get_topk_cossim(test_emb, tr_emb, batchsize = 64, k=10, device='cuda:0',verbose=True):
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


def infer_project_(state, context):
    initialize_network()
    local_path = os.path.join(g.checkpoints_dir, state['selectedCheckpoint'])
    load_weights(local_path)

    workspace = g.api.workspace.get_info_by_id(g.workspace_id)
    project = g.api.project.get_info_by_id(g.project_id)
    datasets = g.api.dataset.get_list(g.project_id)
    datasets_list = []

    remote_weights_path = state['weightsPath']
    remote_embeddings_dir = state['weightsPath'].replace('checkpoints', 'embeddings')

    for dataset in datasets:
        embedding_path = os.path.join(
            remote_embeddings_dir, sly.fs.get_file_name(remote_weights_path),
            workspace.name, project.name, dataset.name + '.pkl')
        datasets_list.append([workspace, project, dataset, embedding_path])

    predicted_embedding_list = []
    predicted_url_list = []
    predicted_gt_list = []
    predicted_parent_img_ids_list = []
    for ds in datasets_list:
        embeddings, gt, parent_img_ids = main.process_dataset(ds)  # HERE IS GT!
        urls = main.get_img_urls(parent_img_ids)
        predicted_embedding_list.extend(embeddings)
        predicted_url_list.extend(urls)
        predicted_gt_list.extend(gt)
        predicted_parent_img_ids_list.extend(parent_img_ids)

    precalculated_embedding_data = {}
    precalculated_embedding_list = []
    precalculated_url_list = []
    precalculated_labels_list = []
    for embedding_file in state['selectedEmbeddings']:
        if isinstance(embedding_file, list):
            for element in embedding_file:
                local_path = os.path.join(g.embeddings_dir, Path(element).name)
                try:
                    with open(local_path, 'rb') as pkl_file:
                        file_content = pickle.load(pkl_file)

                    for k, v in file_content.items():
                        precalculated_embedding_data[k] = v
                        precalculated_labels_list.extend([k for i in range(v.__len__())])
                        precalculated_url_list.extend(list(v.keys()))
                        precalculated_embedding_list.extend(list(v.values()))
                except:
                    pass
        else:
            local_path = os.path.join(g.embeddings_dir, Path(embedding_file).name)
            try:
                with open(local_path, 'rb') as pkl_file:
                    file_content = pickle.load(pkl_file)

                for k, v in file_content.items():
                    precalculated_embedding_data[k] = v
                    precalculated_labels_list.append(k)
                    precalculated_url_list.extend(list(v.keys()))
                    precalculated_embedding_list.extend(list(v.values()))
            except:
                pass

    pred_dist, \
    pred_index_of_labels = get_topk_cossim(
        predicted_embedding_list, precalculated_embedding_list, k=10, device=g.device)

    filtered_urls = []
    filtered_labels = []
    # {'url': None, 'label': None, 'conf': None, color: 'blue'}
    for line_idx, line in enumerate(pred_index_of_labels):
        query_image = get_resized_image(predicted_url_list[line_idx], height=250)
        line_urls_encoder = [query_image]
        line_labels_encoder = [predicted_gt_list[line_idx]]
        for col in line:
            top_n = get_resized_image(precalculated_url_list[col], height=250)
            line_urls_encoder.append(top_n)
            line_labels_encoder.append(precalculated_labels_list[col])
        filtered_labels.append(line_labels_encoder)
        filtered_urls.append(line_urls_encoder)

    g.gallery_data = {
        "labels": filtered_labels,
        "urls": filtered_urls,
        "confidences": pred_dist
    }


@g.my_app.callback("infer_project")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def infer_project(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.inferProject", "payload": True}
    ]
    g.api.app.set_fields(g.task_id, fields)
    infer_project_(state, context)
    fields = [
        {"field": "data.done4", "payload": True},
        {"field": "state.modelLoading", "payload": False},
        {"field": "state.inferProject", "payload": False},
        {"field": "state.collapsed5", "payload": False},
        {"field": "state.disabled5", "payload": False},
        {"field": "state.activeStep", "payload": 5}
    ]
    g.api.app.set_fields(g.task_id, fields)
