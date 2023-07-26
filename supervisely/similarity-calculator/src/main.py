import functools
import itertools
import os

import supervisely as sly

import json
import numpy as np

import sly_globals as g
import sly_functions as f


def warn_on_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            sly.logger.warn(f'{e}', exc_info=True)
        return value

    return wrapper


@g.my_app.callback("get_info")
@warn_on_exception
@sly.timeit
def get_info(api: sly.Api, task_id, context, state, app_logger):
    if g.selected_weights_type == 'pretrained':
        output_data = {'weightsType': g.selected_weights_type,
                       'Model': g.selected_model}

    else:
        output_data = {'weightsType': g.selected_weights_type}

    output_data.update(g.embeddings_stats)

    output_data = json.dumps(str(output_data))

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=output_data)


@g.my_app.callback("load_embeddings_to_memory")
@warn_on_exception
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def load_embeddings_to_memory(api: sly.Api, task_id, context, state, app_logger):
    try:
        if state['selectAllEmbeddings']:
            embeddings_info = g.api.app.get_field(g.task_id, 'data.embeddingsInfo')
            embeddings_paths = np.hstack(list(embeddings_info.values()))
        else:
            embeddings_info = state['selectedEmbeddings']
            embeddings_paths = np.hstack(list(embeddings_info))
    except Exception as ex:
        fields = [
            {"field": "state.loadingEmbeddings", "payload": False}
        ]
        g.api.task.set_fields(g.task_id, fields)
        raise ValueError('No embedding files found. Please reselect embeddings.')

    local_pickles_paths = f.download_embeddings(embeddings_paths)

    g.embeddings_in_memory, g.placeholders_in_memory = f.load_embeddings_to_memory(local_pickles_paths)

    g.embeddings_stats = {
        'Embeddings Count': len(g.embeddings_in_memory['embedding']),
        'Labels Num': len(set(g.embeddings_in_memory['label']).union(g.placeholders_in_memory['label']))
    }

    fields = [
        {"field": "state.embeddingsLoaded", "payload": True},
        {"field": "state.loadingEmbeddings", "payload": False},
        {"field": "data.embeddingsStats", "payload": g.embeddings_stats}
    ]
    g.api.task.set_fields(g.task_id, fields)


@g.my_app.callback("clear_fields")
@warn_on_exception
def clear_fields(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": f"state.selectAllEmbeddings", "payload": True},
        {"field": f"state.selectedEmbeddings", "payload": [f'{g.project_info.name}_{g.project_id}']},
        {"field": f"state.embeddingsLoaded", "payload": False},

    ]
    g.api.task.set_fields(g.task_id, fields)


@g.my_app.callback("select_checkpoint")
@warn_on_exception
@sly.timeit
def select_checkpoint(api: sly.Api, task_id, context, state, app_logger):
    g.selected_weights_type = state['modelWeightsOptions']
    g.selected_model = state['selectedModel']
    if g.selected_weights_type == 'pretrained':
        listed_path = os.path.join(g.remote_embeddings_dir, g.selected_model)
        projects_list_for_checkpoint = g.api.file.list(g.team_id, listed_path)
        files_paths = [current_checkpoint['path'] for current_checkpoint in projects_list_for_checkpoint]

        filtered_paths = f.filter_paths_by_workspace_id(files_paths)
        pickles_files_paths = [current_path for current_path in filtered_paths if current_path.endswith('.pkl')]
        paths_by_projects = f.group_paths_by_project_ids(pickles_files_paths)

    else:
        paths_by_projects = None
        pass

    fields = [
        {"field": "state.selectedEmbeddings",
         "payload": paths_by_projects.get(f'{g.project_info.name}_{g.project_id}', [])},  # HARDCODE
        {"field": f"state.checkpointSelected", "payload": True},
        {"field": f"state.loadingEmbeddingsList", "payload": False},
        {"field": f"data.embeddingsInfo", "payload": paths_by_projects},
    ]

    g.api.task.set_fields(g.task_id, fields)


def get_indexes_of_labels(labels, indexes_list):
    founded_indexes = []
    for label in labels:
        founded_indexes.append(np.where(np.isin(indexes_list, label)))
    return founded_indexes


def get_topk_predictions(pred_dist, pred_index_of_labels, k):
    topk_pred_dist, topk_pred_index_of_labels = [], []

    for dist_vector, index_vector in zip(pred_dist, pred_index_of_labels):
        dict_of_predictions = {}

        unique_labels = list(set(index_vector))
        indexes_of_unique_labels = get_indexes_of_labels(unique_labels, index_vector)

        for unique_label, curr_indexes_of_label in zip(unique_labels, indexes_of_unique_labels):
            dists_for_label = dist_vector[curr_indexes_of_label]
            dict_of_predictions[unique_label] = sum(dists_for_label) / len(dists_for_label)

        sorted_dict_of_predictions = {k: v for k, v in sorted(dict_of_predictions.items(),
                                                              key=lambda item: item[1], reverse=True)}

        sorted_dict_of_predictions = dict(itertools.islice(sorted_dict_of_predictions.items(), k))

        topk_pred_dist.append(list(sorted_dict_of_predictions.values()))
        topk_pred_index_of_labels.append(list(sorted_dict_of_predictions.keys()))

    return topk_pred_dist, topk_pred_index_of_labels


@g.my_app.callback("calculate_similarity")
@warn_on_exception
@sly.timeit
def calculate_similarity(api: sly.Api, task_id, context, state, app_logger):
    g.logger.info(f'calculating similarity for batch; {context["request_id"]}')

    data_to_process = dict(state['input_data'])

    top_k = data_to_process.get('top_k', 5)
    input_embeddings = data_to_process['embeddings']

    input_embeddings = np.asarray(input_embeddings).astype(np.float32)
    embeddings_in_memory = np.asarray(g.embeddings_in_memory["embedding"]).astype(np.float32)

    pred_dist, pred_index_of_labels = f.get_topn_cossim(input_embeddings,
                                                        embeddings_in_memory,
                                                        n=100,
                                                        device='cpu')
    pred_dist = [curr_row for curr_row in list(pred_dist.data.cpu().numpy())]
    pred_index_of_labels = [list(curr_row) for curr_row in list(pred_index_of_labels.data.cpu().numpy())]

    indexes_to_labels = np.asarray(g.embeddings_in_memory['label'])

    pred_labels = [list(indexes_to_labels[list(curr_row)]) for curr_row in list(pred_index_of_labels)]

    pred_dist, pred_labels = get_topk_predictions(pred_dist, pred_labels, k=top_k)

    output_data = json.dumps(str({'pred_dist': pred_dist,
                                  'pred_labels': pred_labels}))

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=output_data)

    g.logger.info(f'successfully calculated! {context["request_id"]}')


@g.my_app.callback("add_new_embeddings_to_reference")
@warn_on_exception
@sly.timeit
def add_new_embeddings_to_reference(api: sly.Api, task_id, context, state, app_logger):
    g.logger.info(f'adding embeddings to reference; {context["request_id"]}')

    data_to_process = dict(state['input_data'])

    # {
    #     'embedding' :[],
    #     'label' :[],
    #     'url' :[],
    #     'bbox' :[],
    #     'figure_id': []
    # }

    images_urls, bboxes = f.add_images_to_project(data_to_process)

    data_to_process['url'] = images_urls
    data_to_process['bbox'] = bboxes

    for key, value in data_to_process.items():
        if key != 'figure_id':
            g.embeddings_in_memory[key].extend(value)
        else:
            g.custom_dataset_figures_in_reference.extend(value)

    g.embeddings_stats = {
        'Embeddings Count': len(g.embeddings_in_memory['embedding']),
        'Labels Num': len(set(g.embeddings_in_memory['label']).union(g.placeholders_in_memory['label']))
    }

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={
        'embeddings_stats': g.embeddings_stats,
        'new_images_url': images_urls
    })

    g.logger.info(f'successfully added! {context["request_id"]}')

    fields = [
        {"field": f"data.embeddingsStats", "payload": g.embeddings_stats}
    ]
    g.api.task.set_fields(g.task_id, fields)


@g.my_app.callback("get_objects_database")
@warn_on_exception
@sly.timeit
def get_objects_database(api: sly.Api, task_id, context, state, app_logger):
    database_in_dict_format = f.prepare_database(g.placeholders_in_memory)
    database_in_dict_format.update(f.prepare_database(g.embeddings_in_memory))
    request_id = context["request_id"]

    g.my_app.send_response(request_id, data={
        'database': database_in_dict_format,
        'figure_id': g.custom_dataset_figures_in_reference
    })

    g.logger.info(f'successfully added! {context["request_id"]}')


def main():
    sly.logger.info("Similarity calculator started")
    data = {}
    state = {}
    data["taskId"] = g.task_id

    f.init_fields(state=state, data=data)
    f.init_progress_bars(state=state, data=data)

    g.my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
