import os

import supervisely_lib as sly

import json
import numpy as np

import sly_globals as g
import sly_functions as f


@g.my_app.callback("get_info")
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
            {"field": f"state.loadingEmbeddings", "payload": False}
        ]
        g.api.task.set_fields(g.task_id, fields)
        raise ValueError('No embedding files found. Please reselect embeddings.')

    local_pickles_paths = f.download_embeddings(embeddings_paths)
    g.embeddings_in_memory = f.load_embeddings_to_memory(local_pickles_paths)

    g.embeddings_stats = {
        'Embeddings Count': len(g.embeddings_in_memory['embedding']),
        'Labels Num': len(set(g.embeddings_in_memory['label']))
    }

    fields = [
        {"field": f"state.embeddingsLoaded", "payload": True},
        {"field": f"state.loadingEmbeddings", "payload": False},
        {"field": f"data.embeddingsStats", "payload": g.embeddings_stats}
    ]
    g.api.task.set_fields(g.task_id, fields)


@g.my_app.callback("clear_fields")
def clear_fields(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": f"state.selectAllEmbeddings", "payload": True},
        {"field": f"state.selectedEmbeddings", "payload": []},
        {"field": f"state.embeddingsLoaded", "payload": False},
    ]
    g.api.task.set_fields(g.task_id, fields)


@g.my_app.callback("select_checkpoint")
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
        {"field": f"state.checkpointSelected", "payload": True},
        {"field": f"state.loadingEmbeddingsList", "payload": False},
        {"field": f"data.embeddingsInfo", "payload": paths_by_projects},
    ]

    g.api.task.set_fields(g.task_id, fields)


@g.my_app.callback("calculate_similarity")
@sly.timeit
def calculate_similarity(api: sly.Api, task_id, context, state, app_logger):
    g.logger.info(f'calculating similarity for batch; {context["request_id"]}')

    data_to_process = dict(state['input_data'])

    top_k = data_to_process.get('top_k', 5)
    input_embeddings = data_to_process['embeddings']

    input_embeddings = np.asarray(input_embeddings).astype(np.float32)
    embeddings_in_memory = np.asarray(g.embeddings_in_memory["embedding"]).astype(np.float32)

    pred_dist, pred_index_of_labels = f.get_topk_cossim(input_embeddings,
                                                        embeddings_in_memory,
                                                        k=top_k,
                                                        device='cpu')

    indexes_to_labels = np.asarray(g.embeddings_in_memory['label'])
    indexes_to_urls = np.asarray(g.embeddings_in_memory['url'])

    pred_dist = [list(curr_row) for curr_row in list(pred_dist.data.cpu().numpy())]
    pred_labels = [list(indexes_to_labels[list(curr_row)]) for curr_row in
                   list(pred_index_of_labels.data.cpu().numpy())]
    pred_urls = [list(indexes_to_urls[list(curr_row)]) for curr_row in
                 list(pred_index_of_labels.data.cpu().numpy())]

    output_data = json.dumps(str({'pred_dist': pred_dist,
                                  'pred_labels': pred_labels,
                                  'pred_urls': pred_urls}))

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=output_data)

    g.logger.info(f'successfully calculated! {context["request_id"]}')


@g.my_app.callback("add_new_embeddings_to_reference")
@sly.timeit
def add_new_embeddings_to_reference(api: sly.Api, task_id, context, state, app_logger):
    g.logger.info(f'adding embeddings to reference; {context["request_id"]}')

    data_to_process = dict(state['input_data'])

    # {
    #     'embedding' :[],
    #     'label' :[],
    #     'url' :[],
    #     'bbox' :[]
    # }

    for key, value in data_to_process.items():
        g.embeddings_in_memory[key].extend(value)

    g.embeddings_stats = {
        'Embeddings Count': len(g.embeddings_in_memory['embedding']),
        'Labels Num': len(set(g.embeddings_in_memory['label']))
    }

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.embeddings_stats)

    g.logger.info(f'successfully added! {context["request_id"]}')


# @TODO: add_new_embeddings_to_reference store to sly_dataset
# @TODO: add_new_embeddings_to_reference store to pickles

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
