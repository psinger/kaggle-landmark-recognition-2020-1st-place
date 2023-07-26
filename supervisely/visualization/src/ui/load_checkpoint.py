import os
from pathlib import Path

import sly_globals as g
import supervisely as sly
from sly_visualization_progress import get_progress_cb, reset_progress, init_progress, _update_progress_ui

local_weights_path = None
available_checkpoints = {}


def init(data, state):
    data["done2"] = False
    data['Checkpoints'] = []
    init_progress(2, data)

    state["collapsed2"] = True
    state["disabled2"] = True
    state["modelLoading"] = False
    state['pathReady'] = False
    state["weightsPath"] = ""


def restart(data, state):
    data["done2"] = False
    sly.fs.clean_dir(g.checkpoints_dir)


def list_files(sly_fs_path):
    nesting_level = len(sly_fs_path.split('/'))

    files_in_dir = g.api.file.list(g.team_id, sly_fs_path)
    pth_paths = [file['path'] for file in files_in_dir
                 if (file['path'].endswith('.pth') or file['path'].endswith('.ckpt')) and
                 len(file['path'].split('/')) == nesting_level]

    return pth_paths


def get_file_sizes(sly_fs_path):
    size_b = []
    files = g.api.file.list(g.team_id, sly_fs_path)
    for file in files:
        size_b.append(file['meta']['size'])
    return sum(size_b)


def download_checkpoint(sly_fs_checkpoints_path):
    files_size_b = get_file_sizes(sly_fs_checkpoints_path)
    download_progress = get_progress_cb(2, "Download checkpoint", files_size_b, is_size=True, min_report_percent=1)
    local_path = os.path.join(g.checkpoints_dir, Path(sly_fs_checkpoints_path).name)
    if not os.path.exists(local_path):
        g.api.file.download(g.team_id, sly_fs_checkpoints_path, local_path, progress_cb=download_progress)
        # g.api.file.download(g.team_id, sly_fs_checkpoints_path.replace('ckpt', 'py'),
        #                     local_path.replace('ckpt', 'py'))
    reset_progress(2)


def get_embedding_list(state, by_file=True):
    root_path = str(Path(state['weightsPath']).parents[0])
    selected_checkpoint = sly.io.fs.get_file_name(state['selectedCheckpoint'])
    embeddings_folder = os.path.join(root_path, 'embeddings', selected_checkpoint)
    embedding_dict = {}
    for file in g.api.file.list2(g.team_id, embeddings_folder):
        if by_file:
            entity = file.name
            embedding_dict[entity] = file.path
        else:
            entity = Path(file.path).parents[1].name
            if entity not in embedding_dict:
                embedding_dict[entity] = []
            embedding_dict[entity].append(file.path)

    return embedding_dict


@g.my_app.callback("set_checkpoints_path")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def set_checkpoints_path(api: sly.Api, task_id, context, state, app_logger):
    global available_checkpoints
    path_list = list_files(state["weightsPath"])
    for path in path_list:
        available_checkpoints[str(Path(path).name)] = path
    fields = [
        {"field": "state.modelLoading", "payload": False},
        {"field": "data.Checkpoints", "payload": available_checkpoints},
        {"field": "state.pathReady", "payload": True}
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("download_selected_checkpoint")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_selected_checkpoint(api: sly.Api, task_id, context, state, app_logger):
    print(available_checkpoints)
    selected_checkpoint = state['selectedCheckpoint']
    download_checkpoint(available_checkpoints[selected_checkpoint])
    embedding_dict = get_embedding_list(state, by_file=False)
    fields = [
        # {"field": "data.Embeddings", "payload": embedding_dict},
        {"field": "data.Embeddings", "payload": embedding_dict},
        {"field": "data.done2", "payload": True},
        {"field": "state.collapsed3", "payload": False},
        {"field": "state.disabled3", "payload": False},
        {"field": "state.activeStep", "payload": 3},
        {"field": "state.modelLoading", "payload": False},
    ]
    # api.app.set_field(task_id, "data.scrollIntoView", f"step{3}")
    g.api.app.set_fields(g.task_id, fields)
