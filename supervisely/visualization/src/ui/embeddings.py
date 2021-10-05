import supervisely_lib as sly
import sly_globals as g
from pathlib import Path
import os
# from functools import partial
from sly_visualization_progress import get_progress_cb, reset_progress, init_progress

# TEAMFILES
#     MetricLearningData
#         checkpoints
#             ckpt1.pth
#             ckpt2.pth
#         embeddings
#             ckpt1
#                 workspace1
#                     project_name
#                         ds1.pkl
#                         ds1.pkl
#                 workspace2
#                     project_name
#                         ds1.pkl
#                         ds1.pkl
#             ckpt2
#                 workspace1
#                     project_name
#                         ds1.pkl
#                         ds1.pkl
#                 workspace2
#                     project_name
#                         ds1.pkl
#                         ds1.pkl


def init(data, state):
    data["modelsTable"] = []
    data["done3"] = False
    data['Embeddings'] = []
    state['selectedEmbeddings'] = []
    state['selectedClass'] = None
    state["statsLoaded"] = False
    state["loadingStats"] = False
    state["selectedModels"] = ['model_last.pth']

    state["collapsed3"] = True
    state["disabled3"] = True


def restart(data, state):
    data['done3'] = False


def get_file_sizes(sly_fs_path):
    size_b = []
    files = g.api.file.list(g.team_id, sly_fs_path)
    for file in files:
        size_b.append(file['meta']['size'])
    return sum(size_b)


def download_file(sly_fs_path):
    files_size_b = get_file_sizes(sly_fs_path)
    # download_progress = get_progress_cb(2, "Download file", files_size_b, is_size=True, min_report_percent=1)
    local_path = os.path.join(g.embeddings_dir, Path(sly_fs_path).name)
    if not os.path.exists(local_path):
        g.api.file.download(g.team_id, sly_fs_path, local_path,
                            # progress_cb=download_progress
                            )
    reset_progress(2)


@g.my_app.callback("download_selected_embeddings")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_selected_embeddings(api: sly.api, task_id, context, state, app_logger):
    selected_count = len(state['selectedEmbeddings'])

    if selected_count == 0:
        raise ValueError('No embedding files selected. Please select files.')

    for file in state['selectedEmbeddings']:
        download_file(file)

    fields = [
        {"field": "data.done3", "payload": True},
        {"field": "state.collapsed4", "payload": False},
        {"field": "state.disabled4", "payload": False},
        {"field": "state.activeStep", "payload": 4},

    ]
    api.task.set_fields(task_id, fields)
