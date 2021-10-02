
import supervisely_lib as sly
import sly_globals as g
import os
import tqdm
import numpy as np
from collections import namedtuple

def download_model_and_config():
    remote_model_dir, remote_model_weights_name = os.path.split(g.remote_weights_path)

    remote_model_config_name = sly.fs.get_file_name(g.remote_weights_path) + '.py'
    remote_config_file = os.path.join(remote_model_dir, remote_model_config_name)

    g.local_weights_path = os.path.join(g.my_app.data_dir, remote_model_weights_name)
    g.local_config_path = os.path.join(g.my_app.data_dir, remote_model_config_name)

    g.api.file.download(g.team_id, g.remote_weights_path, g.local_weights_path)
    g.api.file.download(g.team_id, remote_config_file, g.local_config_path)



def list_related_datasets():
    workspaces = [g.api.workspace.get_info_by_id(int(g.workspace_id))] if g.only_current_workspace else g.api.workspace.get_list(g.team_id)
    datasets = []
    for ws in workspaces:
        for pr in g.api.project.get_list(ws.id):
            for ds in g.api.dataset.get_list(pr.id):
                embedding_path = os.path.join(g.remote_embeddings_dir,
                                    sly.fs.get_file_name(g.remote_weights_path),
                                    ws.name,
                                    pr.name,
                                    ds.name + '.pkl')
                if g.api.file.exists(g.team_id, embedding_path):
                    datasets.append([ws, pr, ds, embedding_path])
    return datasets

def calc_embeddings(ds):
    ws, pr, ds, embedding_path = ds
    raise NotImplementedError
    # g.api.dataset.download(??)


def save_embeddings(embs, ds):
    """
     embeddings_dataset1.pkl ({1: {image_url_1: emb1, image_url_2: emb2, image_url_3: emb3, image_url_4: emb4}}, ...),
     embeddings_dataset2.pkl ({34: {image_hash_66: emb66, emb67, emb68, emb69]}, ...), …
    """
    raise NotImplementedError


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    download_model_and_config()
    sly.logger.info("Model has been successfully downloaded")
    sly.logger.debug("Script arguments", extra={
        "Remote weights": g.remote_weights_path,
        "Local weights": g.local_weights_path,
        "Local config path": g.local_config_path,
        "device": g.device
    })

    related_datasets = list_related_datasets()  # only datasets without calculated_embeddings

    for ds in related_datasets:
        embs = calc_embeddings(ds)
        save_embeddings(embs, ds)



if __name__ == "__main__":
    sly.main_wrapper("main", main)
