import os

import sly_globals as g
import supervisely_lib as sly
import numpy as np
import pickle
import tempfile


def download_model_and_config():
    remote_model_dir, remote_model_weights_name = os.path.split(g.remote_weights_path)

    remote_model_config_name = sly.fs.get_file_name(g.remote_weights_path) + '.py'
    remote_config_file = os.path.join(remote_model_dir, remote_model_config_name)

    g.local_weights_path = os.path.join(g.my_app.data_dir, remote_model_weights_name)
    g.local_config_path = os.path.join(g.my_app.data_dir, remote_model_config_name)

    g.api.file.download(g.team_id, g.remote_weights_path, g.local_weights_path)
    g.api.file.download(g.team_id, remote_config_file, g.local_config_path)


def list_related_datasets():
    workspaces = [
        g.api.workspace.get_info_by_id(int(g.workspace_id))] if g.only_current_workspace else g.api.workspace.get_list(
        g.team_id)
    datasets = []
    for ws in workspaces:
        for pr in g.api.project.get_list(ws.id):
            for ds in g.api.dataset.get_list(pr.id):
                embedding_path = os.path.join(g.remote_embeddings_dir,
                                              sly.fs.get_file_name(g.remote_weights_path),
                                              ws.name,
                                              pr.name,
                                              ds.name + '.pkl')
                if not g.api.file.exists(g.team_id, embedding_path):
                    datasets.append([ws, pr, ds, embedding_path])
    return datasets


def get_image_info_batch(ds):
    image_info_list = g.api.image.get_list(ds.id)
    try:
        image_info_batches = np.array_split(image_info_list, len(image_info_list) // int(g.download_batch_size))
    except ValueError:
        if len(image_info_list) > 0:
            image_info_batches = np.array_split(image_info_list, 1)
        else:
            return None
    return image_info_batches


def get_batches(data, bs):
    try:
        data_batches = np.array_split(data, len(data) // int(bs))
    except ValueError:
        if len(data) > 0:
            data_batches = np.array_split(data, 1)
        else:
            return None
    return data_batches


def download_images_and_infos_batch(ds, img_ids):
    progress = sly.Progress("Images downloaded: ", len(img_ids))
    image_nps = g.api.image.download_nps(ds.id, img_ids, progress_cb=progress.iters_done_report)
    progress = sly.Progress("Annotations downloaded: ", len(img_ids))
    ann_infos = g.api.annotation.download_batch(ds.id, img_ids, progress_cb=progress.iters_done_report)
    return image_nps, ann_infos


def inference(data):
    return [["im super embedding"] for x in data]


def batch_inference(data_batches):
    """
    :param data_batches: np.arrays: [[img, img][img,img]]
    :return: result of inference
    """
    pred = []
    for data in data_batches:
        p = inference(data)
        pred.extend(p)
    return pred


def _get_crops(img, ann):
    crops = []
    for l in ann.labels:
        rectangle = l.geometry.to_bbox()
        rh, rw = rectangle.to_size()
        if (rh, rw) != img.shape[:2]:
            crop = sly.imaging.image.crop(img, rectangle)
        else:
            crop = img
        crops.append(crop)
    return crops


def _get_labels(ann):
    tags = []
    for l in ann.labels:
        try:
            tag = [x.name for x in l.tags][0]  # WARN!! Only 1 tag per obj
        except IndexError:
            tag = None
        tags.append(tag)
    return tags


def get_crops_and_labels(image_nps, ann_infos, meta):
    crops, tags, original_img_ids = [], [], []
    for img, info in zip(image_nps, ann_infos):
        ann = sly.Annotation.from_json(info.annotation, meta)
        crops.extend(_get_crops(img, ann))

        tag = _get_labels(ann)
        tags.extend(tag)
        for _ in tag:
            original_img_ids.append(info.image_id)

    return crops, tags, original_img_ids


def process_dataset(dataset):
    ws, pr, ds, embedding_path = dataset
    meta = sly.ProjectMeta.from_json(g.api.project.get_meta(pr.id))
    image_info_list = g.api.image.get_list(ds.id)
    image_info_batches = get_batches(image_info_list, bs=g.download_batch_size)

    if image_info_batches is None:
        return [], [], []

    result = []
    all_gt_tags = []
    all_image_ids = []

    for batch in image_info_batches:
        img_ids = batch.T[0].tolist()
        image_nps, ann_infos = download_images_and_infos_batch(ds, img_ids)
        crops, tags, original_img_ids = get_crops_and_labels(image_nps, ann_infos, meta)
        all_gt_tags.extend(tags)
        all_image_ids.extend(original_img_ids)

        # Inference section
        crops_batches = get_batches(crops, bs=g.calc_batch_size)
        pred = batch_inference(crops_batches)
        result.extend(pred)

    return result, all_gt_tags, all_image_ids


def create_embeddings_dict(embeddings, gt, urls):
    """
     embeddings_dataset1.pkl ({1: {image_url_1: emb1, image_url_2: emb2, image_url_3: emb3, image_url_4: emb4}}, ...),
    """
    embeddings, gt, urls = np.array(embeddings), np.array(gt), np.array(urls)
    pkl_data = {}
    for unique_label in set(gt):
        if unique_label is not None:
            u = urls[gt == unique_label]
            e = embeddings[gt == unique_label]
            pkl_data[int(unique_label)] = dict(zip(u, e))

    return pkl_data


def upload_embedding_dict(ds, pkl_data):
    _, _, _, remote_pkl_path = ds
    with tempfile.NamedTemporaryFile() as tmp_file:
        pickle.dump(pkl_data, tmp_file)
        g.api.file.upload(g.team_id, tmp_file.name, remote_pkl_path)
    sly.logger.info(f"Upload pkl: {remote_pkl_path}")

def get_img_urls(ids):
    # TODO: Fix it! double image info downloading
    img_infos = g.api.image.get_info_by_id_batch(list(map(int,ids)))
    return [x.full_storage_url for x in img_infos]


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
        embeddings, gt, parent_img_ids = process_dataset(ds) # HERE IS GT!
        urls = get_img_urls(parent_img_ids)
        emb_dict = create_embeddings_dict(embeddings, gt, urls)
        upload_embedding_dict(ds, emb_dict)

if __name__ == "__main__":
    sly.main_wrapper("main", main)
