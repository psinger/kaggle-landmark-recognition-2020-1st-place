import ast

import supervisely_lib as sly

import json
import os
import pickle
import tempfile

import numpy as np

import sly_globals as g
import functions as f

import model_functions


def download_model_and_config():
    remote_model_dir, remote_model_weights_name = os.path.split(g.remote_weights_path)

    remote_model_config_name = sly.fs.get_file_name(g.remote_weights_path) + '.py'
    remote_config_file = os.path.join(remote_model_dir, remote_model_config_name)

    g.local_weights_path = os.path.join(g.my_app.data_dir, remote_model_weights_name)
    # g.local_config_path = os.path.join(g.my_app.data_dir, remote_model_config_name)  # if different configs

    g.api.file.download(g.team_id, g.remote_weights_path, g.local_weights_path)
    # g.api.file.download(g.team_id, remote_config_file, g.local_config_path)


def list_dirs(paths):
    all_paths = []
    for path in paths:
        new_paths = g.api.file.list(g.team_id, path)
        for p in new_paths:
            all_paths.append(p['path'])
    return all_paths if all_paths == paths else list_dirs(all_paths)


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
    # TODO: fix it! separate batches for images and infos.
    # Probably works only section under second except:  data_batches = np.array([e])
    # Value error raises due different shapes of data elements, not only because len_data < bs!
    # i.e. now - inference batch size not fixed
    try:
        data_batches = np.array_split(data, len(data) // int(bs))
    except ValueError:  # if bs > data
        if len(data) > 0:
            try:
                data_batches = np.array_split(data, 1)  # ok for image-info
            except ValueError:  # if data contains arrays of different sizes
                e = np.empty(len(data), dtype=object)
                for i, v in enumerate(data):
                    e[i] = v
                data_batches = np.array([e])
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
    return model_functions.calculate_embeddings_for_nps_batch(data)


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
    img_infos = g.api.image.get_info_by_id_batch(list(map(int, ids)))
    return [x.full_storage_url for x in img_infos]


def check_model_connection():
    try:
        response = g.api.task.send_request(g.session_id, "get_info", data={}, timeout=5)
        response = json.loads(response)
        sly.logger.info("🟩 Model has been successfully connected")
        sly.logger.info(f"⚙️ Model info:\n"
                        f"{response}")
    except Exception as ex:
        sly.logger.info("🟥 Сan not connect to the model!\n"
                        f"{ex}")
        exit()


def split_list_to_batches(input_list):
    batches = np.array_split(input_list, g.batch_size, axis=0)
    return [batch for batch in batches if batch.size > 0]


def jsons_to_annotations(annotations_info):
    return [sly.Annotation.from_json(current_annotation.annotation, g.project_meta)
            for current_annotation in annotations_info]


def get_data_for_each_image(images_annotations):
    data_for_each_image = []  # [{'bbox': [[TOP, LEFT, HEIGHT, WIDTH], [..]], ..],
                              # 'tags': [[[tag1, tag2], ..], ..]}, ..]

    for image_annotation in images_annotations:
        image_bounding_boxes = []
        image_tags = []
        for current_label in image_annotation.labels:
            sly_rectangle = current_label.geometry.to_bbox()
            image_bounding_boxes.append([sly_rectangle.top,
                                         sly_rectangle.left,
                                         sly_rectangle.bottom - sly_rectangle.top,
                                         sly_rectangle.right - sly_rectangle.left])

            image_tags.append(current_label.tags.keys() if len(current_label.tags) > 0 else [None])
        data_for_each_image.append({'bbox': image_bounding_boxes,
                                    'tags': image_tags})

    return data_for_each_image


def generate_batch_for_inference(images_urls, data_for_each_image):
    batch_for_inference = []
    current_index = 0
    for image_url, data_for_each_image in zip(images_urls, data_for_each_image):
        if len(data_for_each_image['bbox']) == 0:
            batch_for_inference.append({
                'index': current_index,
                'url': image_url,
                'bbox': None,
                'tags': data_for_each_image['tags'][0]
            })
            current_index += 1

        for current_patch_index, bounding_box in enumerate(data_for_each_image['bbox']):
            batch_for_inference.append({
                'index': current_index,
                'url': image_url,
                'bbox': bounding_box,
                'tags': data_for_each_image['tags'][current_patch_index]
            })
            current_index += 1

    return batch_for_inference


def pack_data(tag_to_data, batch, embeddings_by_indexes):
    original_indexes = [current_item['index'] for current_item in batch]
    indexes_to_embeddings = {current_item['index']: current_item['embedding'] for current_item in embeddings_by_indexes}

    general_indexes = sorted(list(set(original_indexes) & set(indexes_to_embeddings.keys())))

    for general_index in general_indexes:
        image_data = batch[general_index]

        current_tags = batch[general_index]['tags']
        for current_tag in current_tags:
            data_by_tag = tag_to_data.get(current_tag, [])

            data_by_tag.append({
                'url': image_data['url'],
                'embedding': indexes_to_embeddings[general_index],
                'bbox': image_data['bbox']
            })

            tag_to_data[current_tag] = data_by_tag


def inference_batch(batch):
    response = g.api.task.send_request(g.session_id, "inference", data={'input_data': batch}, timeout=99999)
    embeddings_by_indexes = ast.literal_eval(json.loads(response))

    return embeddings_by_indexes


@g.my_app.callback("calculate_embeddings_for_project")
@sly.timeit
def calculate_embeddings_for_project(api: sly.Api, task_id, context, state, app_logger):
    datasets_list = g.api.dataset.get_list(g.project_id)
    for current_dataset in datasets_list:
        packed_data = {}

        images_info = api.image.get_list(current_dataset.id)

        progress = sly.Progress("processing dataset:", len(images_info))

        images_ids = split_list_to_batches([current_image_info.id for current_image_info in images_info])
        images_urls = split_list_to_batches([current_image_info.full_storage_url for current_image_info in images_info])

        for ids_batch, urls_batch in zip(images_ids, images_urls):
            ann_infos = api.annotation.download_batch(current_dataset.id, json.loads(str(ids_batch.tolist())))
            ann_objects = jsons_to_annotations(ann_infos)

            data_for_each_image = get_data_for_each_image(ann_objects)
            batch_for_inference = generate_batch_for_inference(urls_batch, data_for_each_image)
            embeddings_by_indexes = inference_batch(batch_for_inference)

            pack_data(packed_data, batch_for_inference, embeddings_by_indexes)

            progress.iters_done_report(g.batch_size if len(ids_batch) % g.batch_size == 0
                                       else len(ids_batch) % g.batch_size)


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "context.sessionId": g.session_id
    })

    check_model_connection()
    g.my_app.run(initial_events=[{"command": "calculate_embeddings_for_project"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
