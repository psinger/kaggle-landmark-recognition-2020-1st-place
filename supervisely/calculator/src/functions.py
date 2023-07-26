import ast
import json
import logging
import os
import pickle
import tempfile
import uuid

import numpy as np
import supervisely as sly
from functools import partial

import sly_globals as g
from csv import writer, reader


def get_progress_cb(message, total, is_size=False):
    progress = sly.Progress(message, total, is_size=is_size)
    progress_cb = partial(update_progress, api=g.api, task_id=g.my_app.task_id, progress=progress)
    progress_cb(0)
    return progress_cb


def update_progress(count, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done_report(count)
    _update_progress_ui(api, task_id, progress)


def _update_progress_ui(api: sly.Api, task_id, progress: sly.Progress, stdout_print=False):
    if progress.need_report():
        fields = [
            {"field": "data.progressName", "payload": progress.message},
            {"field": "data.currentProgressLabel", "payload": progress.current_label},
            {"field": "data.totalProgressLabel", "payload": progress.total_label},
            {"field": "data.currentProgress", "payload": progress.current},
            {"field": "data.totalProgress", "payload": progress.total},
        ]
        api.app.set_fields(task_id, fields)
        if stdout_print is True:
            progress.report_if_needed()


def list_dirs(paths):
    all_paths = []
    for path in paths:
        new_paths = g.api.file.list(g.team_id, path)
        for p in new_paths:
            all_paths.append(p['path'])
    return all_paths if all_paths == paths else list_dirs(all_paths)


def check_model_connection():
    try:
        response = g.api.task.send_request(g.session_id, "get_info", data={}, timeout=5)
        response = json.loads(response)
        g.model_info = ast.literal_eval(response)
        sly.logger.info("🟩 Model has been successfully connected")
        sly.logger.info(f"⚙️ Model info:\n"
                        f"{response}")
    except Exception as ex:
        sly.logger.info("🟥 Сan not connect to the model!\n"
                        f"{ex}")
        exit()


def split_list_to_batches(input_list):
    splits_num = int(len(input_list) / g.batch_size) > 0 if int(len(input_list) / g.batch_size) > 0 else 1

    batches = np.array_split(input_list, splits_num, axis=0)
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
        image_descriptions = []
        for current_label in image_annotation.labels:
            sly_rectangle = current_label.geometry.to_bbox()
            image_bounding_boxes.append([sly_rectangle.top,
                                         sly_rectangle.left,
                                         sly_rectangle.bottom - sly_rectangle.top,
                                         sly_rectangle.right - sly_rectangle.left])

            image_tags.append(current_label.tags.keys() if len(current_label.tags) > 0 else [None])
            image_descriptions.append(json.loads(current_label.description) if current_label.description else {})
        data_for_each_image.append({'bbox': image_bounding_boxes,
                                    'tags': image_tags,
                                    'descriptions': image_descriptions})

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
                'tags': data_for_each_image['tags'][0],
                'description': data_for_each_image['descriptions'][0]
            })
            current_index += 1

        for current_patch_index, bounding_box in enumerate(data_for_each_image['bbox']):
            batch_for_inference.append({
                'index': current_index,
                'url': image_url,
                'bbox': bounding_box,
                'tags': data_for_each_image['tags'][current_patch_index],
                'description': data_for_each_image['descriptions'][current_patch_index]

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
                'bbox': image_data['bbox'],
                'description': image_data['description']
            })

            tag_to_data[current_tag] = data_by_tag


def process_placeholder_images(batch):
    placeholder_data = []
    images_to_inference = []

    for item in batch:
        if item['bbox'] == [0, 0, 1 - 1, 1 - 1]:
            placeholder_data.append({'index': item['index'],
                                     'embedding': None})
        else:
            images_to_inference.append(item)

    return placeholder_data, images_to_inference


def inference_batch(batch):
    embeddings_by_indexes, inference_items = process_placeholder_images(batch)
    logging.info('big req sent')
    response = g.api.task.send_request(g.session_id, "inference", data={'input_data': inference_items}, timeout=99999)
    embeddings_by_indexes.extend(ast.literal_eval(json.loads(response)))

    return embeddings_by_indexes


def get_uuid_by_string(input_string):
    return uuid.uuid3(uuid.NAMESPACE_DNS, input_string).hex


def dump_embeddings(dataset_id, packed_data):
    dataset_uuid = get_uuid_by_string(str(dataset_id))

    remote_pkl_path = os.path.join(g.remote_embeddings_dir, g.model_info['Model'],
                                   f"{g.workspace_info.name}_{g.workspace_id}",
                                   f"{g.project_info.name}_{g.project_id}",
                                   dataset_uuid[0], dataset_uuid[1], dataset_uuid[2],
                                   f'{dataset_uuid}.pkl'
                                   )

    temp_file_name_path = f'{g.my_app.data_dir}/temp_file.pkl'
    if os.path.isfile(temp_file_name_path):
        os.remove(temp_file_name_path)

    with open(temp_file_name_path, 'wb') as tmp_file:
        pickle.dump(packed_data, tmp_file)

    if g.api.file.exists(g.team_id, remote_pkl_path):
        g.api.file.remove(g.team_id, remote_pkl_path)

    g.api.file.upload(g.team_id, temp_file_name_path, remote_pkl_path)


def create_table(local_table_path):
    headers = ['dataset_id', 'dataset_name', 'filename', 'embeddings_count']

    with open(local_table_path, 'w') as opened_table:
        writer_object = writer(opened_table)
        writer_object.writerow(headers)


def update_table(dataset_id, packed_data):
    dataset_uuid = get_uuid_by_string(str(dataset_id))
    dataset_info = g.api.dataset.get_info_by_id(dataset_id)

    remote_table_path = os.path.join(
        g.remote_embeddings_dir,
        g.model_info['Model'],
        f"{g.workspace_info.name}_{g.workspace_id}",
        f"{g.project_info.name}_{g.project_id}",
        'embeddings_info.csv',
    )

    local_table_path_filtered = os.path.join(g.local_project_path, 'embeddings_table_filtered.csv')
    local_table_path_origin = os.path.join(g.local_project_path, 'embeddings_table_origin.csv')

    create_table(local_table_path_filtered)

    # Downloading from Team Files
    if g.api.file.exists(g.team_id, remote_table_path):
        g.api.file.download(g.team_id, remote_table_path, local_table_path_origin)
    else:
        create_table(local_table_path_origin)

    with open(local_table_path_origin, 'r') as original_table, open(local_table_path_filtered, 'a') as filtered_table:
        writer_object = writer(filtered_table)
        for row_index, row in enumerate(reader(original_table)):
            if row_index == 0 or row[2] == f"{dataset_uuid}.pkl":
                continue
            writer_object.writerow(row)

        writer_object.writerow([f"{dataset_id}",
                                f"{dataset_info.name}",
                                f"{dataset_uuid}.pkl",
                                f"{len(packed_data)}"
                                ])

    # Uploading to Team Files
    if g.api.file.exists(g.team_id, remote_table_path):
        g.api.file.remove(g.team_id, remote_table_path)
    file_info = g.api.file.upload(g.team_id, local_table_path_filtered, remote_table_path)

    g.api.task.set_output_directory(g.task_id, file_info.id, remote_table_path)


def write_packed_data(dataset_id, packed_data):
    dump_embeddings(dataset_id, packed_data)
    update_table(dataset_id, packed_data)


def get_images_count_in_project(project_id):
    images_count = 0
    datasets = g.api.dataset.get_list(project_id)
    for dataset_info in datasets:
        if dataset_info.images_count:
            images_count += dataset_info.images_count

    return images_count
