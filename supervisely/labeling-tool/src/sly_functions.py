import ast
import json
import os
import re
from collections import OrderedDict
from urllib.parse import urlparse

import supervisely_lib as sly
import sly_globals as g


def camel_to_snake(string_to_process):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', string_to_process).lower()


def process_info_for_showing(info_data):
    processed_info = {}

    for key, value in info_data.items():
        processed_info[camel_to_snake(key).title()] = value

    return processed_info


def remove_keys_from_dict(keys_to_remove, data):
    for key in keys_to_remove:
        data.pop(key, None)


def sly_annotation_to_bbox(sly_label):
    rect: sly.Rectangle = sly_label.geometry.to_bbox()
    return [rect.top, rect.left,
            rect.bottom - rect.top,
            rect.right - rect.left]


def generate_data_for_nn_app(images_ids, figures_ids, annotations, padding):
    data_for_inference = []

    for index, (image_id, figure_id, label) in enumerate(zip(images_ids, figures_ids, annotations)):
        if label is None:
            raise ValueError(
                f"Label with id={figure_id} not found. Maybe cached annotation differs from the actual one. "
                f"Please clear cache on settings tab")

        image_info = g.api.image.get_info_by_id(image_id)
        image_url = image_info.full_storage_url
        bbox = sly_annotation_to_bbox(label)

        data_for_inference.append(
            {
                'index': index,
                'url': image_url,
                'bbox': bbox,
                'figure_id': figure_id
            }
        )

    return data_for_inference


def generate_data_for_calculator_app(embeddings_by_indexes, top_n):
    data_for_calculator = {
        'embeddings': [current_row['embedding'] for current_row in embeddings_by_indexes],
        'top_k': top_n
    }

    return data_for_calculator


def add_embeddings_to_cache_by_figures(embeddings_by_indexes, data_for_nn):
    for current_embedding in embeddings_by_indexes:
        current_figure_id = data_for_nn[current_embedding['index']]['figure_id']
        g.figures2embeddings[current_figure_id] = current_embedding['embedding']


def calculate_nearest_labels(images_ids, annotations, figures_ids, top_n=5, padding=0):
    data_for_nn = generate_data_for_nn_app(images_ids=images_ids, annotations=annotations,
                                           figures_ids=figures_ids, padding=padding)

    response = g.api.task.send_request(g.nn_session_id, "inference", data={
        'input_data': data_for_nn
    }, timeout=99999)
    embeddings_by_indexes = ast.literal_eval(json.loads(response))  # [{'index': 0, 'embedding': [...], ..}, ..]

    add_embeddings_to_cache_by_figures(embeddings_by_indexes, data_for_nn)
    data_for_calculator = generate_data_for_calculator_app(embeddings_by_indexes, top_n)

    response = g.api.task.send_request(g.calculator_session_id, "calculate_similarity", data={
        'input_data': data_for_calculator
    }, timeout=99999)

    nearest_labels = ast.literal_eval(json.loads(response))

    # {
    #     'pred_dist': [[1.0, ..], ..],
    #     'pred_labels': [['label1', ..], ..],
    #     'pred_urls': [['image_url1', ..], ..],
    # }

    return nearest_labels


def get_resized_image(image_storage_url, height):
    parsed_link = urlparse(image_storage_url)

    return f'{parsed_link.scheme}://{parsed_link.netloc}' \
           f'/previews/q/ext:jpeg/resize:fill:0:{height}:0/q:0/plain{parsed_link.path}'


def get_unique_elements(elements_list):
    used = set()
    return [x for x in elements_list if x not in used and (used.add(x) or True)]


def generate_data_to_show(nearest_labels):
    unique_labels = get_unique_elements(nearest_labels['pred_labels'])

    data_to_show = {pred_label: {} for pred_label in unique_labels}
    data_to_show = OrderedDict(data_to_show)

    for dist, label, url in zip(nearest_labels['pred_dist'],
                                nearest_labels['pred_labels'],
                                nearest_labels['pred_urls']):
        updated_dist = data_to_show[label].get('dist', 0) + dist
        updated_url = data_to_show[label].get('url', [])
        # updated_url.append({'preview': get_resized_image(url, 250)})  # ONLY ON DEBUG
        updated_url = [{'preview': get_resized_image(url, g.items_preview_size)}]

        data_to_show[label] = {'dist': updated_dist,
                               'url': updated_url}

    # for index, label in enumerate(data_to_show.keys()):
    #     data_to_show[label].update({'index': index})

    return dict(data_to_show)


def add_info_to_disable_buttons(data_to_show, assigned_tags):
    data_to_show = OrderedDict(data_to_show)
    for label, data in data_to_show.items():
        if label in assigned_tags:
            data_to_show[label].update({'assignDisabled': True})
        else:
            data_to_show[label].update({'assignDisabled': False})
    return dict(data_to_show)


def get_meta(project_id, optimize=True):
    if project_id not in g.project2meta or optimize is False:
        meta_json = g.api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)
        g.project2meta[project_id] = meta
    else:
        meta = g.project2meta[project_id]
    return meta


def update_project_meta(project_id, project_meta: sly.ProjectMeta):
    g.api.project.update_meta(project_id, project_meta.to_json())
    get_meta(project_id, optimize=False)


def _get_or_create_tag_meta(project_id, tag_meta):
    project_meta = get_meta(project_id)
    project_tag_meta: sly.TagMeta = project_meta.get_tag_meta(tag_meta.name)
    if project_tag_meta is None:
        project_meta = project_meta.add_tag_meta(tag_meta)
        update_project_meta(project_id, project_meta)
        project_meta = get_meta(project_id)
        project_tag_meta = project_meta.get_tag_meta(tag_meta.name)
    return project_tag_meta


def _assign_tag_to_object(project_id, figure_id, tag_meta):
    project_tag_meta: sly.TagMeta = _get_or_create_tag_meta(project_id, tag_meta)
    g.api.advanced.add_tag_to_object(project_tag_meta.sly_id, figure_id)


def assign_to_object(project_id, figure_id, class_name):
    # tag_meta = g.model_meta.tag_metas.get(class_name)
    tag_meta = sly.TagMeta(class_name, sly.TagValueType.NONE)
    _assign_tag_to_object(project_id, figure_id, tag_meta)


def get_image_path(image_id):
    info = get_image_info(image_id)
    local_path = os.path.join(g.cache_path, f"{info.id}{sly.fs.get_file_name_with_ext(info.name)}")
    if not sly.fs.file_exists(local_path):
        g.api.image.download_path(image_id, local_path)
    return local_path


def get_annotation(project_id, image_id, optimize=False):
    if image_id not in g.image2ann or not optimize:
        ann_json = g.api.annotation.download(image_id).annotation
        ann = sly.Annotation.from_json(ann_json, get_meta(project_id))
        g.image2ann[image_id] = ann
    else:
        ann = g.image2ann[image_id]
    return ann


def get_image_info(image_id):
    info = None
    if image_id not in g.image2info:
        info = g.api.image.get_info_by_id(image_id)
        g.image2info[image_id] = info
    else:
        info = g.image2info[image_id]
    return info


def clear():
    g.project2meta.clear()
    # image2info.clear()
    g.image2ann.clear()


def convert_dict_to_list(data_to_show):
    data_to_show_list = []
    for key, value in data_to_show.items():
        value['current_label'] = key
        data_to_show_list.append(value)
    return data_to_show_list


def set_flag_of_last_assigned_tag(assigned_tags, card_name, fields):
    current_card = g.api.task.get_field(g.task_id, f"state.{card_name}")

    if current_card:
        if current_card.get('current_label', '') not in assigned_tags:
            fields[f"state.{card_name}.assignDisabled"] = False
        else:
            fields[f"state.{card_name}.assignDisabled"] = True


def get_assigned_tags_names_by_label_annotation(label_annotation):
    assigned_tags = label_annotation.tags.to_json()
    return [assigned_tag.get('name', None) for assigned_tag in assigned_tags
            if assigned_tag.get('name', None) is not None]


def sort_by_dist(data_to_show):
    sorted_predictions_by_dist = sorted(data_to_show, key=lambda d: d['dist'], reverse=True)
    for index, row in enumerate(sorted_predictions_by_dist):
        row['index'] = index
        sorted_predictions_by_dist[index] = row

    return sorted_predictions_by_dist


def upload_data_to_tabs(nearest_labels, label_annotation):
    fields = {}

    assigned_tags = get_assigned_tags_names_by_label_annotation(label_annotation)

    set_flag_of_last_assigned_tag(assigned_tags, 'lastAssignedTag', fields)                        # Last assigned tab
    set_flag_of_last_assigned_tag(assigned_tags, 'selectedDatabaseItem', fields)                   # Database tab

    nearest_labels = {key: value[0] for key, value in nearest_labels.items()}                      # NN Prediction tab
    data_to_show = generate_data_to_show(nearest_labels)
    data_to_show = add_info_to_disable_buttons(data_to_show, assigned_tags)
    data_to_show = convert_dict_to_list(data_to_show)
    data_to_show = sort_by_dist(data_to_show)
    fields['data.predicted'] = data_to_show

    g.api.task.set_fields_from_dict(g.task_id, fields)

    return 0


def disable_assigned_buttons(card_name, fields):
    try:
        current_card = g.api.task.get_field(g.task_id, f"state.{card_name}")
        card_labels = [current_card.get('current_label', '')]

        set_flag_of_last_assigned_tag(card_labels, card_name, fields)
    except:
        pass


