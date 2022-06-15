import ast
import copy
import json
import os
import re
from collections import OrderedDict
from dataclasses import fields
from urllib.parse import urlparse

import supervisely as sly

import sly_globals as g

from functools import lru_cache


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

        image_info = g.spawn_api.image.get_info_by_id(image_id)
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

    if len(embeddings_by_indexes) != len(data_for_nn):
        raise ValueError(f'Data error. Check that the label is selected correctly.')

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

    for dist, label in zip(nearest_labels['pred_dist'],
                           nearest_labels['pred_labels']):
        data_to_show[label]['dist'] = data_to_show[label].get('dist', 0) + dist

        if data_to_show[label].get('url', None) is None:
            data_to_show[label]['url'] = get_urls_by_label(label)

        if data_to_show[label].get('description', None) is None:
            data_to_show[label]['description'] = get_item_description_by_label(label)

    return dict(data_to_show)


def add_info_to_disable_buttons(data_to_show, assigned_tags, fields, state):
    reference_disabled = True
    selected_figure_id = fields.get('state.selectedFigureId', -1)
    if selected_figure_id not in g.figures_in_reference:
        reference_disabled = False

    data_to_show = OrderedDict(data_to_show)
    for label, data in data_to_show.items():
        if label in assigned_tags or (len(assigned_tags) > 0 and state['tagPerImage']):
            data_to_show[label].update({'assignDisabled': True,
                                        'referenceDisabled': reference_disabled})
        else:
            data_to_show[label].update({'assignDisabled': False,
                                        'referenceDisabled': reference_disabled})

    return dict(data_to_show)


def get_meta(project_id, from_server=False):
    if from_server is True or project_id not in g.project2meta:
        meta_json = g.spawn_api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)
        g.project2meta[project_id] = meta
    else:
        meta = g.project2meta[project_id]
    return meta


def update_project_meta(project_id, project_meta: sly.ProjectMeta):
    sly.logger.info(f'update_project_meta: {project_id=}, {g.spawn_user_login=}')
    g.spawn_api.project.update_meta(project_id, project_meta.to_json())


def _get_or_create_tag_meta(project_id, tag_meta):
    for get_from_server_flag in [False, True]:  # check tag in local and remote metas
        project_meta = get_meta(project_id, from_server=get_from_server_flag)
        project_tag_meta: sly.TagMeta = project_meta.get_tag_meta(tag_meta.name)
        sly.logger.info(f'_get_or_create_tag_meta: {project_tag_meta is None=}, {get_from_server_flag=}')
        if project_tag_meta is not None:
            break

    if project_tag_meta is None:
        project_meta = project_meta.add_tag_meta(tag_meta)  # add tag to newest meta
        update_project_meta(project_id, project_meta)
        project_meta = get_meta(project_id, from_server=True)
        project_tag_meta = project_meta.get_tag_meta(tag_meta.name)
    return project_tag_meta


def _assign_tag_to_object(project_id, figure_id, tag_meta):
    project_tag_meta: sly.TagMeta = _get_or_create_tag_meta(project_id, tag_meta)
    g.api.advanced.add_tag_to_object(project_tag_meta.sly_id, figure_id)


def assign_to_object(project_id, figure_id, class_name):
    sly.logger.info(f'assign_to_object: {project_id=}, {figure_id=}, {class_name=}')
    tag_meta = sly.TagMeta(class_name, sly.TagValueType.NONE)
    _assign_tag_to_object(project_id, figure_id, tag_meta)


def get_image_path(image_id):
    info = get_image_info(image_id)
    local_path = os.path.join(g.cache_path, f"{info.id}{sly.fs.get_file_name_with_ext(info.name)}")
    if not sly.fs.file_exists(local_path):
        g.spawn_api.image.download_path(image_id, local_path)
    return local_path


# @lru_cache(maxsize=10)
def get_annotation(project_id, image_id, optimize=False):
    if image_id not in g.image2ann or not optimize:
        ann_json = g.spawn_api.annotation.download(image_id).annotation
        ann = sly.Annotation.from_json(ann_json, get_meta(project_id))
        g.image2ann[image_id] = ann
    else:
        ann = g.image2ann[image_id]

    g.figures_on_frame_count = len(ann.labels)
    return ann


def get_image_info(image_id):
    info = None
    if image_id not in g.image2info:
        info = g.spawn_api.image.get_info_by_id(image_id)
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


def get_assigned_tags_names_by_label_annotation(label_annotation):
    assigned_tags = label_annotation.tags.to_json()
    return [assigned_tag.get('name', None) for assigned_tag in assigned_tags
            if assigned_tag.get('name', None) is not None]


def get_tag_id_by_tag_name(label_annotation, tag_name):
    assigned_tags = label_annotation.tags

    for current_tag in assigned_tags:
        if current_tag.name == tag_name:
            return current_tag.sly_id
            # return None

    return None


def sort_by_dist(data_to_show):
    sorted_predictions_by_dist = sorted(data_to_show, key=lambda d: d['dist'], reverse=True)
    for index, row in enumerate(sorted_predictions_by_dist):
        row['index'] = index
        sorted_predictions_by_dist[index] = row

    return sorted_predictions_by_dist


def get_item_description_by_label(current_label):
    item = copy.deepcopy(g.items_database.get(current_label, {}))
    keys_to_clear = ['url']
    for current_key in keys_to_clear:
        try:
            item.pop(current_key)
        except:
            pass

    return item


def update_review_tags_tab(assigned_tags, fields):
    items_for_review = []
    for current_tag in assigned_tags:
        items_for_review.append({
            'current_label': current_tag,
            'url': get_urls_by_label(current_tag),
            'removingDisabled': False,
            'description': get_item_description_by_label(current_tag)
        })

    if len(items_for_review) == 0:
        fields['state.tagsForReview'] = None
    else:
        fields['state.tagsForReview'] = items_for_review


def update_card_buttons(card_name, assigned_tags, fields, state):
    current_card = fields.get(f"state.{card_name}", None)

    if current_card is None:
        current_card = g.api.task.get_field(g.task_id, f"state.{card_name}")

    if current_card:
        assign_disabled = True
        reference_disabled = True

        if current_card.get('current_label', '') not in assigned_tags and not (
                len(assigned_tags) > 0 and state['tagPerImage']):
            assign_disabled = False

        selected_figure_id = fields.get('state.selectedFigureId', -1)
        if selected_figure_id not in g.figures_in_reference:
            reference_disabled = False

        set_buttons(assign_disabled=assign_disabled, reference_disabled=reference_disabled, card_name=card_name,
                    fields=fields)


def upload_data_to_tabs(nearest_labels, label_annotation, fields, state):
    assigned_tags = get_assigned_tags_names_by_label_annotation(label_annotation)

    update_review_tags_tab(assigned_tags, fields)  # Review tags tab

    update_card_buttons('lastAssignedTag', assigned_tags, fields, state)  # Last assigned tab
    update_card_buttons('selectedDatabaseItem', assigned_tags, fields, state)  # Database tab

    nearest_labels = {key: value[0] for key, value in nearest_labels.items()}  # NN Prediction tab
    data_to_show = generate_data_to_show(nearest_labels)
    data_to_show = add_info_to_disable_buttons(data_to_show, assigned_tags, fields, state)
    data_to_show = convert_dict_to_list(data_to_show)
    data_to_show = sort_by_dist(data_to_show)
    fields['data.predicted'] = data_to_show


def get_urls_by_label(selected_label):
    label_info = g.items_database[selected_label]
    return [{'preview': get_resized_image(current_url, g.items_preview_size)}
            for current_url in label_info['url']][:g.items_preview_count]


def remove_from_object(project_id, figure_id, tag_name, tag_id):
    project_meta = get_meta(project_id)
    project_tag_meta: sly.TagMeta = project_meta.get_tag_meta(tag_name)
    if project_tag_meta is None:
        raise RuntimeError(f"Tag {tag_name} not found in project meta")
    g.api.advanced.remove_tag_from_object(project_tag_meta.sly_id, figure_id, tag_id)


def set_button_flag(card_name, flag_name, flag_value, fields):
    current_card = g.api.task.get_field(g.task_id, f"state.{card_name}")
    if current_card:
        fields[f"state.{card_name}.{flag_name}"] = flag_value


def set_buttons(assign_disabled, reference_disabled, card_name, fields):
    set_button_flag(flag_name='assignDisabled', flag_value=assign_disabled, card_name=card_name, fields=fields)
    set_button_flag(flag_name='referenceDisabled', flag_value=reference_disabled, card_name=card_name, fields=fields)


def get_tagged_objects_count_on_frame(annotation):
    tagged_objects = 0
    for label in annotation.labels:
        if len(label.tags) > 0:
            tagged_objects += 1
    return tagged_objects
