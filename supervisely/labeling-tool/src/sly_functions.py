import ast
import json
import re

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

    for index, (image_id, figure_id, annotation) in enumerate(zip(images_ids, figures_ids, annotations)):
        image_info = g.api.image.get_info_by_id(image_id)
        label = annotation.get_label_by_id(figure_id)

        if label is None:
            raise ValueError(
                f"Label with id={figure_id} not found. Maybe cached annotation differs from the actual one. "
                f"Please clear cache on settings tab")

        image_url = image_info.full_storage_url
        bbox = sly_annotation_to_bbox(label)

        data_for_inference.append(
            {
                'index': index,
                'url': image_url,
                'bbox': bbox
            }
        )

    return data_for_inference


def generate_data_for_calculator_app(embeddings_by_indexes, top_n):
    data_for_calculator = {
        'embeddings': [current_row['embedding'] for current_row in embeddings_by_indexes],
        'top_k': top_n
    }

    return data_for_calculator


def calculate_nearest_labels(images_ids, annotations, figures_ids, top_n=5, padding=0):
    data_for_nn = generate_data_for_nn_app(images_ids=images_ids, annotations=annotations,
                                           figures_ids=figures_ids, padding=padding)

    response = g.api.task.send_request(g.nn_session_id, "inference", data={
        'input_data': data_for_nn
    }, timeout=99999)
    embeddings_by_indexes = ast.literal_eval(json.loads(response))  # [{'index': 0, 'embedding': [...], ..}, ..]

    data_for_calculator = generate_data_for_calculator_app(embeddings_by_indexes, top_n)

    response = g.api.task.send_request(g.calculator_session_id, "calculate_similarity", data={
        'input_data': data_for_calculator
    }, timeout=99999)

    nearest_labels = ast.literal_eval(json.loads(response))  # {
                                                             #     'pred_dist': [1.0, ..],
                                                             #     'pred_labels': ['label1', ..],
                                                             #     'pred_urls': ['image_url1', ..],
                                                             # }

    return nearest_labels
