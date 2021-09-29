import json
import os
import shutil

import cv2

import sly_globals as g


def get_files_paths(src_dir, extensions):
    files_paths = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    files_paths.append(file_path)

    return files_paths


def get_image_size(image_path):
    # return [56, 56]
    return cv2.imread(image_path).shape


def write_white_blank_picture(input_picture_path):
    output_picture_path = os.path.join(g.output_dir, 'ds0', 'img')
    os.makedirs(output_picture_path, exist_ok=True)

    import numpy as np

    img_size = get_image_size(image_path=input_picture_path)

    img_1 = np.zeros([img_size[0], img_size[1], 1], dtype=np.uint8)
    img_1.fill(255)
    # or img[:] = 255
    image_name = input_picture_path[input_picture_path.rindex('/') + 1:]

    cv2.imwrite(os.path.join(output_picture_path, image_name), img_1)


def copy_picture(input_picture_path):
    output_picture_path = os.path.join(g.output_dir, 'ds0', 'img')
    os.makedirs(output_picture_path, exist_ok=True)

    image_name = input_picture_path[input_picture_path.rindex('/') + 1:]

    shutil.copy(input_picture_path, os.path.join(output_picture_path, image_name))


def dump_annotations(annotations, image_name):
    output_ann_path = os.path.join(g.output_dir, 'ds0', 'ann')
    os.makedirs(output_ann_path, exist_ok=True)

    output_ann_path = os.path.join(output_ann_path, f"{image_name}.json")

    with open(output_ann_path, 'w') as file:
        json.dump(annotations.to_json(), file)


def dump_picture_data(input_picture_path, annotation):
    image_name = input_picture_path[input_picture_path.rindex('/') + 1:]


    # write_white_blank_picture(input_picture_path)
    copy_picture(input_picture_path)
    dump_annotations(annotation, image_name)


def dump_meta(project_meta):
    output_meta_path = os.path.join(g.output_dir, 'meta.json')

    # meta_json = json.dumps(project_meta.to_json(), indent=4)
    with open(output_meta_path, 'w') as file:
        json.dump(project_meta.to_json(), file)
