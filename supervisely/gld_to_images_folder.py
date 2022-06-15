import json
import os
import shutil

import supervisely as sly

from tqdm import tqdm

import sly_globals as g
import sly_functions as f

image_suffix = 'jpg'

input_data_dir = f'/root/gld_data/'
output_data_dir = f'/root/gld_images_folder'

os.makedirs(output_data_dir, exist_ok=True)
sly.fs.clean_dir(output_data_dir)

table_path = os.path.join(input_data_dir, 'test_filtered.csv')
class_to_images_mapping = f.read_gld_table(table_path, delimiter=',')


for class_id, images_ids in tqdm(class_to_images_mapping.items(), total=len(class_to_images_mapping.keys())):
    for image_id in images_ids:

        original_image_path = os.path.join(input_data_dir, 'test',
                                           image_id[0], image_id[1], image_id[2], f'{image_id}.{image_suffix}')

        dist_path = os.path.join(output_data_dir, str(class_id))
        os.makedirs(dist_path, exist_ok=True)

        dist_image_path = os.path.join(dist_path, f'{image_id}.{image_suffix}')

        shutil.copy(original_image_path, dist_image_path)

#     try:
#         image_shape = f.get_image_size(input_image_path)
#
#         ann_path = input_image_path.replace('/img/', '/ann/').replace('.png', '.png.json').replace('.jpg', '.jpg.json')
#
#         with open(ann_path, 'r') as file:
#             ann_data = dict(json.load(file))
#
#         # tag_name = ann_data['objects'][0]['tags'][0]['name']
#         tag_name = ann_data['tags'][0]['name']
#
#         class_label_output_path = os.path.join(output_data_dir, f'{tag_name}')
#         os.makedirs(class_label_output_path, exist_ok=True)
#
#         shutil.copy(input_image_path, os.path.join(output_data_dir, f'{tag_name}/{index}.png'))
#     except Exception as ex:
#         print(f"{index} — {ex}")
#
