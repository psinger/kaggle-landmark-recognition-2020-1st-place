import json
import os
import shutil

import supervisely as sly

from tqdm import tqdm

import sly_globals as g
import sly_functions as f

for seq_num in [3]:
    print(f'curr seq: {seq_num}')

    input_data_dir = f'/root/snacks_data/{seq_num}/snacks_{seq_num}'
    output_data_dir = f'/root/snacks_data/{seq_num}/train_image_folder'

    os.makedirs(output_data_dir, exist_ok=True)
    sly.fs.clean_dir(output_data_dir)


    input_images_paths = sorted(f.get_files_paths(input_data_dir, ['.png', '.jpg']))

    annotations = []

    for index, input_image_path in tqdm(enumerate(input_images_paths), total=len(input_images_paths)):
        try:
            image_shape = f.get_image_size(input_image_path)

            ann_path = input_image_path.replace('/img/', '/ann/').replace('.png', '.png.json').replace('.jpg', '.jpg.json')

            with open(ann_path, 'r') as file:
                ann_data = dict(json.load(file))

            # tag_name = ann_data['objects'][0]['tags'][0]['name']
            tag_name = ann_data['tags'][0]['name']

            class_label_output_path = os.path.join(output_data_dir, f'{tag_name}')
            os.makedirs(class_label_output_path, exist_ok=True)

            shutil.copy(input_image_path, os.path.join(output_data_dir, f'{tag_name}/{index}.png'))
        except Exception as ex:
            print(f"{index} — {ex}")

