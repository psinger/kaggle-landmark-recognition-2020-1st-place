import json
import os
import shutil

import pandas
import supervisely as sly

from tqdm import tqdm

import sly_globals as g
import sly_functions as f

image_suffix = 'jpg'

input_data_dir = f'/root/data/products_10k'
output_data_dir = f'/root/data/10k_images_folder'

os.makedirs(output_data_dir, exist_ok=True)
sly.fs.clean_dir(output_data_dir)

table_path = os.path.join(input_data_dir, 'train.csv')
df = pandas.read_csv(table_path)


for img_name, img_class in tqdm(zip(df['name'], df['class']), total=len(df['name'])):
        original_image_path = os.path.join(input_data_dir, 'train', f'{img_name}')
        dist_path = os.path.join(output_data_dir, f'{img_class:06d}')
        os.makedirs(dist_path, exist_ok=True)

        dist_image_path = os.path.join(dist_path, f'{img_name}')

        shutil.copy(original_image_path, dist_image_path)

