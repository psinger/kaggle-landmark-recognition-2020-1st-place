import os
import shutil

from tqdm import tqdm

import uuid
from glob import glob


def get_uuid_by_string(input_string):
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, input_string)).replace('-', '')[:16]


def get_files_paths(src_dir, extensions):
    files_paths = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    files_paths.append(file_path)

    return files_paths


def get_existed_files_names(src_dir, extensions):
    files_names = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_name = file.split('.')[0]
                    files_names.append(file_name)

    return files_names


def load_csv_table(csv_path):
    landmarks_mappings = {}
    with open(csv_path, 'r') as file:
        rows = file.read().split('\n')

    for index, row in tqdm(enumerate(rows), total=len(rows), desc='reading csv'):
        if index == 0 or len(row) == 0:
            continue
        landmark_id = row.split(',')[0].strip()
        images = row.split(',')[1]
        if len(images) > 0:
            landmarks_mappings[landmark_id] = images.split()

    return landmarks_mappings


def write_csv_table(csv_path, table_data, is_train=False):
    with open(csv_path, 'w') as file:

        if is_train:
            file.write('landmark_id,id\n')

        else:
            file.write('landmarks,id\n')

        for curr_landmark_id, data in tqdm(table_data.items(),
                                           total=len(table_data.keys()),
                                           desc='writing to file'):
            if len(data) > 0:
                for curr_image_id in data:
                    file.write(f'{curr_landmark_id},{curr_image_id}\n')


def show_stats(table_data):
    classes_counter = 0
    images_in_train_counter = 0

    for curr_landmark_id, data in tqdm(table_data.items(),
                                       total=len(table_data.keys()),
                                       desc='calculating stats'):
        if len(data) > 0:
            classes_counter += 1
            images_in_train_counter += len(data)

    print(
        f'done.\n\n'
        f'classes count: {classes_counter}\n'
        f'images in {flag}: {images_in_train_counter}\n'
        f'avg img in class: {images_in_train_counter / classes_counter}\n'
    )


flag = 'train'


for seq_num in range(1, 6):
    print(f'processing: {seq_num}')

    dataset_root_path = '/root/snacks_data/'
    input_data_path = os.path.join(dataset_root_path, f'{seq_num}', f'train_image_folder')
    output_data_path = os.path.join(dataset_root_path, f'{seq_num}', f'{flag}')

    csv_output_file_path = f'/root/snacks_data/{seq_num}/{flag}_filtered.csv'

    classes_labels = os.listdir(input_data_path)

    out_data_dict = {}


    for class_label in tqdm(classes_labels):  # by every class in image folder ds
        input_data_label_path = os.path.join(input_data_path, class_label)
        images_paths = get_files_paths(input_data_label_path, ['.jpg', '.png'])

        for image_path in images_paths:   # by every image in current class
            img_name = str(image_path.split('/')[-1]).split('.')[0]
            img_extension = str(image_path.split('/')[-1]).split('.')[-1]

            img_uuid = get_uuid_by_string(img_name + class_label + flag)  # generate unique image name

            gld_output_class_path = os.path.join(output_data_path, f'{img_uuid[0]}/{img_uuid[1]}/{img_uuid[2]}')
            os.makedirs(gld_output_class_path, exist_ok=True)
            gld_output_class_image_path = os.path.join(gld_output_class_path, f'{img_uuid}.{img_extension}')
            shutil.copy(image_path, gld_output_class_image_path)  # copy image with unique name

            class_images = out_data_dict.get(class_label, [])  # update out_data_dict
            class_images.append(img_uuid)
            out_data_dict[class_label] = class_images

    write_csv_table(csv_output_file_path, out_data_dict, is_train=(flag == 'train'))
    show_stats(out_data_dict)






