import os
from tqdm import tqdm


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


def write_csv_table(csv_path, table_data):
    with open(csv_path, 'w') as file:
        file.write('landmark_id,images\n')
        for curr_landmark_id, data in tqdm(table_data.items(),
                                           total=len(table_data.keys()),
                                           desc='writing to file'):
            if len(data) > 0:
                file.write(f'{curr_landmark_id},{" ".join(data)}\n')


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
        f'images in train: {images_in_train_counter}\n'
        f'avg img in class: {images_in_train_counter / classes_counter}\n'
    )


flag = 'test'

input_data_path = f'/root/gld_data/{flag}'
csv_input_file_path = f'/root/gld_data/{flag}.csv'
csv_output_file_path = f'/root/gld_data/{flag}_filtered.csv'

min_images_count_for_id = 1

table_rows = load_csv_table(csv_input_file_path)
images_names_all = set(get_existed_files_names(input_data_path, ['.jpg', '.jpeg']))

filtered_counter = 0

for landmark_id, images_names_id in tqdm(table_rows.items(), total=len(table_rows.keys()), desc='filtering data'):
    images_intersection = list(set(images_names_id) & images_names_all)
    if len(images_intersection) >= min_images_count_for_id:
        table_rows[landmark_id] = images_intersection
    else:
        table_rows[landmark_id] = []
        filtered_counter += 1

write_csv_table(csv_output_file_path, table_rows)
show_stats(table_rows)
