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

    with open(csv_path, 'r') as file:
        rows = file.read().split('\n')

    return rows


def write_csv_table(csv_path, table_data):
    with open(csv_path, 'w') as file:
        file.write('id,landmarks,Usage\n')
        for index, row in tqdm(enumerate(table_data), total=len(table_data), desc='writing to file'):
            file.write(f'{row}\n')


def show_stats(table_rows):

    classes_info = {}

    for index, row in tqdm(enumerate(table_rows), total=len(table_rows), desc='calc stats'):
        landmark_id = row.split(',')[1].strip()
        image_name = row.split(',')[0].strip()

        images_list = classes_info.get(landmark_id, [])
        images_list.append(image_name)
        classes_info[landmark_id] = images_list


    print(
        f'done.\n\n'
        f'classes count: {len(classes_info.keys())}\n'
        f'images in test: {len(table_rows)}\n'
        f'avg img in class: { len(table_rows) / len(classes_info.keys())}\n'
    )


flag = 'test'

input_data_path = f'/root/gld_data/{flag}'
csv_input_file_path = f'/root/gld_data/{flag}.csv'
csv_output_file_path = f'/root/gld_data/{flag}_filtered.csv'

min_images_count_for_id = 1

table_rows = load_csv_table(csv_input_file_path)
images_names_all = set(get_existed_files_names(input_data_path, ['.jpg', '.jpeg']))

filtered_counter = 0

filtered_data = []

for index, row in tqdm(enumerate(table_rows), total=len(table_rows), desc='filtering_data'):
    if index == 0 or len(row) == 0:
        continue
    landmark_id = row.split(',')[1].strip()
    image_name = row.split(',')[0]
    if len(landmark_id) != 0 and image_name in images_names_all:
        filtered_data.append(row)


write_csv_table(csv_output_file_path, filtered_data)
show_stats(filtered_data)
