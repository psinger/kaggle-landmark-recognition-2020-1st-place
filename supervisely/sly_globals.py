import os
import sys
import pathlib
import supervisely as sly
from dotenv import load_dotenv  # pip install python-dotenv\

load_dotenv("../debug.env")
load_dotenv("../secret_debug.env", override=True)
#
# my_app = sly.AppService()
# api = my_app.public_api
# task_id = my_app.task_id
#
# my_app.data_dir

logger = sly.logger

input_dir = '../input_data'
output_dir = '../output_data'


def get_files_paths(src_dir, extensions):
    files_paths = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    files_paths.append(file_path)

    return files_paths
