import json
import os
import sys
from pathlib import Path

import supervisely as sly
from supervisely.app.v1.app_service import AppService
# import dotenv
import ast

# dotenv.load_dotenv('./debug.env')
# dotenv.load_dotenv('./secret_debug.env')

logger = sly.logger

my_app: AppService = AppService()
api = my_app.public_api

selected_weights_type = None
selected_model = None
embeddings_stats = None


remote_embeddings_dir = '/GL-MetricLearning/embeddings/'
local_embeddings_dir = os.path.join(my_app.data_dir, 'local_embeddings')

embeddings_in_memory = []
placeholders_in_memory = []
embeddings_labels = []


team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

project_info = api.project.get_info_by_id(project_id)

task_id = my_app.task_id

batch_size = 64
custom_dataset_images_max_count = 1000
custom_label_title = 'custom_item_for_metric_learning'


custom_dataset_figures_in_reference = []


entry_point_path = Path(sys.argv[0])
root_source_dir = str(entry_point_path.parents[3])

sys.path.append(os.path.join(root_source_dir, 'src'))



# DEBUG
# sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)
