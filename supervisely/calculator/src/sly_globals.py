import os
import sys
from pathlib import Path

import supervisely as sly
from supervisely.app.v1.app_service import AppService
# import dotenv
#
# dotenv.load_dotenv('./debug.env')
# dotenv.load_dotenv('./secret_debug.env')


logger = sly.sly_logger

my_app: AppService = AppService()
api = my_app.public_api


session_id = os.environ["modal.state.sessionId"]


task_id = my_app.task_id
team_id = int(os.environ["context.teamId"])
workspace_id = int(os.environ["context.workspaceId"])
project_id = int(os.environ["modal.state.slyProjectId"])

workspace_info = api.workspace.get_info_by_id(workspace_id)
project_info = api.project.get_info_by_id(project_id)

project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

remote_embeddings_dir = '/GL-MetricLearning/embeddings/'

local_project_path = os.path.join(my_app.data_dir, 'project')

os.makedirs(local_project_path, exist_ok=True)
sly.fs.clean_dir(local_project_path)

batch_size = 256
# batch_size = 10/
model_info = None

root_source_dir = str(Path(sys.argv[0]).parents[3])

sys.path.append(os.path.join(root_source_dir, 'src'))

# DEBUG
# sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)
