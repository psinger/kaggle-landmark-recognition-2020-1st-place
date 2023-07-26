import os
from pathlib import Path
import sys
import supervisely as sly
from supervisely.app.v1.app_service import AppService
from supervisely.io.fs import mkdir

my_app: AppService = AppService()
api = my_app.public_api
task_id = my_app.task_id

logger = sly.logger

model = None
device = 'cuda'

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

download_batch_size = 10
calc_batch_size = 10
gallery_data = None

project_info = api.project.get_info_by_id(project_id)
if project_info is None:  # for debug
    raise ValueError(f"Project with id={project_id} not found")

project_dir = os.path.join(my_app.data_dir, "visualize_MLTask")
temp_files = os.path.join(project_dir, "temp_files")

if os.path.exists(temp_files):  # clean temp
    sly.fs.clean_dir(temp_files)

converted_dir = os.path.join(temp_files, "converted_input")
sly.fs.mkdir(converted_dir)

projects_dir = os.path.join(temp_files, "projects")
sly.fs.mkdir(projects_dir)
checkpoints_dir = os.path.join(temp_files, "checkpoints")
sly.fs.mkdir(checkpoints_dir)
local_info_dir = os.path.join(temp_files, "info")
sly.fs.mkdir(local_info_dir)

project_dir = os.path.join(my_app.data_dir, "sly_project")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

artifacts_dir = os.path.join(my_app.data_dir, "artifacts")
sly.fs.mkdir(artifacts_dir)
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
sly.fs.mkdir(checkpoints_dir)


embeddings_dir = os.path.join(artifacts_dir, "embeddings")
sly.fs.mkdir(embeddings_dir)
sly.fs.clean_dir(embeddings_dir)

root_source_dir = str(Path(sys.argv[0]).parents[1])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)


sys.path.append(os.path.join(root_source_dir, 'src'))
sys.path.append(os.path.join(str(Path(sys.argv[0]).parents[2]), 'calculator'))
sys.path.append(os.path.join(str(Path(sys.argv[0]).parents[3]), 'src'))

source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
sly.logger.info(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

pascal_contour_color = [224, 224, 192]
selected_classes = []
class_color_dict = {}

# code for export-to-coco
# user = api.user.get_info_by_id(user_id)
user_name = "Supervisely"
project = api.project.get_info_by_id(project_id)
meta_json = api.project.get_meta(project_id)
meta = sly.ProjectMeta.from_json(meta_json)

# # embedding calculator
# local_dataset_path = os.path.join(my_app.data_dir, 'sly_dataset')
# remote_weights_path = os.environ['modal.state.slyFile']
# remote_embeddings_dir = os.environ['modal.state.slyEmbeddingsDir']
