import os
import supervisely_lib as sly

my_app = sly.AppService()
api = my_app.public_api

task_id = my_app.task_id
team_id = os.environ["context.teamId"]
workspace_id = os.environ["context.workspaceId"]
device = os.environ['context.deviceId']

remote_weights_path = os.environ['modal.state.slyFile']
remote_embeddings_dir = os.environ['modal.state.slyEmbeddingsDir']
only_current_workspace = int(os.environ['modal.state.OnlyCurrentWorkspace'])  # 0 or 1
# DEBUG
sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)
