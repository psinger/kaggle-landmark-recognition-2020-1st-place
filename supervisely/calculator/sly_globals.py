import os
import sys
from pathlib import Path

import supervisely_lib as sly
import dotenv

dotenv.load_dotenv('./debug.env')
dotenv.load_dotenv('./secret_debug.env')


logger = sly.logger

my_app = sly.AppService()
api = my_app.public_api


model = None

task_id = my_app.task_id
team_id = os.environ["context.teamId"]
workspace_id = os.environ["context.workspaceId"]
device = os.environ['context.deviceId']


remote_weights_path = os.environ['modal.state.slyFile']
remote_embeddings_dir = os.environ['modal.state.slyEmbeddingsDir']

local_dataset_path = os.path.join(my_app.data_dir, 'sly_dataset')
local_weights_path = None

download_batch_size = os.environ['modal.state.downloadBatchSize']
calc_batch_size = os.environ['modal.state.batchSize']
only_current_workspace = int(os.environ['modal.state.OnlyCurrentWorkspace'])  # 0 or 1


root_source_dir = str(Path(sys.argv[0]).parents[2])
sys.path.append(os.path.join(root_source_dir, 'src'))


# DEBUG
sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)
