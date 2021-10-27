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


session_id = os.environ["modal.state.sessionId"]


task_id = my_app.task_id
team_id = os.environ["context.teamId"]
workspace_id = os.environ["context.workspaceId"]


remote_embeddings_dir = '/GL-MetricLearning/embeddings/'

local_project_path = os.path.join(my_app.data_dir, 'project')

calc_batch_size = 512

root_source_dir = str(Path(sys.argv[0]).parents[3])
sys.path.append(os.path.join(root_source_dir, 'src'))

# DEBUG
sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)
