import json
import os
import sys
from pathlib import Path

import supervisely as sly
from supervisely.app.v1.app_service import AppService
import dotenv
import ast

import torch.cuda

dotenv.load_dotenv('./debug.env')
dotenv.load_dotenv('./secret_debug.env')

logger = sly.logger

my_app: AppService = AppService()
api = my_app.public_api

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])

model = None

task_id = my_app.task_id

device = os.environ['modal.state.device'] if 'cuda' in os.environ['modal.state.device'] and torch.cuda.is_available() \
    else 'cpu'

selected_weights_type = str(os.environ['modal.state.modelWeightsOptions'])
pretrained_models_table = ast.literal_eval(os.environ['modal.state.models'])

# pretrained_models_table = list(json.loads(str(os.environ['modal.state.models'])))  # debug

if selected_weights_type == 'pretrained':
    selected_model = os.environ['modal.state.selectedModel']
    model_info = None
    for row in pretrained_models_table:
        logger.info(row)
        if row['Model'] == selected_model:
            model_info = row
            break
    remote_weights_path = model_info['weightsUrl']
else:
    remote_weights_path = os.environ['modal.state.weightsPath']

local_dataset_path = os.path.join(my_app.data_dir, 'sly_dataset')
local_weights_path = None

batch_size = int(os.environ['modal.state.batchSize'])

entry_point_path = Path(sys.argv[0])
root_source_dir = str(entry_point_path.parents[3])

print(root_source_dir)

sys.path.append(os.path.join(root_source_dir, 'src'))

# DEBUG
# sly.fs.clean_dir(my_app.data_dir, ignore_errors=True)
