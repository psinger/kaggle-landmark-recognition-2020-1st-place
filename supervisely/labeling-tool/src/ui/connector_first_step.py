import ast
import json

import supervisely as sly
import sly_globals as g
import info_tab
import ui

import sly_functions as f


def init_fields(state, data):
    state['done1'] = False
    state['connectingToNN'] = False
    data['modelStats'] = {}

    data["ssOptionsNN"] = {
        "sessionTags": ["deployed_nn_embeddings"],
        "showLabel": False,
        "size": "small"
    }


def handle_model_errors(data):
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


@g.my_app.callback("connect_to_model")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def connect_to_model(api: sly.Api, task_id, context, state, app_logger):
    try:
        task_id = state['nnId']

        response = api.task.send_request(task_id, "get_info", data={}, timeout=3)
        model_info = ast.literal_eval(json.loads(response))

        keys_to_remove = ['weightsUrl', 'config']
        f.remove_keys_from_dict(keys_to_remove, model_info)

        g.model_info = model_info

        fields = [
            {"field": f"data.modelStats", "payload": f.process_info_for_showing(g.model_info.copy())},
            {"field": f"state.connectingToNN", "payload": False},
            {"field": f"state.done1", "payload": True},
            {"field": f"state.activeStep", "payload": 2},
        ]
        g.api.task.set_fields(g.task_id, fields)

        g.nn_session_id = task_id

    except Exception as ex:
        fields = [
            {"field": f"state.connectingToNN", "payload": False}
        ]
        g.api.task.set_fields(g.task_id, fields)
        raise ConnectionError('Cannot connect to Metric Learning model. '
                              f'Reason: {ex}')

