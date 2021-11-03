import ast
import json

import supervisely_lib as sly
import sly_globals as g
import sly_functions as f


import info_tab
import ui
import cache


def init_fields(state, data):
    state['done2'] = False

    # state['done2'] = True  # DEBUG
    state['connectingToCalculator'] = False
    data['calculatorStats'] = {}

    data["ssOptionsCalculator"] = {
        "sessionTags": ["deployed_nn_recommendations"],
        "showLabel": False,
        "size": "small"
    }


def handle_model_errors(data):
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


@g.my_app.callback("connect_to_calculator")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def connect_to_calculator(api: sly.Api, task_id, context, state, app_logger):
    try:
        task_id = state['calculatorId']
        response = api.task.send_request(task_id, "get_info", data={}, timeout=3)

        calculator_info = ast.literal_eval(json.loads(response))

        if calculator_info['Model'] != g.model_info['Model']:
            raise ValueError('Metric Learning model and Similarity Calculator model must be the same!')

        keys_to_remove = ['weightsType', 'Model']
        f.remove_keys_from_dict(keys_to_remove, calculator_info)

        g.calculator_info = calculator_info

        fields = [
            {"field": f"data.calculatorStats", "payload": f.process_info_for_showing(g.calculator_info.copy())},
            {"field": f"state.connectingToCalculator", "payload": False},
            {"field": f"state.done2", "payload": True},
            {"field": f"state.activeStep", "payload": 2},
        ]
        g.api.task.set_fields(g.task_id, fields)

        g.calculator_session_id = task_id

    except Exception as ex:
        fields = [
            {"field": f"state.connectingToCalculator", "payload": False}
        ]
        g.api.task.set_fields(g.task_id, fields)
        raise ConnectionError('Cannot connect to calculator.'
                              f'Reason: {ex}')
