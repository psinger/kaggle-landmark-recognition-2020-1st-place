import ast
import copy
import json

import supervisely as sly
import sly_globals as g
import sly_functions as f




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
            raise ValueError('Metric Learning model and AI Recommendations model must be the same!')

        keys_to_remove = ['weightsType', 'Model']
        f.remove_keys_from_dict(keys_to_remove, calculator_info)

        g.calculator_info = calculator_info

        response = api.task.send_request(task_id, "get_objects_database", data={}, timeout=999)
        g.items_database = response['database']

        database_to_show = [{'label': key, **value} for key, value in g.items_database.items()]

        for row in database_to_show:
            row.pop('url')

        g.figures_in_reference = response['figure_id']

        database_keys = list(database_to_show[0].keys()) if len(database_to_show) > 0 else []
        keys_to_show = [key for key in database_keys if 'name' in key.lower()][:3]

        fields = [
            {"field": f"state.databaseKeys", "payload": database_keys},
            {"field": f"state.selectedDescriptionsToShow", "payload": keys_to_show},
            {"field": f"data.calculatorStats", "payload": f.process_info_for_showing(g.calculator_info.copy())},
            {"field": f"data.itemsDatabase", "payload": database_to_show},
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
