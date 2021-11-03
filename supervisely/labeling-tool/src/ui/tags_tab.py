import ast
import json

import supervisely_lib as sly
import tag_utils
import review_tab
import cache
import prediction

import sly_globals as g
import sly_functions as f


def init(data, state):
    data['predicted'] = None

    state["tagToAssign"] = None

    state["lastAssignedTag"] = None  # assign button
    state["predictedDataToReference"] = None  # reference button

    state["selectedFigureId"] = None
    state["collapsedTagsTabs"] = ['nn_predictions']


@g.my_app.callback("assign_tag_to_figure")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def assign_tag_to_figure(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.loading", True)
    fields = {"state.loading": False}
    try:
        project_id = context["projectId"]
        figure_id = state["selectedFigureId"]
        class_name = state["lastAssignedTag"]['current_label']

        f.assign_to_object(project_id, figure_id, class_name)
        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


@g.my_app.callback("add_to_reference")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def add_to_reference(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.loading", True)
    fields = {"state.loading": False}
    try:
        reference_data = state["predictedDataToReference"]

        # {
        #     'embedding' :[],
        #     'label' :[],
        #     'url' :[],
        #     'bbox' :[]
        # }

        project_id, image_id, figure_id = context["projectId"], context["imageId"], context["figureId"]

        reference_label = reference_data['current_label']
        embedding = g.figures2embeddings[figure_id]

        image_info = g.api.image.get_info_by_id(image_id)
        image_url = image_info.full_storage_url

        annotations_for_image = f.get_annotation(project_id, image_id)
        label_annotation = annotations_for_image.get_label_by_id(figure_id)
        bbox = f.sly_annotation_to_bbox(label_annotation)

        data_to_add = {'input_data': {
            'embedding': [embedding],
            'label': [reference_label],
            'url': [image_url],
            'bbox': [bbox]
        }}

        response = api.task.send_request(g.calculator_session_id, "add_new_embeddings_to_reference",
                                         data=data_to_add,
                                         timeout=999)

        updated_calculator_info = ast.literal_eval(json.loads(response))
        for key, value in updated_calculator_info.items():
            g.calculator_info[key] = value

        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e
