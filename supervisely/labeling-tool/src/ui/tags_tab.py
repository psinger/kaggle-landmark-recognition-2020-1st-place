import ast
import json

import supervisely as sly
import tag_utils

import sly_globals as g
import sly_functions as f


def init(data, state):
    data['predicted'] = None  # NN predictions tab items
    state['tagsForReview'] = None  # review assigned tags items

    state["tagToAssign"] = None
    state["tagToRemove"] = None

    state["lastAssignedTag"] = None  # last assigned tab item
    state["predictedDataToReference"] = None  # reference button

    state["selectedFigureId"] = None
    state["collapsedTagsTabs"] = ['nn_predictions']

    state["copyingMode"] = False

    state['annotatedFiguresCount'] = 0
    state['allFiguresCount'] = 0


@g.my_app.callback("assign_tag_to_figure")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def assign_tag_to_figure(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.loading", True)
    fields = {"state.loading": False}
    try:
        project_id = context["projectId"]
        figure_id = state["selectedFigureId"]
        class_name = state["lastAssignedTag"]['current_label']

        f.assign_to_object(project_id, figure_id, class_name)

        g.annotated_figures_count += 1
        fields.update({"state.annotatedFiguresCount": g.annotated_figures_count,
                       "state.allFiguresCount": g.figures_on_frame_count})

        api.task.set_fields_from_dict(task_id, fields)

        if state.get('addEveryAssignedToReference', False):
            state['itemToReference'] = state["lastAssignedTag"]
            add_to_reference(api, task_id, context, state, app_logger)

    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        if "Tag already exists" in e.args[0]:
            return -1
        raise e


@g.my_app.callback("remove_tag_from_figure")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def remove_tag_from_figure(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.loading", True)
    fields = {"state.loading": False}
    try:
        project_id, image_id, figure_id = context["projectId"], context["imageId"], state["selectedFigureId"]

        tag_to_remove_name = state["tagToRemove"]

        annotations_for_image = f.get_annotation(project_id, image_id)
        label_annotation = annotations_for_image.get_label_by_id(figure_id)

        tag_id = f.get_tag_id_by_tag_name(label_annotation, tag_to_remove_name)
        f.remove_from_object(project_id, figure_id, tag_name=tag_to_remove_name, tag_id=tag_id)

        # assigned_tags_names = f.get_assigned_tags_names_by_label_annotation(label_annotation)
        # f.update_review_tags_tab(assigned_tags_names, fields)

        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


@g.my_app.callback("add_to_reference")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def add_to_reference(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.loading", True)
    fields = {"state.loading": False,
              'state.itemToReference': None}
    try:
        project_id, image_id, figure_id = context["projectId"], context["imageId"], context["figureId"]
        if figure_id in g.figures_in_reference:
            raise ValueError(f'Figure with id {figure_id} already in reference set!')

        reference_data = state["itemToReference"]

        # {
        #     'embedding' :[],
        #     'label' :[],
        #     'url' :[],
        #     'bbox' :[]
        # }

        reference_label = reference_data['current_label']
        embedding = g.figures2embeddings[figure_id]

        image_info = g.spawn_api.image.get_info_by_id(image_id)
        image_url = image_info.full_storage_url

        annotations_for_image = f.get_annotation(project_id, image_id)
        label_annotation = annotations_for_image.get_label_by_id(figure_id)
        bbox = f.sly_annotation_to_bbox(label_annotation)

        data_to_add = {'input_data': {
            'embedding': [embedding],
            'label': [reference_label],
            'url': [image_url],
            'bbox': [bbox],
            'figure_id': [figure_id]
        }}

        response = api.task.send_request(g.calculator_session_id, "add_new_embeddings_to_reference",
                                         data=data_to_add,
                                         timeout=999)

        updated_calculator_info = response['embeddings_stats']
        for key, value in updated_calculator_info.items():  # updating AI Recommendations stats
            g.calculator_info[key] = value

        fields['data.calculatorStats'] = g.calculator_info
        g.figures_in_reference.append(figure_id)

        new_image_urls = response['new_images_url']  # updating label urls
        g.items_database[reference_label]['url'] = new_image_urls + g.items_database[reference_label]['url']

        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e
