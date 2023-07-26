import supervisely as sly

import sly_globals as g
import tags_tab

import ui
import tag_utils

# to register callbacks

import sly_functions as f


@g.my_app.callback("manual_selected_image_changed")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def image_changed(api: sly.Api, task_id, context, state, app_logger):
    project_id, image_id = context["projectId"], context["imageId"]
    annotation = f.get_annotation(project_id, image_id)

    g.annotated_figures_count = f.get_tagged_objects_count_on_frame(annotation)
    g.figures_on_frame_count = len(annotation.labels)

    api.task.set_field(task_id, 'state.allFiguresCount', g.figures_on_frame_count)
    api.task.set_field(task_id, 'state.annotatedFiguresCount', g.annotated_figures_count)



#
# @g.my_app.callback("manual_selected_image_changed")
# @sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
#     fields = {}
#     try:
#         nn_session = state["nnId"]
#         project_id = context["projectId"]
#         if nn_session is None:
#             return
#         if state["applyTo"] == "object":
#             return
#         api.task.set_field(task_id, "state.loading", True)
#         fields["state.loading"] = False
#
#         api.task.set_field(task_id, "state.loading", True)
#         image_id = context["imageId"]
#         results = tag_utils.classify_image(nn_session, image_id, state["topn"])
#
#         prediction.show(results, fields)
#         review_tab.refresh_image(project_id, image_id, fields)
#         api.task.set_fields_from_dict(task_id, fields)
#
#     except Exception as e:
#         api.task.set_fields_from_dict(task_id, fields)
#         raise e


def _select_object(api: sly.Api, task_id, context, state, iterate_func):
    fields = {
        "state.loading": False
    }

    try:
        sly.logger.debug("Context", extra={"context": context})

        project_id, image_id, figure_id = context["projectId"], context["imageId"], context["figureId"]

        ann_tool_session = context["sessionId"]
        ann = f.get_annotation(project_id, image_id)
        if len(ann.labels) == 0:
            g.my_app.show_modal_window("There are no figures on image")
            api.task.set_fields_from_dict(task_id, fields)
        else:
            iter_figure_id = iterate_func(ann, figure_id)
            if iter_figure_id is not None:
                context['figureId'] = iter_figure_id
                api.img_ann_tool.set_figure(ann_tool_session, iter_figure_id)
                api.img_ann_tool.zoom_to_figure(ann_tool_session, iter_figure_id, zoom_factor=3)
                manual_selected_figure_changed(api, task_id, context, state, g.my_app.logger)
            else:
                g.my_app.show_modal_window("All figures are visited.")
                api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


@g.my_app.callback("prev_object")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def prev_object(api: sly.Api, task_id, context, state, app_logger):
    _select_object(api, task_id, context, state, tag_utils.get_next)


@g.my_app.callback("next_object")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def next_object(api: sly.Api, task_id, context, state, app_logger):
    _select_object(api, task_id, context, state, tag_utils.get_prev)


@g.my_app.callback("manual_selected_figure_changed")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def manual_selected_figure_changed(api: sly.Api, task_id, context, state, app_logger):
    fields = {}

    if context.get("figureId", None) is None or g.nn_session_id is None or g.calculator_session_id is None:
        fields["state.selectedFigureId"] = None
        f.set_buttons(assign_disabled=True, reference_disabled=True, card_name='lastAssignedTag', fields=fields)
        f.set_buttons(assign_disabled=True, reference_disabled=True, card_name='selectedDatabaseItem', fields=fields)

        api.task.set_fields_from_dict(task_id, fields)
        return 2

    api.task.set_field(task_id, "state.loading", True)
    fields["state.loading"] = False

    try:
        sly.logger.debug("Context", extra={"context": context})

        project_id, image_id, figure_id = context["projectId"], context["imageId"], context["figureId"]
        fields["state.selectedFigureId"] = figure_id

        annotations_for_image = f.get_annotation(project_id, image_id)
        label_annotation = annotations_for_image.get_label_by_id(figure_id)

        nearest_labels = f.calculate_nearest_labels(images_ids=[image_id],
                                                    annotations=[label_annotation],
                                                    figures_ids=[figure_id], top_n=state.get('topn', 5), padding=0)

        if state['copyingMode']:
            state["selectedFigureId"] = figure_id
            tags_tab.assign_tag_to_figure(api, task_id, context, state, app_logger)
        else:
            f.upload_data_to_tabs(nearest_labels, label_annotation, fields, state)

        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


def main():
    g.my_app.compile_template(g.root_source_dir)

    data = {}
    state = {}
    ui.init(data, state)

    g.my_app.run(data=data, state=state)


#  @TODO: patches padding

if __name__ == "__main__":
    sly.main_wrapper("main", main)
