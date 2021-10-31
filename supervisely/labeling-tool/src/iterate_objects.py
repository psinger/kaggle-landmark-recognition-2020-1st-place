import supervisely_lib as sly
import globals as g
import cache
import tag_utils
import prediction
import review_tab


def _select_object(api: sly.Api, task_id, context, state, iterate_func):
    try:
        fields = {
            "state.loading": False
        }
        sly.logger.debug("Context", extra={"context": context})
        project_id = context["projectId"]
        image_id = context["imageId"]
        figure_id = context["figureId"]
        ann_tool_session = context["sessionId"]
        ann = cache.get_annotation(project_id, image_id)
        if len(ann.labels) == 0:
            g.my_app.show_modal_window("There are no figures on image")
        else:
            iter_figure_id = iterate_func(ann, figure_id)
            if iter_figure_id is not None:
                api.img_ann_tool.set_figure(ann_tool_session, iter_figure_id)
                api.img_ann_tool.zoom_to_figure(ann_tool_session, iter_figure_id, zoom_factor=2)
                results = tag_utils.classify(state["nnId"], image_id, state["topn"], ann, iter_figure_id, state["pad"])
                prediction.show(results, fields)
                review_tab.refresh_figure(project_id, iter_figure_id, fields)
            else:
                g.my_app.show_modal_window("All figures are visited. Select another figure or clear selection to iterate over objects again")
                prediction.hide(fields)
                review_tab.reset(fields)
        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


@g.my_app.callback("prev_object")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def prev_object(api: sly.Api, task_id, context, state, app_logger):
    _select_object(api, task_id, context, state, tag_utils.get_prev)


@g.my_app.callback("next_object")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def next_object(api: sly.Api, task_id, context, state, app_logger):
    _select_object(api, task_id, context, state, tag_utils.get_next)