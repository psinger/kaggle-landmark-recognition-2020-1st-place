import supervisely_lib as sly
import sly_globals as g
import tag_utils
import review_tab
import cache
import prediction


def init(data, state):
    state["activeNames"] = []
    state["activeNamesPred"] = []
    data["predTags"] = None
    data["predTagsNames"] = None
    state["previousName"] = None


@g.my_app.callback("assign_to_item")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def assign_to_item(api: sly.Api, task_id, context, state, app_logger):
    try:
        api.task.set_field(task_id, "state.loading", True)
        fields = {
            "state.loading": False
        }

        project_id = context["projectId"]
        image_id = context["imageId"]
        figure_id = context["figureId"]
        class_name = state["assignName"]
        apply_to = state["applyTo"]

        if apply_to == "object":
            tag_utils.assign_to_object(project_id, figure_id, class_name)
            review_tab.refresh_figure(project_id, figure_id, fields)
        elif apply_to == "image":
            tag_utils.assign_to_image(project_id, image_id, class_name)
            review_tab.refresh_image(project_id, image_id, fields)

        fields["state.previousName"] = class_name
        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


@g.my_app.callback("predict")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def predict(api: sly.Api, task_id, context, state, app_logger):
    try:
        api.task.set_field(task_id, "state.loading", True)
        fields = {
            "state.loading": False
        }

        project_id = context["projectId"]
        image_id = context["imageId"]
        figure_id = context["figureId"]
        apply_to = state["applyTo"]
        nn_session = state["nnId"]

        if apply_to == "object":
            ann = cache.get_annotation(project_id, image_id)
            results = tag_utils.classify(nn_session, image_id, state["topn"], ann, figure_id, state["pad"])
            prediction.show(results, fields)
            review_tab.refresh_figure(project_id, figure_id, fields)
        elif apply_to == "image":
            results = tag_utils.classify_image(nn_session, image_id, state["topn"])
            prediction.show(results, fields)
            review_tab.refresh_image(project_id, image_id, fields)

        api.task.set_fields_from_dict(g.task_id, fields)
    except Exception as e:
        prediction.hide(fields)
        api.task.set_fields_from_dict(g.task_id, fields)
        raise e


@g.my_app.callback("mark_unknown")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def mark_unknown(api: sly.Api, task_id, context, state, app_logger):
    try:
        api.task.set_field(task_id, "state.loading", True)
        fields = {
            "state.loading": False
        }

        project_id = context["projectId"]
        image_id = context["imageId"]
        figure_id = context["figureId"]
        apply_to = state["applyTo"]
        if apply_to == "object":
            tag_utils._assign_tag_to_object(project_id, figure_id, g.unknown_tag_meta)
            review_tab.refresh_figure(project_id, figure_id, fields)
        elif apply_to == "image":
            tag_utils._assign_tag_to_image(project_id, image_id, g.unknown_tag_meta)
            review_tab.refresh_image(project_id, image_id, fields)
        fields["state.previousName"] = g.unknown_tag_meta.name
        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


@g.my_app.callback("mark_as_previous")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def mark_as_previous(api: sly.Api, task_id, context, state, app_logger):
    try:
        api.task.set_field(task_id, "state.loading", True)
        fields = {
            "state.loading": False
        }

        project_id = context["projectId"]
        image_id = context["imageId"]
        figure_id = context["figureId"]
        apply_to = state["applyTo"]

        if apply_to == "object":
            tag_utils.assign_to_object(project_id, figure_id, state["previousName"])
            review_tab.refresh_figure(project_id, figure_id, fields)
        elif apply_to == "image":
            tag_utils.assign_to_image(project_id, image_id, state["previousName"])
            review_tab.refresh_image(project_id, image_id, fields)
        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e