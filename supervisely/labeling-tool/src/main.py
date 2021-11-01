import supervisely_lib as sly
import sly_globals as g
import cache
import prediction
import ui
import review_tab
import tag_utils

import connector_first_step
import connector_second_step

# to register callbacks

import iterate_objects

import sly_functions as f


@g.my_app.callback("manual_selected_image_changed")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def image_changed(api: sly.Api, task_id, context, state, app_logger):
    fields = {}
    try:
        nn_session = state["nnId"]
        project_id = context["projectId"]
        if nn_session is None:
            return
        if state["applyTo"] == "object":
            return
        api.task.set_field(task_id, "state.loading", True)
        fields["state.loading"] = False

        api.task.set_field(task_id, "state.loading", True)
        image_id = context["imageId"]
        results = tag_utils.classify_image(nn_session, image_id, state["topn"])

        prediction.show(results, fields)
        review_tab.refresh_image(project_id, image_id, fields)
        api.task.set_fields_from_dict(task_id, fields)

    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


@g.my_app.callback("manual_selected_figure_changed")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def figure_changed(api: sly.Api, task_id, context, state, app_logger):
    fields = {}
    api.task.set_field(task_id, "state.loading", True)

    try:
        sly.logger.debug("Context", extra={"context": context})

        project_id = context["projectId"]
        nn_session = state["nnId"]
        similarity_session = state["similarityId"]

        figure_id = context["figureId"]

        image_id = context["imageId"]
        figure_id = context["figureId"]

        ann = cache.get_annotation(project_id, image_id)

        results = f.calculate_nearest_images(nn_session, image_id, state["topn"], ann, figure_id, state["pad"])

        prediction.show(results, fields)
        review_tab.refresh_figure(project_id, figure_id, fields)
        api.task.set_fields_from_dict(task_id, fields)

    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e


def main():
    g.my_app.compile_template(g.root_source_dir)

    data = {}
    state = {}
    ui.init(data, state)
    connector_first_step.init_fields(data=data, state=state)
    connector_second_step.init_fields(data=data, state=state)

    g.my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
