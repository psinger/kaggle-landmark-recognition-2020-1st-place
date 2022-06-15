import supervisely as sly
import sly_globals as g
import input_project
import load_checkpoint
import embeddings
import infer_project
import grid_gallery


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None
    input_project.init(data, state)
    load_checkpoint.init(data, state)
    embeddings.init(data, state)
    infer_project.init(data, state)
    grid_gallery.init(data, state)



@g.my_app.callback("show_info")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def show_info(api: sly.Api, task_id, context, state, app_logger):
    print(state)
    pass


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}

    # if restart_from_step <= 2:
    #     train_val_split.init(g.project_info, g.project_meta, data, state)
    # if restart_from_step <= 3:
    #     if restart_from_step == 3:
    #         tags.restart(data, state)
    #     else:
    #         tags.init(data, state)

    fields = [
        {"field": "data", "payload": data, "append": True, "recursive": False},
        {"field": "state", "payload": state, "append": True, "recursive": False},
        {"field": "state.restartFrom", "payload": None},
        {"field": f"state.collapsed{restart_from_step}", "payload": False},
        {"field": f"state.disabled{restart_from_step}", "payload": False},
        {"field": "state.activeStep", "payload": restart_from_step},
    ]
    g.api.app.set_fields(g.task_id, fields)
    g.api.app.set_field(task_id, "data.scrollIntoView", f"step{restart_from_step}")
