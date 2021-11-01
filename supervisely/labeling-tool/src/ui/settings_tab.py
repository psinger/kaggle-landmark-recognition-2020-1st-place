import supervisely_lib as sly
import globals as g
import cache


def init(data, state):
    state["applyTo"] = "image" #"object"  # "image" #@TODO: for debug
    state["assignMode"] = "append"
    state["topn"] = 5
    state["pad"] = 10
    state["addEveryPatchToReference"] = None


@g.my_app.callback("clear_cache")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def clear_cache(api: sly.Api, task_id, context, state, app_logger):
    cache.clear()
