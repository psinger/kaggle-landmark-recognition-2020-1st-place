import supervisely as sly
import sly_globals as g


def init(data, state):
    # state["applyTo"] = "image"  # "object"  # "image" #@TODO: for debug
    # state["assignMode"] = "append"

    state["topn"] = 5
    state["pad"] = 10
    state["addEveryAssignedToReference"] = True
    state["tagPerImage"] = True

    state["selectedDescriptionsToShow"] = []
