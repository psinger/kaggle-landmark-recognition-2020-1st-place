from collections import defaultdict
import supervisely as sly

import sly_globals as g
import info_tab
import tags_tab
import settings_tab
import catalog_tab


import connector_first_step
import connector_second_step


def init(data, state):
    state["loading"] = False

    state["activeStep"] = 1

    # state["connecting"] = False
    # state["nextLoading"] = False

    data["ownerId"] = g.owner_id
    data["teamId"] = g.team_id

    state["nnId"] = None  # task id of deployed model
    state["calculatorId"] = None  # task id of deployed model

    state["tabName"] = "info"

    info_tab.init(data, state)
    tags_tab.init(data, state)
    settings_tab.init(data, state)

    connector_first_step.init_fields(data=data, state=state)
    connector_second_step.init_fields(data=data, state=state)
    catalog_tab.init_fields(data=data, state=state)

    state["assignLoading"] = False
    state["assignName"] = None

    state["predictLoading"] = False


