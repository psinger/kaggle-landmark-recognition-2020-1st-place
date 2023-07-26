import time

import supervisely as sly
import sly_globals as g
import sly_functions as f


def init_fields(data, state):
    state["selectedDatabaseItem"] = None

    state["selectedRowLabel"] = None
    data['itemsDatabase'] = []


@g.my_app.callback("show_database_row")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def show_database_row(api: sly.Api, task_id, context, state, app_logger):
    fields = {}

    api.task.set_field(task_id, "state.loading", True)
    fields["state.loading"] = False

    row_label = state['selectedRowLabel']
    # for i in range(10):
    #     row_label = g.api.app.get_field(g.task_id, 'state.selectedRowLabel')
    #     time.sleep(1e-3)

    label_urls = f.get_urls_by_label(row_label)

    selected_database_item = {
        'url': label_urls,
        'current_label': row_label,
        'assignDisabled': True,
        'referenceDisabled': True,
        'description': f.get_item_description_by_label(row_label)
    }

    project_id, image_id, figure_id = context["projectId"], context["imageId"], context["figureId"]

    fields["state.selectedDatabaseItem"] = selected_database_item

    if figure_id:
        annotations_for_image = f.get_annotation(project_id, image_id)
        label_annotation = annotations_for_image.get_label_by_id(figure_id)
        assigned_tags = f.get_assigned_tags_names_by_label_annotation(label_annotation)
        fields["state.selectedFigureId"] = figure_id
        api.task.set_fields_from_dict(task_id, fields)

        f.update_card_buttons('selectedDatabaseItem', assigned_tags, fields, state)  # Database tab
    else:
        f.set_buttons(assign_disabled=True, reference_disabled=True, card_name='selectedDatabaseItem', fields=fields)

    api.task.set_fields_from_dict(task_id, fields)




