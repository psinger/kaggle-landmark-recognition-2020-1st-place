import time

import supervisely_lib as sly
import sly_globals as g
import sly_functions as f


def init_fields(data, state):
    state["selectedDatabaseItem"] = None

    state["selectedRowLabel"] = None
    data['items_database'] = []


def get_urls_by_label(selected_label):
    urls = []
    for row in g.items_database:
        if row['label'] == selected_label:
            urls.append({'preview': f.get_resized_image(row['url'], g.items_preview_size)})
    return urls


@g.my_app.callback("show_database_row")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def show_database_row(api: sly.Api, task_id, context, state, app_logger):
    fields = {}

    api.task.set_field(task_id, "state.loading", True)
    fields["state.loading"] = False

    row_label = None
    for i in range(10):
        row_label = g.api.app.get_field(g.task_id, 'state.selectedRowLabel')
        time.sleep(0.01)

    label_urls = get_urls_by_label(row_label)

    selected_database_item = {
        'url': label_urls,
        'current_label': row_label,
        'assignDisabled': True,
        'referenceDisabled': True
    }

    project_id, image_id, figure_id = context["projectId"], context["imageId"], context["figureId"]

    if figure_id:
        annotations_for_image = f.get_annotation(project_id, image_id)
        label_annotation = annotations_for_image.get_label_by_id(figure_id)
        assigned_tags = f.get_assigned_tags_names_by_label_annotation(label_annotation)

        if selected_database_item.get('current_label', '') not in assigned_tags:
            selected_database_item['assignDisabled'] = False

        fields["state.selectedFigureId"] = figure_id

    fields["state.selectedDatabaseItem"] = selected_database_item
    api.task.set_fields_from_dict(task_id, fields)




