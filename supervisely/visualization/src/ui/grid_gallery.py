import sly_globals as g
import supervisely as sly
from sly_visualization_progress import init_progress
from grid_gallery_widget import CompareGallery
import time
import numpy as np


def init(data, state):
    state["collapsed5"] = True
    state["disabled5"] = True
    state["modelLoading"] = False
    init_progress(5, data)

    state["weightsPath"] = ""
    state['galleryInitialized'] = False
    data["done5"] = False

    data['rows'] = 1
    data['cols'] = 1
    data['Gallery'] = {}
    state['galleryPage'] = None
    state['galleryMaxPage'] = None
    state['galleryIsFirstPage'] = True
    state['galleryIsLastPage'] = False


def restart(data, state):
    data["done5"] = False


@g.my_app.callback("set_grid_size")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def set_grid_size(api: sly.Api, task_id, context, state, app_logger):
    image_gallery.update_grid_size(state['rows'], state['cols'])
    max_page_num = (len(g.gallery_data['urls']) // state['rows']) + 1
    fields = [
        {"field": "data.cols", "payload": state['cols']},
        {"field": "data.rows", "payload": state['rows']},
        {"field": "state.galleryInitialized", "payload": True},
        {"field": "state.galleryPage", "payload": 1},
        {"field": "state.galleryIsFirstPage", "payload": True},
        {"field": "state.galleryIsLastPage", "payload": False},
        {"field": "state.galleryMaxPage", "payload": max_page_num}
    ]
    g.api.app.set_fields(g.task_id, fields)
    time.sleep(1)
    image_urls = np.asarray(g.gallery_data['urls'])
    image_labels = np.asarray(g.gallery_data['labels'])
    image_confidences = np.asarray(g.gallery_data['confidences'])

    image_gallery.set_data(title='11', image_url=image_urls[:state['rows'], :state['cols'] + 1], ann=None,
                           additional_data={'labels': image_labels, 'confidences': image_confidences})
    image_gallery.update()


def update_gallery_by_page(current_page, state):
    current_page = current_page - 1

    image_urls = np.asarray(g.gallery_data['urls'])
    image_labels = np.asarray(g.gallery_data['labels'])
    image_confidences = np.asarray(g.gallery_data['confidences'])
    rows = state['rows']

    curr_image_urls = image_urls[
                  current_page * rows:(current_page + 1) * rows,
                  :state['cols'] + 1]

    image_gallery.update_grid_size(curr_image_urls.shape[0], state['cols'])

    image_gallery.set_data(
        title='11',
        image_url=curr_image_urls,
        ann=None,
        additional_data={
            'labels': image_labels[
                      current_page * rows:(current_page + 1) * rows,
                      :state['cols'] + 1],
            'confidences': image_confidences[
                           current_page * rows:(current_page + 1) * rows]
        }
    )
    image_gallery.update()


@g.my_app.callback("next_page")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def next_page(api: sly.Api, task_id, context, state, app_logger):
    current_page = state['galleryPage']
    is_first_page = True if current_page + 1 == 1 else False
    is_last_page = True if current_page + 1 == state['galleryMaxPage'] else False
    fields = [
        {"field": "state.galleryIsFirstPage", "payload": is_first_page},
        {"field": "state.galleryIsLastPage", "payload": is_last_page},
        {"field": "state.galleryPage", "payload": current_page + 1}
    ]

    g.api.app.set_fields(g.task_id, fields)

    update_gallery_by_page(current_page + 1, state)


@g.my_app.callback("previous_page")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def previous_page(api: sly.Api, task_id, context, state, app_logger):
    current_page = state['galleryPage']
    is_first_page = True if current_page - 1 == 1 else False
    is_last_page = True if current_page - 1 == state['galleryMaxPage'] else False
    fields = [
        {"field": "state.galleryPage", "payload": current_page - 1},
        {"field": "state.galleryIsFirstPage", "payload": is_first_page},
        {"field": "state.galleryIsLastPage", "payload": is_last_page},
    ]
    g.api.app.set_fields(g.task_id, fields)

    update_gallery_by_page(current_page - 1, state)


v_model = 'data.Gallery'
image_gallery = CompareGallery(g.task_id, g.api, v_model, g.project_meta)
