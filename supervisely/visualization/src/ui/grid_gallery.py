import sly_globals as g
import supervisely_lib as sly
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
    max_page_num = (len(g.gallery_data) // state['rows']) + 1
    fields = [
        {"field": "data.cols", "payload": state['cols']},
        {"field": "data.rows", "payload": state['rows']},
        {"field": "state.galleryInitialized", "payload": True},
        {"field": "state.galleryPage", "payload": 1},
        {"field": "state.galleryMaxPage", "payload": max_page_num}
    ]
    g.api.app.set_fields(g.task_id, fields)
    time.sleep(1)
    image_urls = np.asarray(g.gallery_data)

    image_gallery.set_data(title='11', image_url=image_urls[:state['rows'], :state['cols'] + 1], ann=None)
    image_gallery.update()


@g.my_app.callback("update_gallery")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def set_checkpoints_path(api: sly.Api, task_id, context, state, app_logger):
    o_image_url = "http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/w/9/La/ECAY6HZjZdEpvqVmRbJBRHQog1BH2iD3KlTGDbA9WO03cpnfM4zALOnUSMCaAPwhm3ZkZmMkHVBCqrY4TgMYlzdw12bwjtRV0sqLE5iJplulUSMFsvUifvBQ4xpA.jpg"
    p_image_url = "http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/C/Y/Hq/DFbnet16ohj7d1RdB7fH8rEcJl8NaQp5NvhgYez0P8j46iU4662WeVMm4n89AIWatgeb6Atd4SYDSuqZIJecEZ48CeMJBvtuWYrn6M80JK3UWxZGyeRegNoWbPZb.jpg"
    image_url = [o_image_url, p_image_url]
    image_gallery.set_data(title='11', image_url=image_url, ann=None)
    image_gallery.update()


@g.my_app.callback("next_page")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def next_page(api: sly.Api, task_id, context, state, app_logger):
    current_page = state['galleryPage']
    is_first_page = True if current_page + 1 == 1 else False
    is_last_page = True if current_page + 1 == state['galleryMaxPage'] else False
    fields = [
        {"field": "state.galleryPage", "payload": current_page + 1},
        {"field": "state.galleryIsFirstPage", "payload": is_first_page},
        {"field": "state.galleryIsLastPage", "payload": is_last_page}
    ]
    g.api.app.set_fields(g.task_id, fields)
    image_urls = np.asarray(g.gallery_data)
    rows = state['rows']
    image_gallery.set_data(title='11',
                           image_url=image_urls[
                                     current_page * rows:(current_page + 1) * rows,
                                     :state['cols'] + 1],
                           ann=None)
    image_gallery.update()


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
        {"field": "state.galleryIsLastPage", "payload": is_last_page}
    ]
    g.api.app.set_fields(g.task_id, fields)
    image_urls = np.asarray(g.gallery_data)
    rows = state['rows']
    image_gallery.set_data(title='11',
                           image_url=image_urls[
                                     (current_page - 1) * rows:
                                     current_page * rows,
                                     :state['cols'] + 1],
                           ann=None)

    image_gallery.update()


v_model = 'data.Gallery'
image_gallery = CompareGallery(g.task_id, g.api, v_model,
                               g.project_meta)
