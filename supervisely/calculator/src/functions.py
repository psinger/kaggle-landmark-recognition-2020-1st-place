import supervisely_lib as sly
from functools import partial

import sly_globals as g


def get_progress_cb(message, total, is_size=False):
    progress = sly.Progress(message, total, is_size=is_size)
    progress_cb = partial(update_progress, api=g.api, task_id=g.my_app.task_id, progress=progress)
    progress_cb(0)
    return progress_cb


def update_progress(count, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done_report(count)
    _update_progress_ui(api, task_id, progress)


def _update_progress_ui(api: sly.Api, task_id, progress: sly.Progress, stdout_print=False):
    if progress.need_report():
        fields = [
            {"field": "data.progressName", "payload": progress.message},
            {"field": "data.currentProgressLabel", "payload": progress.current_label},
            {"field": "data.totalProgressLabel", "payload": progress.total_label},
            {"field": "data.currentProgress", "payload": progress.current},
            {"field": "data.totalProgress", "payload": progress.total},
        ]
        api.app.set_fields(task_id, fields)
        if stdout_print is True:
            progress.report_if_needed()
