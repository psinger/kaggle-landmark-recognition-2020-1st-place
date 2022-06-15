import math

import supervisely as sly


class SlyProgress:
    def __init__(self, api, task_id, pbar_element_name):
        self.api = api
        self.task_id = task_id
        self.pbar_element_name = pbar_element_name
        self.pbar = None

    def refresh_params(self, desc, total, is_size=False):
        self.pbar = sly.Progress(desc, total, is_size=is_size)
        # if total > 0:
        self.refresh_progress()
        # self.reset_params()

    def refresh_progress(self):
        curr_step = math.floor(self.pbar.current * 100 /
                   self.pbar.total) if self.pbar.total != 0 else 0

        fields = [
            {"field": f"data.{self.pbar_element_name}", "payload": curr_step},
            {"field": f"data.{self.pbar_element_name}Message", "payload": self.pbar.message},
            {"field": f"data.{self.pbar_element_name}Current", "payload": self.pbar.current_label},
            {"field": f"data.{self.pbar_element_name}Total", "payload": self.pbar.total_label},
            {"field": f"data.{self.pbar_element_name}Percent", "payload":
                curr_step},
        ]
        self.api.task.set_fields(self.task_id, fields)

    def reset_params(self):
        fields = [
            {"field": f"data.{self.pbar_element_name}", "payload": None},
            {"field": f"data.{self.pbar_element_name}Message", "payload": None},
            {"field": f"data.{self.pbar_element_name}Current", "payload": None},
            {"field": f"data.{self.pbar_element_name}Total", "payload": None},
            {"field": f"data.{self.pbar_element_name}Percent", "payload": None},
        ]
        self.api.task.set_fields(self.task_id, fields)

    def next_step(self):
        self.pbar.iter_done_report()
        self.refresh_progress()

    def upload_monitor(self, monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)

        self.refresh_progress()

    def update_progress(self, count, api: sly.Api, task_id, progress: sly.Progress):
        # hack slight inaccuracies in size convertion
        count = min(count, progress.total - progress.current)
        progress.iters_done(count)
        if progress.need_report():
            progress.report_progress()
            self.refresh_progress()

    def set_progress(self, current, api: sly.Api, task_id, progress: sly.Progress):
        old_value = progress.current
        delta = current - old_value
        self.update_progress(delta, api, task_id, progress)