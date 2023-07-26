from typing import Union
from supervisely.project.project_meta import ProjectMeta
from supervisely.api.api import Api
from supervisely.annotation.annotation import Annotation
import sly_globals as g
import numpy as np

# html example
# <sly-grid-gallery
#         v-if="data.gallery"
#         :content="data.gallery.content"
#         :options="data.gallery.options">
#     <template v-slot:card-footer="{ annotation }">
#         <div class="mt5" style="text-align: center">
#             <el-tag type="primary">{{annotation.title}}</el-tag>
#         </div>
#     </template>
# </sly-grid-gallery>


class CompareGallery:

    def __init__(self, task_id, api: Api, v_model,
                 project_meta: ProjectMeta, columns=2, rows=1):
        self.cols = columns
        self.rows = rows
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        self._project_meta = project_meta.clone()

        self._options = {
            "enableZoom": False,
            "syncViews": False,
            "showPreview": False,
            "selectable": False,
            "opacity": 0.5,
            "showOpacityInHeader": True,
            "viewHeight": 250,
        }

    def update_project_meta(self, project_meta: ProjectMeta):
        self._project_meta = project_meta.clone()

    def update_grid_size(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def _set_item(self, name, title, image_url, ann: Union[Annotation, dict] = None):
        setattr(self, f"_{name}_title", title)
        setattr(self, f"_{name}_image_url", image_url)
        res_ann = Annotation((1, 1))
        if ann is not None:
            if type(ann) is dict:
                res_ann = Annotation.from_json(ann, self._project_meta)
            else:
                res_ann = ann.clone()
        setattr(self, f"_{name}_ann", res_ann)

    def set_col_data(self, row_idx, image_urls: list, pred_labels=None, confidences=None,
                     annotations: list = None, orig_cls=None):
        matched = False
        for col_idx, url in enumerate(image_urls):
            color = "gray"
            if col_idx == 0:
                if pred_labels[col_idx] == orig_cls:
                    color = "success"
                    matched = True
                else:
                    color = "danger"
            else:
                if pred_labels[col_idx] == orig_cls and not matched:
                    color = "warning"
                    matched = True

            title = {
                "label": pred_labels[col_idx],
                "confidence": str(round(confidences[col_idx], 2)),
                'color': color
            }
            ann = annotations[col_idx] if annotations else None
            self._set_item(f"top_{row_idx}_{col_idx}", title, url, ann)

    def set_data(self, title, image_url, ann: Union[Annotation, dict] = None, additional_data=None):
        orig_labels = None
        pred_labels = None
        confidences = None
        if additional_data:
            if 'labels' in additional_data:
                labels = additional_data['labels']
                orig_labels = labels[:, 0]
                pred_labels = labels[:, 1:]
            if 'confidences' in additional_data:
                confidences = additional_data['confidences']

        orig_urls = image_url[:, 0]
        pred_urls = image_url[:, 1:]
        num_rows = pred_urls.shape[0] if pred_urls.shape[0] < self.rows else self.rows
        for row in range(num_rows):
            title = {
                'label': orig_labels[row] if orig_labels is not None else '',
                'confidence': None,
                'color': "primary"
            }
            self._set_item(f"original_{row}", title, orig_urls[row], ann)
            # image_urls = [p for i in range(self.cols)]
            image_urls = pred_urls[row]
            annotations = None
            self.set_col_data(row, image_urls, pred_labels[row], confidences[row], annotations, orig_labels[row])

    def _get_item_annotation(self, name):
        if hasattr(self, f"_{name}_ann"):
            if getattr(self, f"_{name}_ann"):
                figures = [label.to_json() for label in getattr(self, f"_{name}_ann").labels]
            else:
                figures = None
        return {
            "url": g.api.image.preview_url(getattr(self, f"_{name}_image_url"), height=150),
            "figures": figures,
            "title": getattr(self, f"_{name}_title"),
        }

    def update(self):
        gallery_json = self.to_json()
        self._api.task.set_field(self._task_id, self._v_model, gallery_json)

    def to_json(self, ):
        layout = []
        annotations = {}
        views_bindings = []
        for row_idx in range(self.rows):
            original_name = f"original_{row_idx}"
            row_layout = [original_name]
            annotations[original_name] = self._get_item_annotation(original_name)
            for col_idx in range(self.cols):
                predicted_name = f"top_{row_idx}_{col_idx}"
                row_layout.append(predicted_name)
                annotations[predicted_name] = self._get_item_annotation(predicted_name)
            layout.append(row_layout)
            views_bindings.append(row_layout)
        layout = np.transpose(layout).tolist()
        # views_bindings = np.transpose(views_bindings).tolist()
        return {
            "content": {
                "projectMeta": self._project_meta.to_json(),
                "layout": layout,
                "annotations": annotations
            },
            "options": {
                **self._options,
                "syncViewsBindings": views_bindings
            }
        }