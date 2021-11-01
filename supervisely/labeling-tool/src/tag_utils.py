import supervisely_lib as sly
import sly_globals as g
import cache


def get_prev(ann: sly.Annotation, cur_figure_id):
    if len(ann.labels) == 0:
        return None
    if cur_figure_id is None:
        # nothing is selected, return first figure
        return ann.labels[0].geometry.sly_id
    else:
        for idx, label in enumerate(ann.labels):
            if cur_figure_id == label.geometry.sly_id:
                if idx == 0:
                    # all labels are visited
                    return None
                else:
                    return ann.labels[idx - 1].geometry.sly_id


def get_next(ann: sly.Annotation, cur_figure_id):
    if len(ann.labels) == 0:
        return None
    if cur_figure_id is None:
        # nothing is selected, return first figure
        return ann.labels[0].geometry.sly_id
    else:
        for idx, label in enumerate(ann.labels):
            if cur_figure_id == label.geometry.sly_id:
                if idx == len(ann.labels) - 1:
                    # all labels are visited
                    return None
                else:
                    return ann.labels[idx + 1].geometry.sly_id





def classify_image(nn_id, image_id, topn):
    predictions = g.api.task.send_request(nn_id, "inference_image_id",
                                          data={
                                              "image_id": image_id,
                                              "topn": topn,
                                          })
    return predictions


def assign_to_object(project_id, figure_id, class_name):
    tag_meta = g.model_meta.tag_metas.get(class_name)
    _assign_tag_to_object(project_id, figure_id, tag_meta)


def _get_or_create_tag_meta(project_id, tag_meta):
    project_meta = cache.get_meta(project_id)
    project_tag_meta: sly.TagMeta = project_meta.get_tag_meta(tag_meta.name)
    if project_tag_meta is None:
        project_meta = project_meta.add_tag_meta(tag_meta)
        cache.update_project_meta(project_id, project_meta)
        project_meta = cache.get_meta(project_id)
        project_tag_meta = project_meta.get_tag_meta(tag_meta.name)
    return project_tag_meta


def _assign_tag_to_object(project_id, figure_id, tag_meta):
    project_tag_meta: sly.TagMeta = _get_or_create_tag_meta(project_id, tag_meta)
    g.api.advanced.add_tag_to_object(project_tag_meta.sly_id, figure_id)


def remove_from_object(project_id, figure_id, tag_name, tag_id):
    project_meta = cache.get_meta(project_id)
    project_tag_meta: sly.TagMeta = project_meta.get_tag_meta(tag_name)
    if project_tag_meta is None:
        raise RuntimeError(f"Tag {tag_name} not found in project meta")
    g.api.advanced.remove_tag_from_object(project_tag_meta.sly_id, figure_id, tag_id)


def assign_to_image(project_id, image_id, class_name):
    tag_meta = g.model_meta.tag_metas.get(class_name)
    _assign_tag_to_image(project_id, image_id, tag_meta)


def _assign_tag_to_image(project_id, image_id, tag_meta):
    project_tag_meta: sly.TagMeta = _get_or_create_tag_meta(project_id, tag_meta)
    g.api.image.add_tag(image_id, project_tag_meta.sly_id)


def remove_from_image(project_id, image_id, tag_name, tag_id):
    project_meta = cache.get_meta(project_id)
    project_tag_meta: sly.TagMeta = project_meta.get_tag_meta(tag_name)
    if project_tag_meta is None:
        raise RuntimeError(f"Tag {tag_name} not found in project meta")
    g.api.advanced.remove_tag_from_image(project_tag_meta.sly_id, image_id, tag_id)
