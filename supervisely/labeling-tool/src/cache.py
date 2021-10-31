import os
import supervisely_lib as sly
import globals as g

project2meta = {}  # project_id -> project_meta
image2info = {}
image2ann = {}  # image_id -> annotation


def get_image_path(image_id):
    info = get_image_info(image_id)
    local_path = os.path.join(g.cache_path, f"{info.id}{sly.fs.get_file_name_with_ext(info.name)}")
    if not sly.fs.file_exists(local_path):
        g.api.image.download_path(image_id, local_path)
    return local_path


def get_meta(project_id, optimize=True):
    meta = None
    if project_id not in project2meta or optimize is False:
        meta_json = g.api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)
        project2meta[project_id] = meta
    else:
        meta = project2meta[project_id]
    return meta


def get_annotation(project_id, image_id, optimize=True):
    ann = None
    if image_id not in image2ann or optimize is False:
        ann_json = g.api.annotation.download(image_id).annotation
        ann = sly.Annotation.from_json(ann_json, get_meta(project_id))
        image2ann[image_id] = ann
    else:
        ann = image2ann[image_id]
    return ann


def get_image_info(image_id):
    info = None
    if image_id not in image2info:
        info = g.api.image.get_info_by_id(image_id)
        image2info[image_id] = info
    else:
        info = image2info[image_id]
    return info


def clear():
    project2meta.clear()
    #image2info.clear()
    image2ann.clear()


def update_project_meta(project_id, project_meta: sly.ProjectMeta):
    g.api.project.update_meta(project_id, project_meta.to_json())
    get_meta(project_id, optimize=False)