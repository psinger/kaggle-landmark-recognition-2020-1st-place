import supervisely_lib as sly
import globals as g
import cache
import review_tab
import tag_utils


def init(data, state):
    state["activeNamesReview"] = []
    data["reviewTags"] = None
    data["reviewTagsNames"] = None
    state["removeTagName"] = None


def set_data(tags, tags_names, active_names, fields):
    fields.update({
        "state.activeNamesReview": list(set(active_names)),
        "data.reviewTags": tags,
        "data.reviewTagsNames": list(set(tags_names)) if tags_names is not None else tags_names,
    })


def reset(fields):
    set_data(tags=None, tags_names=None, active_names=[], fields=fields)


def _refresh_tags(project_meta: sly.ProjectMeta, tags_collection: sly.TagCollection, fields):
    review_tags = []
    reviewTagsNames = []
    activeNamesReview = []
    for tag in tags_collection:
        tag: sly.Tag
        tag_meta = project_meta.tag_metas.get(tag.meta.name)
        if tag_meta is not None:
            review_tags.append({
                **tag_meta.to_json(),
                "labelerLogin": tag.labeler_login,
                "id": tag.sly_id
            })
            reviewTagsNames.append(tag_meta.name)
            activeNamesReview.append(tag_meta.name)
    set_data(review_tags, reviewTagsNames, activeNamesReview, fields)


def refresh_figure(project_id, figure_id, fields):
    if figure_id is None:
        reset(fields)
    else:
        project_meta = cache.get_meta(project_id)
        object_tags_json = g.api.advanced.get_object_tags(figure_id)
        object_tags = sly.TagCollection.from_json(object_tags_json, project_meta.tag_metas)
        _refresh_tags(project_meta, object_tags, fields)


def refresh_image(project_id, image_id, fields):
    if image_id is None:
        reset(fields)
    else:
        project_meta = cache.get_meta(project_id)
        id_to_tagmeta = project_meta.tag_metas.get_id_mapping()
        image_info = g.api.image.get_info_by_id(image_id)
        img_tags = sly.TagCollection.from_api_response(image_info.tags, project_meta.tag_metas, id_to_tagmeta)
        _refresh_tags(project_meta, img_tags, fields)


@g.my_app.callback("remove_tag")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def remove_tag(api: sly.Api, task_id, context, state, app_logger):
    try:
        api.task.set_field(task_id, "state.loading", True)
        fields = {
            "state.loading": False,
            "state.removeTagName": None,
            "state.removeTagId": None
        }
        project_id = context["projectId"]
        image_id = context["imageId"]
        figure_id = context["figureId"]
        apply_to = state["applyTo"]

        if apply_to == "object":
            tag_utils.remove_from_object(project_id, figure_id, state["removeTagName"], state["removeTagId"])
            review_tab.refresh_figure(project_id, figure_id, fields)
        elif apply_to == "image":
            tag_utils.remove_from_image(project_id, image_id, state["removeTagName"], state["removeTagId"])
            review_tab.refresh_image(project_id, image_id, fields)

        api.task.set_fields_from_dict(task_id, fields)
    except Exception as e:
        api.task.set_fields_from_dict(task_id, fields)
        raise e