import sly_globals as g
import supervisely_lib as sly


def show(results, fields):
    pred_tags_names = []
    pred_tags = []
    for item in results:
        tag_meta = g.model_meta.tag_metas.get(item["class"])
        if tag_meta is None:
            raise KeyError(f"Predicted tag with name \"{item['class']}\" not found in model meta")
        tag_meta_json = tag_meta.to_json()
        tag_meta_json["score"] = "{:.3f}".format(item["score"])
        pred_tags.append(tag_meta_json)
        pred_tags_names.append(tag_meta.name)

    fields.update({
        "data.predTags": pred_tags,
        "data.predTagsNames": pred_tags_names,
        "state.activeNamesPred": pred_tags_names
    })


def hide(fields):
    fields["data.predTags"] = None
