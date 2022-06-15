import supervisely as sly

import json

import sly_globals as g
import functions as f


@g.my_app.callback("calculate_embeddings_for_project")
@sly.timeit
def calculate_embeddings_for_project(api: sly.Api, task_id, context, state, app_logger):
    datasets_list = g.api.dataset.get_list(g.project_id)
    images_count = f.get_images_count_in_project(project_id=g.project_id)
    progress = sly.Progress("processing images:", images_count)
    for current_dataset in datasets_list:
        packed_data = {}
        images_info = api.image.get_list(current_dataset.id)

        images_ids = f.split_list_to_batches([current_image_info.id for current_image_info in images_info])
        images_urls = f.split_list_to_batches([current_image_info.full_storage_url
                                               for current_image_info in images_info])

        for ids_batch, urls_batch in zip(images_ids, images_urls):
            ann_infos = api.annotation.download_batch(current_dataset.id, json.loads(str(ids_batch.tolist())))
            ann_objects = f.jsons_to_annotations(ann_infos)

            data_for_each_image = f.get_data_for_each_image(ann_objects)
            batch_for_inference = f.generate_batch_for_inference(urls_batch, data_for_each_image)
            embeddings_by_indexes = f.inference_batch(batch_for_inference)

            f.pack_data(packed_data, batch_for_inference, embeddings_by_indexes)

            progress.iters_done_report(g.batch_size if len(ids_batch) % g.batch_size == 0
                                       else len(ids_batch) % g.batch_size)

        f.write_packed_data(current_dataset.id, packed_data)

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "context.sessionId": g.session_id
    })

    f.check_model_connection()
    g.my_app.run(initial_events=[{"command": "calculate_embeddings_for_project"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
