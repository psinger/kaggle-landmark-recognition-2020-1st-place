import json
import os
import urllib.request

import cv2
import numpy as np
import sly_globals as g
import supervisely_lib as sly

import model_functions

import requests

from tqdm import tqdm

from functools import lru_cache


def download_file_by_url(url, local_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc='⏬ downloading model'):
                f.write(chunk)
    return 0


def download_model_and_config():
    if g.selected_weights_type == 'pretrained':
        g.local_weights_path = os.path.join(g.my_app.data_dir, f'{g.selected_model}.pth')
        if not os.path.exists(g.local_weights_path):
            download_file_by_url(g.remote_weights_path, g.local_weights_path)

    else:
        remote_model_weights_name = g.remote_weights_path.split('/')[-1]
        g.local_weights_path = os.path.join(g.my_app.data_dir, remote_model_weights_name)
        g.api.file.download(g.team_id, g.remote_weights_path, g.local_weights_path)


def inference_one_batch(data):
    return model_functions.calculate_embeddings_for_nps_batch(data)


def batch_inference(data):
    """
    :param data: np.arrays: [[img, img, img, img]]
    :return: embedding for every image [[emb1, emb2, emb3, emb4]]
    """
    batches = np.array_split(data, g.batch_size, axis=0)
    batches = [batch for batch in batches if batch.size > 0]

    embeddings = []
    for current_batch in tqdm(batches, desc='✨ calculating embeddings'):
        temp_embedding = inference_one_batch(current_batch)
        embeddings.extend(temp_embedding)
    return embeddings


def crop_images(data):
    """
     FOR EACH image in data
     crop image if bbox is not None
    """
    for row in data:
        if row['cached_image'] is not None and row['bbox']:
            top, left, height, width = row['bbox'][0], row['bbox'][1], row['bbox'][2], row['bbox'][3]
            crop = row['cached_image'][top:top + height, left:left + width]

            if crop.shape[0] > 0 and crop.shape[1] > 0:
                row['cached_image'] = crop
            else:
                g.logger.warn(f'image not cropped: {row["url"]}\n'
                              f'reason: {crop.shape}')
                row['cached_image'] = None


@lru_cache(maxsize=32)
def url_to_image(url):
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image


def cache_images(data):
    """
     FOR EACH url in data
     download image to RAM by url
    """
    for row in tqdm(data, desc='⏬ downloading images'):
        try:
            image_in_memory = url_to_image(row['url'])
            row['cached_image'] = image_in_memory
        except Exception as ex:
            g.logger.warn(f'image not downloaded: {row["url"]}\n'
                          f'reason: {ex}')

            row['cached_image'] = None


@g.my_app.callback("inference")
@sly.timeit
def inference(api: sly.Api, task_id, context, state, app_logger):
    data_to_process = list(state['input_data'])

    cache_images(data_to_process)
    crop_images(data_to_process)

    filtered_data = [row for row in data_to_process if row['cached_image'] is not None]
    images_to_process = np.asarray([row['cached_image'] for row in filtered_data])

    embeddings = batch_inference(images_to_process)

    output_data = json.dumps(str([{'index': row['index'],
                                   'embedding': list(embeddings[index])} for index, row in enumerate(filtered_data)]))

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=output_data)


@g.my_app.callback("get_info")
@sly.timeit
def get_info(api: sly.Api, task_id, context, state, app_logger):
    if g.selected_weights_type == 'pretrained':
        output_data = {'modelType': g.selected_weights_type}
        output_data.update(g.model_info)

    else:
        output_data = {
            'modelType': g.selected_weights_type,
            'modelName': g.remote_weights_path.split('/')[-1]
        }

    output_data = json.dumps(str(output_data))

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=output_data)


def main():
    sly.logger.info("Script arguments", extra={

        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    model_functions.initialize_network()
    download_model_and_config()
    model_functions.load_weights(g.local_weights_path)

    sly.logger.info("🟩 Model has been successfully deployed")
    sly.logger.debug("Script arguments", extra={
        "Remote weights": g.remote_weights_path,
        "Local weights": g.local_weights_path,
        "device": g.device
    })

    g.my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
