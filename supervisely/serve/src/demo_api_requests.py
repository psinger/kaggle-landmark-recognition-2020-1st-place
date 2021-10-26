import json
import supervisely_lib as sly


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 10495

    # get embedding for test image
    response = api.task.send_request(task_id, "inference", data={
        'input_data': [{
            'index': 0,
            'url': 'https://supervise.ly/_nuxt/img/video.00ef1f3.jpg',
            'bbox': [212, 203, 257, 594]
        }]

    }, timeout=60)
    print("APP returns data:")
    print(json.loads(response))


if __name__ == "__main__":
    main()
