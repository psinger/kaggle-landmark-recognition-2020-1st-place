import json
import supervisely as sly


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 12483

    # get embedding for test image
    response = api.task.send_request(task_id, "inference", data={
        'input_data': [{
            'index': i,
            'url': 'https://supervise.ly/_nuxt/img/video.00ef1f3.jpg',
            'bbox': [212, 203, 257, 594]
        } for i in range(100)]

    }, timeout=600)
    # print("APP returns data:")
    # print(json.loads(response))

    response = api.task.send_request(task_id, "get_info", data={}, timeout=60)
    print("APP returns data:")
    print(json.loads(response))


if __name__ == "__main__":
    main()
