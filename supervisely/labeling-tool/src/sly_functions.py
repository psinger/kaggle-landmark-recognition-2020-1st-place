import re


def camel_to_snake(string_to_process):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', string_to_process).lower()


def process_info_for_showing(info_data):
    processed_info = {}

    for key, value in info_data.items():
        processed_info[camel_to_snake(key).title()] = value

    return processed_info


def remove_keys_from_dict(keys_to_remove, data):
    for key in keys_to_remove:
        data.pop(key, None)


