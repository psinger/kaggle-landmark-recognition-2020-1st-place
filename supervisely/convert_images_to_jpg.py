import os

def convert_png_to_jpg(path_to_png):
    from PIL import Image

    im = Image.open(path_to_png)
    rgb_im = im.convert('RGB')
    new_image_path = path_to_png.replace('.png', '.jpg')
    rgb_im.save(new_image_path)

    return new_image_path


def get_files_paths(src_dir, extensions):
    files_paths = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    files_paths.append(file_path)

    return files_paths


src_dir = '/root/land_pictures_data/inference/train/'


png_images_paths = get_files_paths(src_dir, ['.png'])

counter = 0
for image_path in png_images_paths:
    convert_png_to_jpg(image_path)
    os.remove(image_path)

    counter += 1

print(f'{counter} images removed')


