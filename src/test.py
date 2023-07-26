import os

def pack_to_zip_and_copy(input_path, output_path):
    import shutil

    shutil.make_archive('', 'zip', input_path)


checkpoint_path = '/root/gld_pd/models/config_snacks4_2/'
visualizations_path = os.path.join(checkpoint_path, 'visualizations')



pack_to_zip_and_copy(source_folder_path)