import shutil
import os

def zip_folder(folder_path, output_path):
    shutil.make_archive(output_path, 'zip', folder_path)

# Example usage
folder_to_zip = 'output/'
output_zip = 'out.zip'

zip_folder(folder_to_zip, output_zip)