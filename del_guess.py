import os
import shutil


def del_():
    folder = 'guesses'
    for subfolder in os.listdir(folder):
        subfolder = f'{folder}/{subfolder}'
        for filename in os.listdir(subfolder):
            file_path = os.path.join(subfolder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
