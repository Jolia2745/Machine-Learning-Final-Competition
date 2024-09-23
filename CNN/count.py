import os

def count_files(directory):
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)
    return file_count

directory = './MLproject/data/test_images'
print("Number of files:", count_files(directory))