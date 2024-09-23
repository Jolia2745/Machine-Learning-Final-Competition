import os

def count_files(directory):
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)
    return file_count

directory1 = './data_v3/test_images'
print("Number of test files:", count_files(directory1)) #2447

directory2 = './data_v3/train_images'
print("Number of train files:", count_files(directory2)) # 11886