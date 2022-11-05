import os
import random
import shutil


ORIGINAL_DTS_PATH = '../Celeb-DF'
DST_DTS_PATH = '../formatted_dataset'
DATA_PROPORTION = [0.8, 0.1, 0.1]  # train: validation: test

labeled_dts = {
    "real": set([]),
    "synthesis": set([])
}

with open(os.path.join(ORIGINAL_DTS_PATH, 'List_of_testing_videos.txt')) as f:
    lines = f.readlines()
    for line_str in lines:
        line_list = line_str.split()

        if line_list[0] == '1':
            labeled_dts['real'].add(line_list[1])
        elif line_list[0] == '0':
            labeled_dts['synthesis'].add(line_list[1])

total_labeled_dts = {
    "real": len(labeled_dts['real']),
    "synthesis": len(labeled_dts['synthesis'])
}
os.makedirs(DST_DTS_PATH, exist_ok=True)


def feed_data_into(data_group, data_proportion):
    group_path = os.path.join(DST_DTS_PATH, data_group)
    os.makedirs(group_path, exist_ok=True)

    for data_class in ('real', 'synthesis'):
        class_path = os.path.join(group_path, data_class)
        os.makedirs(class_path, exist_ok=True)
        total_data = round(data_proportion * total_labeled_dts[data_class])
        files = set(random.sample(labeled_dts[data_class], k=total_data))
        labeled_dts[data_class] = labeled_dts[data_class] - files

        for j, file in enumerate(files):
            source = os.path.join(ORIGINAL_DTS_PATH, file)
            file_extension = os.path.splitext(file)[1]
            new_file_name = f"{j:04d}{file_extension}"
            destination = os.path.join(class_path, new_file_name)
            shutil.copyfile(source, destination)


for i, group_name in enumerate(('train', 'validation', 'test',)):
    feed_data_into(group_name, DATA_PROPORTION[i])
