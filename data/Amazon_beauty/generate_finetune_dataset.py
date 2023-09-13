import os
from tqdm import tqdm
import itertools

folder = "ood_generate_dataset_tiny_10_30u30i"
for file in ["train.txt", "test.txt"]:
    records = []
    with open(os.path.join(folder, file), "r+") as reader:
        for line in tqdm(reader):
            user_id, item_id, trigger_seq, click_seq, label = line.strip("\n").split(";")
            # records.append([int(user_id), int(item_id), trigger_seq, click_seq, int(label)])
            records.append([user_id, item_id, trigger_seq, click_seq, int(label)])

    records.sort()
    for user_id, actions in tqdm(itertools.groupby(records, key=lambda x: x[0])):
        user_folder = os.path.join(folder, "finetune_dataset", user_id)
        user_file = os.path.join(user_folder, file)
        actions = list(actions)
        # print(actions)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        with open(user_file, "w+") as writer:
            for line in actions:
                user_id, item_id, trigger_seq, click_seq, label = line
                print(int(user_id), int(item_id), trigger_seq, click_seq, int(label), sep=";", file=writer)

