import os
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from embeding_dataset import CustomDataset
from model import H14_NSFW_Detector

train_ratio = 0.7
label_ind_dict = {'drawings': 0, 'hentai': 1, 'neutral': 2, 'porn': 3, 'sexy': 4}


# trainfile https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing 据说没标注
# testfile  https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/nsfw_testset.zip 据说标注过

class ModelTrainer:
    def __init__(self, data_dir: str, model_dir: str, dim=768,
                 additional_train_label_emb_dict=None,
                 additional_test_label_emb_dict=None):
        # data
        self.additional_train_label_emb_dict = additional_train_label_emb_dict
        self.additional_test_label_emb_dict = additional_test_label_emb_dict
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.model = H14_NSFW_Detector(dim, len(label_ind_dict))
        self.model_dir = model_dir

    def download_origin_data(self, ):
        train_url = "https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing"
        res = os.system(
            f"wget --no-check-certificate '{train_url}' -O {self.data_dir}/train.zip")
        assert res == 0
        test_url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/nsfw_testset.zip"
        res = os.system(
            f"wget --no-check-certificate '{test_url}' -O {self.data_dir}/test.zip")
        assert res == 0
        os.system(f"unzip {self.data_dir}/train.zip -d {self.data_dir}")
        os.system(f"unzip {self.data_dir}/test.zip -d {self.data_dir}")

    def load_dataloader(self):
        train_emb_dict = self.load_dir_emb_dict(os.path.join(self.data_dir, "train"))
        test_emb_dict = self.load_dir_emb_dict(os.path.join(self.data_dir, "test/nsfw_testset"))
        all_data_list = []
        for k, v in train_emb_dict.items():
            for i in range(len(v)):
                all_data_list.append((v[i], k))
        if self.additional_train_label_emb_dict is not None:
            for k, v in self.additional_train_label_emb_dict.items():
                for i in range(len(v)):
                    all_data_list.append((v[i], k))

        random.shuffle(all_data_list)
        train_data_len = int(len(all_data_list) * train_ratio)
        train_data_list = all_data_list[:train_data_len]
        val_data_list = all_data_list[train_data_len:]
        train_dataset = CustomDataset(train_data_list, label_ind_dict)
        val_dataset = CustomDataset(val_data_list, label_ind_dict)
        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_data_list = []
        for k, v in test_emb_dict.items():
            label = k.split("-")[0]
            for i in range(len(v)):
                test_data_list.append((v[i], label))
        if self.additional_test_label_emb_dict is not None:
            for k, v in self.additional_test_label_emb_dict.items():
                for i in range(len(v)):
                    test_data_list.append((v[i], k))
        test_dataset = CustomDataset(test_data_list, label_ind_dict)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader

    def load_dir_emb_dict(self, file_path):
        dir_list = os.listdir(file_path)
        dir_emb_dict = {}
        for dir1 in dir_list:
            emd_file_path = os.path.join(os.path.join(file_path, dir1), "img_emb/img_emb_0.npy")
            if os.path.exists(emd_file_path):
                dir_emb_dict[dir1] = np.load(emd_file_path)
        return dir_emb_dict

    def train(self, epoch_amount):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_func = torch.nn.CrossEntropyLoss()
        save_dir = "./train_model"
        os.makedirs(save_dir, exist_ok=True)
        val_acc_list = []
        test_acc_list = []
        for epoch in range(epoch_amount):
            self.model.train()
            for i, (data, label) in enumerate(self.train_dataloader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_func(output, label)
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print(f"epoch: {epoch}, step: {i}, loss: {loss.item()}")
            self.model.eval()
            label_count_dict = defaultdict(int)
            label_acc_dict = defaultdict(int)
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (data, label) in enumerate(self.val_dataloader):
                    output = self.model(data)
                    _, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    for p, l in zip(predicted, label):
                        if p == l:
                            label_acc_dict[int(l)] += 1
                        label_count_dict[int(l)] += 1

                log(f"epoch: {epoch}, accuracy: {correct / total}")
                val_acc_list.append(correct / total)
            # with torch.no_grad():
            #     correct = 0
            #     total = 0
            #     for i, (data, label) in enumerate(self.test_dataloader):
            #         output = self.model(data)
            #         _, predicted = torch.max(output, 1)
            #         total += label.size(0)
            #         correct += (predicted == label).sum().item()
            #     log(f"epoch: {epoch}, accuracy: {correct / total}")
            #     test_acc_list.append(correct / total)
            torch.save(self.model, os.path.join(save_dir, f"model_{epoch}.pth"))


self = ModelTrainer("./train", "./test/nsfw_testset/", 768)
