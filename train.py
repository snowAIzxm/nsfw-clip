import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from embeding_dataset import CustomDataset
from model import H14_NSFW_Detector

train_ratio = 0.7
label_ind_dict = {'drawings': 0, 'hentai': 1, 'neutral': 2, 'porn': 3, 'sexy': 4}
bad_label_list = ["porn", "hentai"]


# trainfile https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing 据说没标注
# testfile  https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/nsfw_testset.zip 据说标注过

class ModelTrainer:
    def __init__(self, data_dir: str, model_dir: str, dim=768, device="cuda",
                 additional_train_label_emb_dict=None,
                 additional_test_label_emb_dict=None):
        # data
        self.additional_train_label_emb_dict = additional_train_label_emb_dict
        self.additional_test_label_emb_dict = additional_test_label_emb_dict
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.model = H14_NSFW_Detector(dim, len(label_ind_dict)).to(device)
        self.model_dir = model_dir
        self.device = device
        os.makedirs(self.model_dir, exist_ok=True)

    def download_origin_data(self, ):
        # 国内网络问题，未debug，可直接下载后放入文件夹
        train_url = "https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing"
        res = os.system(
            f"wget --no-check-certificate '{train_url}' -O {self.data_dir}/train.zip")
        assert res == 0
        test_url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/nsfw_testset.zip"
        res = os.system(
            f"wget --no-check-certificate '{test_url}' -O {self.data_dir}/test.zip")
        assert res == 0
        os.system(f"unzip {self.data_dir}/train.zip -d {self.data_dir}/train")
        os.system(f"unzip {self.data_dir}/test.zip -d {self.data_dir}/test")

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
        val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
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
        # 主函数，需要先下载数据之后就可以了
        train_dataloader, val_dataloader, test_dataloader = self.load_dataloader()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch in range(epoch_amount):
            self.model.train()
            for i, (data, label) in enumerate(train_dataloader):
                label = label.to(self.device)
                optimizer.zero_grad()
                output = self.model(data.to(self.device))
                loss = loss_func(output, label)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"epoch: {epoch}, step: {i}, loss: {loss.item()}")
            print("val")
            self.show_model_dataloader_result(self.model, val_dataloader)
            print("test")
            self.show_model_dataloader_result(self.model, test_dataloader)

            torch.save(self.model.state_dict(), os.path.join(self.model_dir, f"model_{epoch}.pth"))

    @staticmethod
    def show_model_dataloader_result(model, dataloader):
        # 测试集drawing效果较差，不确定是不是测试集数据or训练数据集问题，但是本身不影响nsfw的训练，
        # 本身有展示bad 和good的ratio
        model.eval()
        label_count_dict = defaultdict(int)
        label_acc_dict = defaultdict(int)
        label_error_count_dict = defaultdict(lambda: defaultdict(int))
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (data, label) in enumerate(dataloader):
                output = model(data.to(self.device)).cpu()
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                for p, l in zip(predicted, label):
                    if p == l:
                        label_acc_dict[int(l)] += 1
                    else:
                        label_error_count_dict[int(l)][int(p)] += 1
                    label_count_dict[int(l)] += 1

        print(f"accuracy: {correct / total}")

        result = []
        ind_label_dict = {v: k for k, v in label_ind_dict.items()}
        for label, ind in label_ind_dict.items():
            amount = label_count_dict[ind]
            acc = label_acc_dict[ind]
            acc_ratio = f"{round(acc / amount * 100, 1)}%"
            new_error_count_dict = {}
            for k, v in label_error_count_dict[ind].items():
                new_error_count_dict[ind_label_dict[k]] = v

            good_amount = 0
            bad_amount = 0
            if label in bad_label_list:
                bad_amount = acc
            else:
                good_amount = acc
            for k, v in label_error_count_dict[ind].items():
                if ind_label_dict[k] in bad_label_list:
                    bad_amount += v
                else:
                    good_amount += v

            result.append(
                dict(label=label, amount=amount, acc=acc, aac_ratio=acc_ratio,
                     good_ratio=round(good_amount / amount * 100, 1),
                     bad_ratio=round(bad_amount / amount * 100, 1), error_count_dict=new_error_count_dict, )
            )

        print(pd.DataFrame(result))
        return result



self = ModelTrainer("./nsfw_data", "./tmp+model", 768)
