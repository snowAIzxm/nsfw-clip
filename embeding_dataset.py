from torch.utils.data import Dataset, DataLoader
import torch


class CustomDataset(Dataset):
    def __init__(self, data_label_list, label_ind_dict):
        data_list = [x[0] for x in data_label_list]
        label_list = [label_ind_dict[x[1]] for x in data_label_list]
        self.data = torch.tensor(data_list, dtype=torch.float32)  # 转换为张量
        self.labels = torch.tensor(label_list, dtype=torch.long)  # 转换为张量

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
