import os

import numpy as np

from model import H14_NSFW_Detector

train_ratio = 0.7
label_ind_dict = {'drawings': 0, 'hentai': 1, 'neutral': 2, 'porn': 3, 'sexy': 4}


# trainfile https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing 据说没标注
# testfile  https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/nsfw_testset.zip 据说标注过

class ModelTrainer:
    def __init__(self, data_dir: str, dim=768,
                 additional_train_label_emb_dict=None,
                 additional_test_label_emb_dict=None):
        # data
        self.additional_train_label_emb_dict = additional_train_label_emb_dict
        self.additional_test_label_emb_dict = additional_test_label_emb_dict
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_origin_data(self, ):
        raise NotImplementedError

    def load_dir_emb_dict(self, file_path):
        dir_list = os.listdir(file_path)
        dir_emb_dict = {}
        for dir1 in dir_list:
            emd_file_path = os.path.join(os.path.join(file_path, dir1), "img_emb/img_emb_0.npy")
            if os.path.exists(emd_file_path):
                dir_emb_dict[dir1] = np.load(emd_file_path)
        return dir_emb_dict

    def download

        def train(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            loss_func = torch.nn.CrossEntropyLoss()
            save_dir = "./train_model"
            os.makedirs(save_dir, exist_ok=True)
            val_acc_list = []
            test_acc_list = []
            for epoch in range(20):
                self.model.train()
                for i, (data, label) in enumerate(self.train_dataloader):
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = loss_func(output, label)
                    loss.backward()
                    optimizer.step()
                    if i % 10 == 0:
                        log(f"epoch: {epoch}, step: {i}, loss: {loss.item()}")
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
