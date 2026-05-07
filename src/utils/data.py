import torch
import torch.utils.data as data_utils


class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len, parallel_ag=False):
        self.id2seq = id2seq
        self.max_len = max_len
        self.parallel = parallel_ag

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        hist = seq[:-1]
        hist = hist[-self.max_len :]
        mask_len = self.max_len - len(hist)
        hist_pad = [0] * mask_len + hist
        if self.parallel is True:
            target = [0] * mask_len + seq[-len(hist) :]
            assert sum([i > 0 for i in hist_pad]) == sum([i > 0 for i in target])
        else:
            target = [0] * (self.max_len - 1) + [seq[-1]]

        return torch.LongTensor(hist_pad), torch.LongTensor(target)

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train:
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.id_seq = data_train
        self.split = args.split_onebyone
        self.parallel_ag = args.parallel_ag
        if self.split:
            print("splitting data onebyone ...")
            self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {}
        idx = 0
        for seq_temp in self.u2seq:
            seq_temp = seq_temp[-self.max_len - 1 :]
            for star in range(len(seq_temp) - 1):
                self.id_seq[idx] = seq_temp[: star + 2]
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len, self.parallel_ag)
        return data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        seq = self.u2seq[index]
        hist = seq[-self.max_len :]
        padding_len = self.max_len - len(hist)
        hist_pad = [0] * padding_len + hist
        answer_pad = [0] * padding_len + seq[-(len(hist) - 1) :] + self.u2answer[index]
        assert sum([i > 0 for i in hist_pad]) == sum([i > 0 for i in answer_pad])
        return torch.LongTensor(hist_pad), torch.LongTensor(answer_pad)


class Data_Val:
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        seq = self.u2seq[index] + self.u2seq_add[index]
        hist = seq[-self.max_len :]
        padding_len = self.max_len - len(hist)
        hist_pad = [0] * padding_len + hist
        answer_pad = [0] * padding_len + seq[-(len(hist) - 1) :] + self.u2answer[index]
        assert sum([i > 0 for i in hist_pad]) == sum([i > 0 for i in answer_pad])
        return torch.LongTensor(hist_pad), torch.LongTensor(answer_pad)


class Data_Test:
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        return dataloader
