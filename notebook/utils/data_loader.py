import numpy as np
import torch


class DataLoader():
    def __init__(self, train_val_reader, test_reader=None, validation_split=0.2):

        indices = list(range(len(train_val_reader)))
        np.random.RandomState(0).shuffle(indices)
        split = int(np.floor(validation_split * len(train_val_reader)))
        train_indices, val_indices = indices[split:], indices[:split]

        self.train_dataset = torch.utils.data.Subset(train_val_reader, train_indices)
        self.val_dataset = torch.utils.data.Subset(train_val_reader, val_indices)

        if test_reader is not None:
            self.test_dataset=test_reader

    def get_train_loader(self, batch_size=4, num_workers=2):
        print('INFO: Training data loader initialized.')
        return  torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers)

    def get_validation_loader(self, batch_size, num_workers):
        print('INFO: Validation data loader initialized.')
        return  torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers)

    def get_test_loader(self, batch_size, num_workers):
        print('INFO: Test data loader initialized.')
        return  torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers)

