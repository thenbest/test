import numpy as np
import glob, os
import data_split
from tqdm import tqdm
from multiprocessing.pool import Pool
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import params
import torch
import librosa


# dataset class for DataLoader
class MelSpecDataset(Dataset):
    # constructor
    def __init__(self, root_dir, transform=None, mode='train', normalize=True):
        classes = params.class_labels
        self.files = []
        self.transform = transform
        self.mode = mode
        self.normalize = normalize
        self.root_dir = root_dir

        if self.mode == 'train':
            for c in classes:
                lists = glob.glob(params.train_list_path + c + "/*.csv")
                for l in lists:
                    lines = open(l, 'r').readlines()
                    lines = [{"file": l.strip(), "class": c} for l in lines]

                    self.files.extend(lines)

        elif self.mode == 'test':
            lines = open(params.test_list_path, 'r').readlines()
            lines = [l.strip() for l in lines]

            self.files.extend(lines)

    def __len__(self):
        return len(self.files)

    def __getfilename__(self, idx):
        if self.mode == 'train':
            return self.root_dir + self.files[idx]['file']
        else:
            return self.root_dir + self.files[idx]

    def __getitem__(self, idx):
        if self.mode == 'train':
            spec = loadData(self.root_dir + self.files[idx]['file'])
        else:
            spec = loadData(self.root_dir + self.files[idx])

        if self.normalize:
            # Normalize all values in a spectrogram to [0, 1]
            spec_one_column = spec.reshape([-1, 1])
            scaler = MinMaxScaler()
            spec_one_column = scaler.fit_transform(spec_one_column)
            spec = spec_one_column.reshape(spec.shape)

        if self.transform:
            spec = self.transform(spec)
            spec = torch.squeeze(spec, dim=0)
            spec = spec.transpose(0, 1)

        if self.mode == 'train':
            class_name = self.files[idx]['class']
            label = params.class_labels.index(class_name)

            return spec, label

        elif self.mode == 'test':
            # time_len = spec.shape[2]
            # Test feat can be shorter than training feats. if shorter, pad it
            # if time_len < params.num_frames:
            # only pad time axis
            # spec = torch.from_numpy(np.pad(spec, ((0,0), (0,0),(0, params.num_frames - time_len)), 'constant'))
            # print(spec.dtype)

            # Test data have no labels, so skip label acquisition
            return spec, self.files[idx]


def loadData(filename):
    # load numpy matrix from a file and trim
    mel = np.load(filename)
    return mel


def main():
    # create a dataset containing all necessary data
    # dataset = MelSpecDataset("/data8/audio_dataset/KGDataset/feats/train/", transform=transforms.ToTensor())
    # dataset = MelSpecDataset(params.train_feats_path, transform=transforms.ToTensor(), mode='train')
    dataset = MelSpecDataset(params.train_feats_path, transform=transforms.ToTensor(), mode='train')
    ds_len = dataset.__len__()

    print(ds_len)
    '''
    for i in range(ds_len):
        try:
            dataset.__getitem__(i)[0]
            print(dataset.__getfilename__(i),  dataset.__getitem__(i)[1], "pass")
        except Exception:
            print(dataset.__getfilename__(i),  dataset.__getitem__(i)[1], "failed")
    '''

    # print(dataset.__getitem__(0)[0].shape)
    # print(dataset.__getitem__(5)[0].shape)
    # print(dataset.__getitem__(8)[0].shape)

    # test_loader = tqdm(torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4))

    # for data in test_loader:
    #    X, y = data

    ''' 
    for spec, label in dataset:

        if count <= 4000:
            continue
        else:
            print(spec.shape)
            print(label)

        count +=1

    split = data_split.DataSplit(dataset, shuffle=True)
    tr_loader, _, test_loader = split.get_split(batch_size=8, num_workers=2)

    # placeholder for training
    for specs, labels in tr_loader:
        print(specs.shape)

    for specs, labels in test_loader:
        print(specs.shape)
    '''


if __name__ == "__main__":
    main()