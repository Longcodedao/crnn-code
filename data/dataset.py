#!/usr/bin/env python
# coding: utf-8
import os 
import torch

from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
import torchvision.transforms  as transforms


class Synth90kDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHARS2LABELS = {char: label for label, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHARS2LABELS.items()}

    def __init__(self, dataset_path, mode, img_height = 32, img_width = 100):
        self.files, self.texts = self.load_data(dataset_path, mode)
        self.img_height = 32
        self.img_width = 100

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5], std = [0.5])
        ])
        
    def load_data(self, dataset_path, mode):
        print(f"Loading Dataset with mode: {mode}")
        mapping_strings = {}
        with open(os.path.join(dataset_path, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping_strings[i] = line.strip()

        annotation_files = None
        if mode == 'train':
            annotation_files = 'annotation_train.txt'
        elif mode == 'val':
            annotation_files = 'annotation_val.txt'
        elif mode == 'test':
            annotation_files = 'annotation_test.txt'

        files = []
        texts = []
        with open(os.path.join(dataset_path, annotation_files), 'r') as fr:
            for line in fr.readlines():
                file_path, index = line.strip().split(' ')
                file_path = os.path.join(dataset_path, file_path)
                files.append(file_path)
                texts.append(mapping_strings[int(index)])
                
        return files, texts

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_file = self.files[index]
        # print(image_file)
        try: 
            image = Image.open(image_file)
        except IOError:
            print(f'Corrupt Image at {index}')
            return self[index + 1]

        # Preprocessing Image
        image = self.transform(image)
        text = self.texts[index]
        target = [self.CHARS2LABELS[c] for c in text]
        return {"image" : image, 
                "target": torch.tensor(target, dtype = torch.int64),  
                "target_length": torch.tensor([len(target)], dtype = torch.int64)}

def synth90k_collate_fn(batch):
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    target_lengths = [item["target_length"] for item in batch]

    # We concat because the length output of each is different 
    images = torch.stack(images, dim = 0)
    targets = torch.cat(targets, dim = 0)
    target_lengths = torch.cat(target_lengths, dim = 0)

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths
    }

