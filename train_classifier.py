from __future__ import print_function, division
import torch
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import sys
# from sklearn.model_selection import train_test_split, KFold
from glob import glob


warnings.filterwarnings("ignore")


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def train_model(label, dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                scheduler.step()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # running_total = 0
            print(phase)
            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs = batch["image"].to(device)
                labels = batch[label]
                labels = torch.from_numpy(np.asarray(labels)).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.float()  # ADDED AS A FIX
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # print("Loss: {}/{}".format(running_loss, dataset_sizes[phase]))
            print("Accuracy: {}/{}".format(running_corrects,
                                           dataset_sizes[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            training_results.append([phase, epoch, epoch_loss, epoch_acc])
            if epoch > 10:
                if phase == 'val' and epoch_loss < best_loss:
                    print("New leading accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == 'val':
                best_loss = epoch_loss
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy"]
    return model, training_results


class SkinDataset():
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.df.loc[self.df.index[idx], 'hasher']) + '.jpg'
        image = io.imread(img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        # high = self.df.loc[self.df.index[idx], 'high']
        # mid = self.df.loc[self.df.index[idx], 'mid']
        low = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick_scale'] 
        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    # 'high': high,
                    # 'mid': mid,
                    'low': low,
                    'hasher': hasher,
                    'fitzpatrick_scale': fitzpatrick
                }
        return sample

def custom_load(
        batch_size=256,
        num_workers=20,
        real_train_dir='',
        syn_train_dir='',
        val_dir='',
        real_image_dir='/ssd/janet/fitz_aug_final_version/data/finalfitz17k',
        syn_image_dir='',
        source=None):
    val = pd.read_csv(val_dir)
    real_train = pd.read_csv(real_train_dir)
    syn_train = pd.read_csv(syn_train_dir)
    
    if source == 'real':  
        class_sample_count = np.array(real_train[label].value_counts().sort_index())
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in real_train[label]])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(
            samples_weight.type('torch.DoubleTensor'),
            len(samples_weight),
            replacement=True)
        
        transformed_train = SkinDataset(
            csv_file=real_train_dir,
            root_dir=real_image_dir,
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                ])
            )
        
    elif source == 'syn':
        class_sample_count = np.array(syn_train[label].value_counts().sort_index())
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in syn_train[label]])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(
            samples_weight.type('torch.DoubleTensor'),
            len(samples_weight),
            replacement=True)
        
        transformed_train = SkinDataset(
            csv_file=syn_train_dir,
            root_dir=syn_image_dir,
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                ])
            )
        
    else:
        real_syn_labels = pd.concat([real_train[label], syn_train[label]])
        class_sample_count = np.array((real_syn_labels).value_counts().sort_index())
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in (real_syn_labels)])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(
            samples_weight.type('torch.DoubleTensor'),
            len(samples_weight),
            replacement=True)
        
        transformed_real_train = SkinDataset(
            csv_file=real_train_dir,
            root_dir=real_image_dir,
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                ])
            )
        transformed_syn_train = SkinDataset(
                    csv_file=syn_train_dir,
                    root_dir=syn_image_dir,
                    transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(),
                        transforms.RandomHorizontalFlip(),
                        transforms.CenterCrop(size=224),  # Image net standards
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                        ])
                    )
                        
        transformed_train = torch.utils.data.ConcatDataset(
            [transformed_real_train, transformed_syn_train])
        
    transformed_test = SkinDataset(
        csv_file=val_dir,
        root_dir=real_image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
            ])
        )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            # shuffle=True,
            num_workers=num_workers),
        "val": torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
        }
    
    dataset_sizes = {"train": len(transformed_train), "val": len(transformed_test)}

    return dataloaders, dataset_sizes

if __name__ == '__main__':
    # In the custom_load() function, make sure to specify the path to the images
    # print("\nPlease specify number of epochs and 'dev' mode or not... e.g. python train.py 10 full \n")
    n_epochs = int(sys.argv[1])
    dev_mode = sys.argv[2]
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev_mode == "dev":
        df = pd.read_csv("fitzpatrick17k.csv").sample(1000)
    else:
        df = pd.read_csv("./data_splits/Fitz_subset.csv") 
    print(df['fitzpatrick_scale'].value_counts())
    print("Rows: {}".format(df.shape[0]))
    df["low"] = df['label'].astype('category').cat.codes
    # df["mid"] = df['nine_partition_label'].astype('category').cat.codes
    # df["high"] = df['three_partition_label'].astype('category').cat.codes
    df["hasher"] = df["md5hash"]
    
    SPLIT = 'light_dark_seed_to_dark'
    # for SEED in [0,42,123,1111,1234]:
    for SEED in [0]:
        seed_path = f"./data_splits/data_seed={SEED}/seed_4/*/*/*.jpg" # change; path to seed imgs 
        syn_img = f"./inference_img/{SPLIT}_SEED={SEED}_steps=100_strength=0.5_guidance=2" # change; path to syn imgs
        syn_path = f"{syn_img}.csv"
        synthetic_train = pd.read_csv(syn_path)
        synthetic_train["low"] = synthetic_train['label'].astype('category').cat.codes
        synthetic_train["hasher"] = synthetic_train["md5hash"]

        test_files = [os.path.splitext(os.path.basename(path))[0] for path in glob(seed_path)]
        test_mask = df.md5hash.isin(test_files)
        
        all_train = df[~test_mask]
        real_train = all_train[(all_train.fitzpatrick_scale == 1)|(all_train.fitzpatrick_scale == 2)]
        
        seed = df[test_mask]
        seed_dark = seed[(seed.fitzpatrick_scale == 5)|(seed.fitzpatrick_scale == 6)]
        
        real_train = pd.concat([real_train, seed_dark])
        
        # test = df[test_mask]
        # test = test[(test.fitzpatrick_scale == 5)| (test.fitzpatrick_scale == 6)]
        test = all_train[(all_train.fitzpatrick_scale == 5)|(all_train.fitzpatrick_scale == 6)]
        
        print('*'*10, SEED, '*'*10)
        print(real_train.shape[0], synthetic_train.shape[0], test.shape[0])
        print(real_train.fitzpatrick_scale.unique(), test.fitzpatrick_scale.unique())
        
        ### change; paths to intermediate csv files
        real_train_path = "./temp_csv/temp_train_light_to_dark.csv" 
        syn_train_path = './temp_csv/temp_syn_light_to_dark.csv'
        test_path = "./temp_csv/temp_test_light_to_dark.csv"
        
        real_train.to_csv(real_train_path, index=False)
        synthetic_train.to_csv(syn_train_path, index=False)
        test.to_csv(test_path, index=False)

    
        for source in ["real", "syn", 'real_syn']:
            print(f'source: {source}')
            label = 'low'
            if source == 'real':
                weights = np.array(max(real_train[label].value_counts())/real_train[label].value_counts().sort_index())
                label_codes = sorted(list(real_train[label].unique()))
            elif source == 'syn':
                weights = np.array(max(synthetic_train[label].value_counts())/synthetic_train[label].value_counts().sort_index())
                label_codes = sorted(list(synthetic_train[label].unique()))
            else:
                real_syn_labels = pd.concat([real_train[label], synthetic_train[label]])
                weights = np.array(max((real_syn_labels).value_counts())/(real_syn_labels).value_counts().sort_index())
                label_codes = sorted(list((real_syn_labels).unique()))
            
                # weights = np.array(max(train[label].value_counts())/train[label].value_counts().sort_index())
                # label_codes = sorted(list(train[label].unique()))
            
            dataloaders, dataset_sizes = custom_load(
                256,
                20,
                "{}".format(real_train_path),
                "{}".format(syn_train_path),
                "{}".format(test_path),
                real_image_dir='./data/finalfitz17k', # change; path to real imgs
                syn_image_dir=syn_img,
                source=source)
            model_ft = models.vgg16(pretrained=True)
            for param in model_ft.parameters():
                param.requires_grad = False
            model_ft.classifier[6] = nn.Sequential(
                        nn.Linear(4096, 256), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, len(label_codes)),                   
                        nn.LogSoftmax(dim=1))
            total_params = sum(p.numel() for p in model_ft.parameters())
            print('{} total parameters'.format(total_params))
            total_trainable_params = sum(
                p.numel() for p in model_ft.parameters() if p.requires_grad)
            print('{} total trainable parameters'.format(total_trainable_params))
            model_ft = model_ft.to(device)
            model_ft = nn.DataParallel(model_ft)
            class_weights = torch.FloatTensor(weights).cuda()
            criterion = nn.NLLLoss()
            optimizer_ft = optim.Adam(model_ft.parameters())
            exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer_ft,
                step_size=n_epochs//3,
                gamma=0.1)
            print("\nTraining classifier for {}........ \n".format(label))
            print("....... processing ........ \n")
            model_ft, training_results = train_model(
                label,
                dataloaders, device,
                dataset_sizes, model_ft,
                criterion, optimizer_ft,
                exp_lr_scheduler, n_epochs)
            print("Training Complete")
