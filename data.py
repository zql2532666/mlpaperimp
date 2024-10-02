import torchvision.transforms as transforms
from torchvision import datasets
import torch
import random
    
# Custom transformation for scale jittering
class RandomScaleJitter:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        # Randomly sample a new size from the given range [min_size, max_size]
        new_size = random.randint(self.min_size, self.max_size)
        # Resize the image while maintaining the aspect ratio
        return transforms.Resize(new_size)(img)
    

def load_data(batch_size, S=256, Q=None,train_dir='./data/train', val_dir='./data/valid', test_dir='./data/test', crop_size=224):

    # Normalize does the following for each channel:
    # image = (image - mean) / std
    # The parameters mean, std are passed as 0.5, 0.5. 
    # This will normalize the image in the range [-1,1]. For example, the minimum value 0 will be converted to (0-0.5)/0.5=-1, 
    # the maximum value of 1 will be converted to (1-0.5)/0.5=1.

    if isinstance(S, tuple):
        # uses scale jittering
        transform_train = transforms.Compose([
                                            RandomScaleJitter(S[0], S[1]),
                                            transforms.RandomCrop(crop_size), 
                                            transforms.RandomHorizontalFlip(p=0.7), 
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], 
                                                                [0.5, 0.5, 0.5])
                                        ])
        if not Q:
            Q = int((S[0] + S[1]) / 2)

    else:
        transform_train = transforms.Compose([
                                            transforms.Resize((S, S)), 
                                            transforms.RandomCrop(crop_size),
                                            transforms.RandomHorizontalFlip(p=0.7), 
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], 
                                                                [0.5, 0.5, 0.5])
                                        ])
    
    if not Q:
        Q = S

    transform_valid = transforms.Compose([ 
                                            transforms.Resize((S, S)),  # Resize to fixed size
                                            transforms.CenterCrop(crop_size),  # Center crop to the same size as during training
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Ensure same normalization as training
                                        ])

    transform_test = transforms.Compose([
                                        # transforms.Resize((Q, Q)),
                                        transforms.Resize((crop_size, crop_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], 
                                                                [0.5, 0.5, 0.5])
                                    ])


    dataset_train = datasets.ImageFolder(train_dir, transform=transform_train)
    dataset_valid = datasets.ImageFolder(val_dir, transform=transform_valid)
    dataset_test = datasets.ImageFolder(test_dir, transform=transform_valid)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


