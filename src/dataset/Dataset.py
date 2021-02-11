import numpy as np

from utils/absolute_paths import absolute_paths
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, folder_A, folder_B, mode, rescale_size):
        super().__init__()
        self.files_A = [i for i in absolute_paths(folder_A)]
        self.files_B = [i for i in absolute_paths(folder_B)]
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)
        self.mode = mode
        self.rescale_size = rescale_size

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image
    
    def _prepare_sample(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image = image.resize((self.rescale_size, self.rescale_size))
        image = np.array(image)
        image = np.array(image/255, dtype='float32')
        return transform(image)

    def __getitem__(self, index):
        A = self.load_sample(self.files_A[index % self.len_A])
        A = self._prepare_sample(A)
        
        B = self.load_sample(self.files_B[index % self.len_B])
        B = self._prepare_sample(B)

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.len_A, self.len_B)