import os #used for interacrting with the file system
import glob #used for finding all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order
from PIL import Image #PIL is a library in Python that adds support for opening, manipulating, and saving many different image file formats.
import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms #for vision tasks

class AerialSLAMDataset(Dataset):
    def __init__(self,root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths=sorted(glob.glob(os.path.join(root_dir,"Tile *","images","image_part_*.jpg")))
        self.mask_paths=sorted(glob.glob(os.path.join(root_dir,"Tile *","masks","image_part_*.png")))

        if len(self.image_paths)==0:
            print(f"\n Warning :No images found in path :{os.path.abspath(root_dir)}")
            print("Double check paths and folders")

        elif len(self.mask_paths)==0:
            print("Warning: images found but not masks")

        elif len(self.image_paths)!=len(self.mask_paths):
            print("Size mismatch warning")
        else:
            print("Integrity check passed")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self , idx):
        img_path=self.image_paths[idx]
        mask_path=self.mask_paths[idx]

        image= Image.open(img_path).convert("RGB")
        mask= Image.open(mask_path).convert("L")

        if self.transform:
            image =self.transform(image)
            mask=self.transform(mask)

        mask = (mask*255).long().squeeze(0)

        return image, mask
    
if __name__=="__main__":
    data_transforms=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    dataset= AerialSLAMDataset(root_dir="data/Dataset" ,transform=data_transforms)
    print("Initialisation complete")

    if len(dataset)>0:
        dataloader=DataLoader(dataset,batch_size=2,shuffle=True)
        images,masks=next(iter(dataloader))
        print(images.shape)
        print("\n")
        print(masks.shape)

