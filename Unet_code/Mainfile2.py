PROJECT_NAME = "Test UNET Run"

######################################
# Imports
######################################

import os
import random
import torch
import torch.nn.functional as F
from torchvision import transforms, utils
import numpy as np
from unet import UNet
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image as im, ImageOps
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import cv2
import pickle as pkl


#from monai.networks.nets import UNet
#from monai.networks.layers import Norm
import shutil
from fastprogress.fastprogress import master_bar, progress_bar
from torchsummary import summary
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger






######################################
# Functions
######################################

def sampleRV(BOUNDS):
    return np.random.uniform(BOUNDS[0], BOUNDS[1])

def clip_img(image):
    image = np.where(image > 1, 1, image)
    return np.where(image < 0, 0, image)

def normalize(image):
    new_image = image - np.min(image)
    return new_image / np.max(new_image)

def show(image):
    display(im.fromarray(np.uint8(image * 255)))

def add_rician_noise(image, gain=0.1):
    image = image.numpy()#.resize((image.shape[-2], image.shape[-1]))
    n1 = normalize(np.random.normal(0, 1, (image.shape[-2], image.shape[-1])))
    n2 = normalize(np.random.normal(0, 1, (image.shape[-2], image.shape[-1])))

    return torch.tensor(clip_img(np.abs(image + gain*n1 + gain*n2*1j)))

def noiseSimV1(img_tensor):
    img_tensor = img_tensor.detach().cpu()
    SIZE = img_tensor.shape[-1]
    x = np.linspace(-1, 1, num=SIZE)

    # Vertical
    y_vert = (1 / (0.8 + np.exp(-3*x))) + 0.2
    vert_mask = (normalize(np.tile(np.reshape(y_vert, (-1, 1)), (1, SIZE))) * 0.95) + 0.1

    # Horizontal
    y_horiz = -(x**4)
    y_horiz = (y_horiz / np.max(np.absolute(y_horiz))) + 1
    horiz_mask = normalize(np.tile(np.reshape(y_horiz, (1, -1)), (SIZE, 1))) + 0.05

    # Combined
    mask = torch.tensor(normalize(4*vert_mask * horiz_mask))

    noise_intensity = random.random()*0.05 + 0.05 # 0.05 to 0.10, uniformly distributed
    return add_rician_noise(torch.mul(img_tensor, mask), noise_intensity)

def noiseSimV2(img_tensor, bias_tensor, noise_gain=0.1):
    img_tensor = img_tensor.detach().cpu()
    SIZE = img_tensor.shape[-1] # TODO Remove if possible

    if img_tensor.shape != bias_tensor.shape:
        print('ERROR: noiseSimV2: args img_tensor and bias_tensor have different shapes')
        print('  -> img_tensor.shape = ' + str(img_tensor.shape))
        print('  -> bias_tensor.shape = ' + str(bias_tensor.shape))
    
    biased_tensor = (img_tensor * bias_tensor)# * 255
    return add_rician_noise(biased_tensor, gain=noise_gain)


def display_tensor(img_tensor):
    img_tensor = img_tensor.detach().cpu()
    plt.imshow(  img_tensor.view(img_tensor.shape[1], img_tensor.shape[2]), cmap='gray')




######################################
# Dataloader
######################################

class KneeDataset(Dataset):
    """Knee dataset."""

    def __init__(self, root_dir, bias_fields, noise_bounds, size=512, preload=False):
        super().__init__()
        """
        Args:
            root_dir      (string):         Path of directory with all the image files
            bias_fields   ([np.ndarray]):   List of bias field numpy arrays
            noise_bounds  ((LOWER, UPPER)): Tuple of lower and upper bounds for noise gain
            size          (int):            Square image side length in pixels
        """
        self.root_dir = root_dir
        self.file_names = [name for name in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, name))]
        self.length = len(self.file_names)
        self.bias_fields = bias_fields

        # Make sure the dimensions of the bias fields match the images
        if bias_fields[0].shape[-1] != size:
            for i in range(len(bias_fields)):
                self.bias_fields[i] = cv2.resize(bias_fields[i], dsize=(size, size), interpolation=cv2.INTER_CUBIC)
            
        # Fix any NaNs in bias fields
        for i in range(len(bias_fields)):
            np.nan_to_num(self.bias_fields[i], copy=False, nan=0.0, posinf=1.0, neginf=0.0)

        self.noise_bounds = noise_bounds
        self.SIZE = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.SIZE, self.SIZE))
        ])
        
        self.preload = preload
        if not preload:
            return
        
        # Load images into memory
        self.images = []
        for idx in range(self.length):
            img_path = os.path.join(self.root_dir, self.file_names[idx])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.
            img_tensor = self.transform(img).float()
            img_tensor -= img_tensor.min()
            img_tensor /= (img_tensor.max() + 1e-9)

            self.images.append(img_tensor)
            '''
            bias_tensor = torch.tensor(random.choice(self.bias_fields)).view(1, self.SIZE, self.SIZE)
            noise_gain = sampleRV(self.noise_bounds)

            sim_singlecoil_tensor = noiseSimV2(img_tensor, bias_tensor, noise_gain=noise_gain).float()
            sim_singlecoil_tensor -= sim_singlecoil_tensor.min()
            sim_singlecoil_tensor /= (sim_singlecoil_tensor.max() + 1e-9)

            self.images.append((sim_singlecoil_tensor, img_tensor))
            '''

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.preload:
            img_tensor = self.images[idx]
        else:
            img_path = os.path.join(self.root_dir, self.file_names[idx])
            if os.path.exists(img_path):
                #img = im.open(img_path)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.
            else:
                print('ERROR: __getitem__: Image index ' + str(idx) + ' was not found')
                print('  -> Missing file path: ' + str(img_path))

            img_tensor = self.transform(img).float()
            img_tensor -= img_tensor.min()
            img_tensor /= (img_tensor.max() + 1e-9)

        bias_tensor = torch.tensor(random.choice(self.bias_fields)).view(1, self.SIZE, self.SIZE)
        noise_gain = sampleRV(self.noise_bounds)

        sim_singlecoil_tensor = noiseSimV2(img_tensor, bias_tensor, noise_gain=noise_gain).float()
        sim_singlecoil_tensor -= sim_singlecoil_tensor.min()
        sim_singlecoil_tensor /= (sim_singlecoil_tensor.max() + 1e-9)

        return sim_singlecoil_tensor, img_tensor







if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")



    ssim = StructuralSimilarityIndexMeasure()
    psnr = PeakSignalNoiseRatio()


    ######################################
    # Setting up model stuff
    ######################################

    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    SIZE = 320
    TEST_RATIO = 0.2
    VAL_RATIO = 0.2

    ######################################
    # Open bias fields and set up knee dataset
    ######################################

    #with open(r'/content/drive/MyDrive/Summer 2022/ECE 697/Project/share/data/synth_bias_fields/fields1.pkl', 'rb') as bias_file:
    with open(r'fields1.pkl', 'rb') as bias_file:
        bias_fields = pkl.load(bias_file)
    all_dataset = KneeDataset(root_dir = r'multicoil', bias_fields=bias_fields, noise_bounds=(0., 0.1), size=SIZE, preload = False)


    model = UNet(in_channels=1, n_classes=1, wf=6, depth=6, padding=True, up_mode='upconv', batch_norm=True).to(device)
    #model = UNet(dimensions=2, in_channels=1, out_channels=1, channels=(32, 64, 128, 256, 512, 1024),
    #           strides=(2, 2, 2, 2, 2), num_res_units=3, norm=Norm.BATCH).to(device)
    optim = torch.optim.Adam(model.parameters())

    #all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)




    ######################################
    # Data Partition
    ######################################

    train_size = int( (1 - (TEST_RATIO + VAL_RATIO)) * len(all_dataset) )
    test_val_size = len(all_dataset) - train_size
    test_val_set, train_set = random_split(all_dataset, [train_size, test_val_size])

    test_size = int( (TEST_RATIO / (TEST_RATIO + VAL_RATIO)) * len(test_val_set) )
    val_size = len(test_val_set) - test_size
    test_set, val_set = random_split(test_val_set, [test_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    #val_dataset = KneeDataset(root_dir = r'/content/knee_test', bias_fields=bias_fields, noise_bounds=(0., 0.1), size=SIZE)
    #valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_lens = {'train':len(train_set), 'val':len(val_set)}




    ######################################
    # Init Weights and Biases
    ######################################

    wandb.init(project=PROJECT_NAME)

    wandb.config = {
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE
    }



    ######################################
    # More functions
    ######################################

    def criterion(pred, y, weight):
        return F.mse_loss(pred, y)
        #return F.l1_loss(pred, y)
        #return F.mse_loss(pred, y) + F.l1_loss(pred, y)
        #return (1 - weight)*F.mse_loss(pred, y) + weight*F.l1_loss(pred, y)



    ######################################
    # Train Model
    ######################################

    start_epoch = 0 # match number in weights file name

    model.to(device)
    ssim = ssim.to(device)#ssim.cpu()
    psnr = psnr.to(device)#psnr.cpu()
    val_dict = {}
    mb = master_bar(range(NUM_EPOCHS)[start_epoch:])
    for epoch in mb:
        epoch_frac = (float(epoch) / (float(NUM_EPOCHS) - 1.))
        print('Epoch: ' + str(epoch + 1) + ' / ' + str(NUM_EPOCHS))
        ssim_tot = 0
        psnr_tot = 0
        val_cnt = 0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            for b_idx, (X, y) in enumerate(progress_bar(dataloaders[phase], parent=mb)):
                #print('X: ' + str(X.shape) + ', y: ' + str(y.shape))
                X = X.to(device)  # [N, 1, H, W]
                y = y.to(device)  # [N, H, W]

                optim.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    prediction = torch.clamp(model(X), min=0., max=1)  # [N, 1, H, W]
                    loss = criterion(prediction, y, epoch_frac)

                    # ✍️ Log your loss for this step to wandb ✍️
                    log_dict = {f"{phase}/loss": loss}

                    # Validation metrics
                    if phase == 'val':
                        pred_val = prediction.detach()
                        y_val = y.detach()
                        #ssim_tot += ssim(pred_val, y_val).item()
                        #psnr_tot += psnr(pred_val, y_val).item()
                        val_cnt += 1

                    #if phase == 'train':
                    # ✍️ Occasionally log 10 images from this batch for inspection ✍️
                    if (b_idx % 100 == 0):
                        log_dict[f"{phase}/X"] = [wandb.Image(i) for i in X[:10]]
                        log_dict[f"{phase}/predictions"] = [wandb.Image(i) for i in prediction[:10]]
                        log_dict[f"{phase}/y"] = [wandb.Image(i) for i in y[:10]]

                    # ✍️ Log to W&B ✍️
                    wandb.log(log_dict)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optim.step()

                # statistics
                running_loss += loss.item() * X.size(0)

                # TEST 10 BATCHES  -  REMOVE
                if b_idx >= 100:
                    break

            epoch_loss = running_loss / dataset_lens[phase]
            wandb.log({f'{phase}/loss':epoch_loss})
            print(f'{phase} Loss: {epoch_loss:.4f}')

        print()

        val_dict['val/ssim'] = ssim_tot / val_cnt
        val_dict['val/psnr'] = psnr_tot / val_cnt
        wandb.log(val_dict)

        try:
            os.makedirs('model_params/'+PROJECT_NAME)
        except:
            pass

        # Save model state
        state_filepath = r'model_params/' + PROJECT_NAME + '/after_epoch_' + str(epoch+1) + '.pth'
        torch.save(model.state_dict(), state_filepath)

    wandb.run.finish()
