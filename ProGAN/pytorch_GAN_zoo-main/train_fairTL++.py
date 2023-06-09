# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:02:17 2022

@author: https://github.com/facebookresearch/pytorch_GAN_zoo
"""

import numpy as np
import os
import torch
# from torchsummary import summary
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm 

import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
#-----Diff Code----------------------------------------------------
use_gpu = True if torch.cuda.is_available() else False

# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-512',
#                        pretrained=True, useGPU=use_gpu)

# this model outputs 256 x 256 pixel images
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-256',
#                        pretrained=True, useGPU=use_gpu)

#-----------------------------------------------------------------------

import torch.nn.functional as F

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    """
    fixed to take in density ratio estimates
    """
    # properly match up dimensions, and only reweight real examples
    # loss_real = torch.mean(F.relu(1. - dis_real))
    
    #Attempt 1
    weighted = F.relu(1. - dis_real)
    loss_real = torch.mean(weighted)
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    
    #Doesn't work
    # weighted = (1. - dis_real)
    # loss_real = torch.mean(weighted)
    # loss_fake = torch.mean(1. + dis_fake)
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    # with torch.no_grad():
    #   dis_fake_norm = torch.exp(dis_fake).mean()
    #   dis_fake_ratio = torch.exp(dis_fake) / dis_fake_norm
    # dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake)
    return loss

def PGAN(pretrained=False, *args, **kwargs):
    """
    Progressive growing model
    pretrained (bool): load a pretrained model ?
    model_name (string): if pretrained, load one of the following models
    celebaHQ-256, celebaHQ-512, DTD, celeba, cifar10. Default is celebaHQ.
    """
    from models.progressive_gan import ProgressiveGAN as PGAN
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = PGAN(useGPU=kwargs.get('useGPU', True),
                 storeAVG=True,
                 **kwargs['config'])

    # Web Download----------------------------------------------------------------------------------------------------------------
    # checkpoint = {"celebAHQ-256": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ_s6_i80000-6196db68.pth',
    #               "celebAHQ-512": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ16_december_s7_i96000-9c72988c.pth',
    #               "DTD": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/testDTD_s5_i96000-04efa39f.pth',
    #               "celeba": "https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s5_i83000-2b0acc76.pth"}
    # if pretrained:
    #     if "model_name" in kwargs:
    #         if kwargs["model_name"] not in checkpoint.keys():
    #             raise ValueError("model_name should be in "
    #                                 + str(checkpoint.keys()))
    #     else:
    #         print("Loading default model : celebaHQ-256")
    #         kwargs["model_name"] = "celebAHQ-256"
    #     state_dict = model_zoo.load_url(checkpoint[kwargs["model_name"]],
    #                                     map_location='cpu')
    #     model.load_state_dict(state_dict)
    #-------------------------------------------------------------------------------------------------------------------------------
    
    #Offline Download of pre-trained model --------------------------------------------------------------------------------------------------------------
    if pretrained and loadFromIndex==None:
        if "model_name" in kwargs:
            if kwargs["model_name"]=="celebAHQ-256":
                PATH="./saved_pth/celebaHQ_s6_i80000-6196db68.pth"
                model.load_state_dict(torch.load(PATH))
            elif kwargs["model_name"]=="celebAHQ-256_improved":
                PATH="./saved_pth/celebaHQ_s6_i80000-6196db68_improved.pth"
                model.load_state_dict(torch.load(PATH))
                # torch.load("../output/optimG_dict_%i.pth"%loadFromIndex)
            else: 
                print ("Error: path not available")
    else:
        PATH="../output/state_dict_%i.pth"%loadFromIndex
        model.load_state_dict(torch.load(PATH))
    
    return model

def train(model,GOptim,DOptim,trainloader,batchSize,numDperGUpdates,multiFeedbackD):
    #set all models and data to cude
    device = 'cuda'
    # G=G.to(device)
    # D=D.to(device)
    # dataX.to(device)
    # dataY.to(device)
    
    #Intiialize the optimizer
    GOptim.zero_grad()
    DOptim.zero_grad()
    # x = torch.split(dataX, batchSize)
    # y = torch.split(dataY, batchSize)
    
    for i, data in enumerate(trainloader):    
        for j in range(numDperGUpdates):
            DOptim.zero_grad()
            #Discriminate real Data
            Dreal=model.netD(data[0].to(device))
            
            #Discriminate fake Data
            noise, _ = model.buildNoiseData(batchSize)
            fake_images=model.netG(noise.to(device))
            Dfake=model.netD(fake_images)
            
            #Update discriminator
            D_loss_real, D_loss_fake = loss_hinge_dis(Dfake,Dreal)
            D_loss = (D_loss_real + D_loss_fake)     
            D_loss.backward()
            optimD.step()
            
        
        
        #Update Generator
        GOptim.zero_grad()
        fake_images=model.netG(noise.to(device))
        Dfake=model.netD(fake_images)
        if multiFeedback==0:
            G_loss = loss_hinge_gen(Dfake)    
        else:
            _lambda=0.8
            Dfake_Quality=multiFeedbackD(fake_images)
            G_loss = _lambda*loss_hinge_gen(Dfake)+(1-_lambda)*loss_hinge_gen(Dfake_Quality)
            
        G_loss.backward()
        optimG.step()
        
        
        
        
#Hyper Parameters      
lrG=2e-4
lrD=1e-4
batch_size=16
LPepoch=10
epochs=200
numDperGUpdates=2
LPFTSwitch=0
multiFeedback=0
loadFromIndex=None #None if using pre-trained (original) else input index
saveEvery=10
model_name='celebAHQ-256_improved'
SA="Black_Hair"
#0) Load Prepare D-ref dataset
path="../../Data"
# x=torch.load(os.path.join(path,"CelebAHQ_0.025000_Male_data.pt"))
x=torch.load(os.path.join(path,"CelebAHQ_0.050000_%s_data.pt"%SA))


# x=torch.from_numpy(np.load(os.path.join(path,"CelebAHQ_all_data.npz"))['x']) #Improved training
# y=torch.load(os.path.join(path,"CelebAHQ_0.025000_Male_labels.pt"))

#Pre-process x into [-1,1]
transformList = [#NumpyResize(size),
                 #NumpyToTensor(),
                 Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transforms = Transforms.Compose(transformList)
x=(x/255.0) #Normalise to [0,1] as per dataset.ImageFolder
x=transforms(x.float())

#Comments: Need to edit (now the range isn't between 0-1)
# x=transform(x.float())
trainData=torch.utils.data.TensorDataset(torch.tensor(x))
# trainData.dataset.transform=Transforms.Compose([ Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainloader = DataLoader(trainData, batch_size=batch_size,
                              shuffle=True)
#1) Load pre-trained model
# Load pre-trained model
model=PGAN(pretrained=True, model_name=model_name, loadFrom=loadFromIndex )
# Load second frozen discriminator if multi-feedback is used
if multiFeedback==1:
    multiFeedbackD=PGAN(pretrained=True, model_name=model_name, loadFrom=loadFromIndex ).netD
    for param in multiFeedbackD.parameters():
        param.requires_grad = False #Works
else:
    multiFeedbackD=None
    

#Optimizer
optimG = optim.Adam(model.netG.parameters(),lr=lrG)
optimD= optim.Adam(model.netD.parameters(),lr=lrD)

if loadFromIndex!=None:
    optimG.load_state_dict(torch.load("../output/optimG_dict_%i.pth"%loadFromIndex))
    optimD.load_state_dict(torch.load("../output/optimD_dict_%i.pth"%loadFromIndex))
#2) Evaluate debiased transfer-learning Model 

#LOad Fixed Noise
num_images = 56
noisePath="../output/savedNoise%i.pth"%num_images
if not os.path.isfile(noisePath):
    noise, _ = model.buildNoiseData(num_images)
    torch.save(noise,noisePath)
else:
    noise=torch.load(noisePath)
    print ("Save Z Loaded")

#Load old checkpoint
if loadFromIndex==None:
    start=0
else:
    start=loadFromIndex

for i in tqdm(range(start,epochs+1)):
    #Toggle LP-FT on and off (ablation)
    if LPFTSwitch==1:
        if i<LPepoch:
            mode="LP"
        else: 
            mode="FT"
    else:
        mode="FT"
        
    # Plot samples for illustration
    with torch.no_grad():
        generated_images = model.test(noise)

    if multiFeedback==1 and LPFTSwitch==1:
        pathPrefix="../output/fairTL++/%s/"%SA
    elif multiFeedback==1 and LPFTSwitch==0:
        pathPrefix="../output/fairTLwMF/%s/"%SA
    elif multiFeedback==0 and LPFTSwitch==1:
        pathPrefix="../output/fairTLwLPFT/%s/"%SA
    else:
        pathPrefix="../output/fairTL/%s/"%SA    
        
    if not os.path.isdir(pathPrefix):
        os.makedirs(pathPrefix)
        
    # let's plot these images using torchvision and matplotlib
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.imsave(pathPrefix+"generatedImages_%i.jpg"%i , grid.permute(1, 2, 0).cpu().numpy())
    # plt.show()
    
    # #save Model
    if i%saveEvery==0:
        model.save(pathPrefix+"state_dict_%i.pth"%i)
        torch.save(optimG.state_dict(), pathPrefix+ "optimG_dict_%i.pth"%i)
        torch.save(optimD.state_dict(), pathPrefix+ "optimD_dict_%i.pth"%i)
    
    
    del(generated_images)
    del(grid)
    
    if mode=="FT":
        for param in model.netG.parameters():
                param.requires_grad = True #Works
        for param in model.netD.parameters():
                param.requires_grad = True #Works
        train(model,optimG,optimD,trainloader,batch_size,numDperGUpdates,multiFeedbackD)
        
    elif mode=="LP":
        # for param in model.netG.parameters():
        #         param.requires_grad = True #Works
        for param in model.netG.parameters():
                param.requires_grad = True #Works
        
        freezeLayers=["module.scaleLayers.0.0.module.weight",
                    "module.scaleLayers.0.0.module.bias",
                    "module.scaleLayers.0.1.module.weight",
                    "module.scaleLayers.0.1.module.bias",
                    "module.scaleLayers.1.0.module.weight",
                    "module.scaleLayers.1.0.module.bias",
                    "module.scaleLayers.1.1.module.weight",
                    "module.scaleLayers.1.1.module.bias",
                    # "module.scaleLayers.2.0.module.weight",
                    # "module.scaleLayers.2.0.module.bias",
                    # "module.scaleLayers.2.1.module.weight",
                    # "module.scaleLayers.2.1.module.bias",
                    # "module.scaleLayers.3.0.module.weight",
                    # "module.scaleLayers.3.0.module.bias",
                    # "module.scaleLayers.3.1.module.weight",
                    # "module.scaleLayers.3.1.module.bias",
                    "module.fromRGBLayers.0.module.bias",
                    "module.fromRGBLayers.1.module.weight",
                    "module.fromRGBLayers.1.module.bias",
                    # "module.fromRGBLayers.2.module.weight",
                    # "module.fromRGBLayers.2.module.bias",
                    # "module.fromRGBLayers.3.module.weight",
                    # "module.fromRGBLayers.3.module.bias",
                    # "module.groupScaleZero.0.module.weight",
                    # "module.groupScaleZero.0.module.bias",
                    ]
        for name, param in model.netD.named_parameters():
            if name in freezeLayers:
                param.requires_grad = False
                print ("Froze: :"+name)
            else:
                param.requires_grad = True
        
        train(model,optimG,optimD,trainloader,batch_size,numDperGUpdates,multiFeedbackD)