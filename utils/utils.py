import os
import torch
import torchvision.transforms as transforms
import torchvision.models as torchmodels
import torch.nn.functional as F
import numpy as np
import shutil
import json

from utils import utils_detector
from utils.ResNet import ResNetCifar as ResNet
from dataset.dataloader import CustomDatasetFromImages, CustomDatasetFromImagesTest, CustomDatasetFromImagesTestFM, CustomDatasetFromImagesTestVIS
from constants import base_dir_groundtruth, base_dir_detections_cd, base_dir_detections_fd, base_dir_metric_cd, base_dir_metric_fd
from constants import num_windows, img_size_fd, img_size_cd
from vit_pytorch import ViT

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)
    return data

def xywh2xyxy(x):
    y = np.zeros(x.shape)
    y[:,0] = x[:, 0] - x[:, 2] / 2.
    y[:,1] = x[:, 1] - x[:, 3] / 2.
    y[:,2] = x[:, 0] + x[:, 2] / 2.
    y[:,3] = x[:, 1] + x[:, 3] / 2.
    return y

def get_detected_boxes(policy, file_dirs, metrics, set_labels):
    for index, file_dir_st in enumerate(file_dirs):
        counter = 0
        for xind in range(num_windows):
            for yind in range(num_windows):
                # ---------------- Read Ground Truth ----------------------------------
                outputs_all = []
                gt_path = '{}/{}_{}_{}.txt'.format(base_dir_groundtruth, file_dir_st, xind, yind)
                if os.path.exists(gt_path):
                    gt = np.loadtxt(gt_path).reshape([-1, 5])
                    targets = np.hstack((np.zeros((gt.shape[0], 1)), gt))
                    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                    # ----------------- Read Detections -------------------------------
                    if policy[index, counter] == 1:
                        preds_dir = '{}/{}_{}_{}'.format(base_dir_detections_fd, file_dir_st, xind, yind)
                        targets[:, 2:] *= img_size_fd
                        if os.path.exists(preds_dir):
                            preds = np.loadtxt(preds_dir).reshape([-1,7])
                            outputs_all.append(torch.from_numpy(preds))
                    else:
                        preds_dir = '{}/{}_{}_{}'.format(base_dir_detections_cd, file_dir_st, xind, yind)
                        targets[:, 2:] *= img_size_cd
                        if os.path.exists(preds_dir):
                            preds = np.loadtxt(preds_dir).reshape([-1,7])
                            outputs_all.append(torch.from_numpy(preds))
                    set_labels += targets[:, 1].tolist()
                    metrics += utils_detector.get_batch_statistics(outputs_all, torch.from_numpy(targets), 0.5)
                else:
                    continue
                counter += 1

    return metrics, set_labels

def read_offsets(image_ids, num_actions):
    offset_fd = torch.zeros((len(image_ids), num_actions)).cuda()
    offset_cd = torch.zeros((len(image_ids), num_actions)).cuda()
    for index, img_id in enumerate(image_ids):
        offset_fd[index, :] = torch.from_numpy(np.loadtxt('{}/{}'.format(base_dir_metric_fd, img_id)).flatten())
        offset_cd[index, :] = torch.from_numpy(np.loadtxt('{}/{}'.format(base_dir_metric_cd, img_id)).flatten())

    return offset_fd, offset_cd

def performance_stats(policies, rewards):
    # Print the performace metrics including the average reward, average number
    
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return reward, num_unique_policy, variance, policy_set

def performance_stats_search(policies, rewards):
    # Print the performace metrics including the average reward, average number
    
    policies = torch.cat(policies, 0)
    
    reward = sum(rewards)
    

    return reward


def compute_reward(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    reward = torch.zeros(int(targets.shape[0])).cuda()
    for sample_id in range(int(targets.shape[0])):
        
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1
        else:
            reward[sample_id] = -1
    
    return reward
def compute_reward_latest(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # for incorrect search reward is 0 instead of 1 (trial)
    reward = torch.zeros(int(targets.shape[0])).cuda()
    for sample_id in range(int(targets.shape[0])):
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1
        else:
            reward[sample_id] = 0 #-1
    
    return reward

def compute_reward_greedy(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    temp_re = torch.eq(targets.cuda(), policy).long()
    temp_re[temp_re==0] = -1
    reward = torch.sum(temp_re, dim=1).unsqueeze(1).float()
    return reward



def compute_reward_batch(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    reward = torch.zeros(int(targets.shape[0])).cuda()
    for sample_id in range(int(targets.shape[0])):
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1
        else:
            reward[sample_id] = 0
    
    return reward
        
    
def compute_reward_search(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    if (targets[0,int(policy)]== 0):
        reward = torch.tensor(-1).float()
    else:
        reward = torch.tensor(1).float()
    
    return reward.reshape(1)
def compute_reward_search_test(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    if (targets[0,int(policy)]== 0):
        reward = torch.tensor(0).float()
    else:
        reward = torch.tensor(1).float()
    
    return reward.reshape(1)
   
    
def compute_reward_test(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    
    temp_re = torch.mul(targets.cuda(), policy)
    target_found = torch.sum(temp_re)
    num_targets = torch.sum(targets.cuda())
    total_search = torch.sum(policy.cuda())
    
    return target_found, num_targets, total_search 
 
def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Scale(img_size),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Scale(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test

def get_dataset(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_chip300.csv', transform_train) 
    
    testset = CustomDatasetFromImagesTest(root+'train_chip300.csv', transform_test) 
    

    return trainset, testset

def get_dataset_DOTA_TTT(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_dota.csv', transform_train) 
    
    testset = CustomDatasetFromImagesTest(root+'train_dota.csv', transform_test) 
    

    return trainset, testset

def get_datasetFM(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_dota_ship.csv', transform_train) 
    
    testset = CustomDatasetFromImagesTestFM(root+'train_dota_ship.csv', transform_test) 

    return trainset, testset


def get_datasetVIS(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)

    # use first 70% data for training
    trainset = CustomDatasetFromImages(root+'train.csv', transform_train) 
    
    # use remaining 30% data for testing
    testset = CustomDatasetFromImagesTest(root+'train.csv', transform_test) 
    
    return trainset, testset

def get_datasetVISx(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_dota_ship.csv', transform_train) 
      
    testset = CustomDatasetFromImagesTestVIS(root+'train_dota_ship.csv', transform_test) 

    return trainset, testset

def get_datasetVIS_(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_dota.csv', transform_train) 
    
    
    testset = CustomDatasetFromImagesTestVIS(root+'train_dota.csv', transform_test) 
    

    return trainset, testset

def get_datasetTTA(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    
    trainset = CustomDatasetFromImages(root+'train_building_tta.csv', transform_train)
    
    testset = CustomDatasetFromImagesTestVIS(root+'train_building_tta.csv', transform_test)
    
    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(num_output):
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)

    return agent

### a base policy network architecture (* not VAS)
class Model_search(torch.nn.Module):
    def __init__(self):
        super(Model_search, self).__init__()
        self.agent = torchmodels.resnet34(pretrained=True)
        set_parameter_requires_grad(self.agent, False)
        num_ftrs = self.agent.fc.in_features    
        self.agent.fc = torch.nn.Linear(num_ftrs, 60)
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(96, 80), 
            torch.nn.ReLU(),
            torch.nn.Linear(80, 64), 
        )
         
    def forward(self, x, search_info):
        feat_ext = self.agent(x)
        inp_search_concat = torch.cat((feat_ext, search_info), dim=1)
        
        logits = self.linear_relu_stack(inp_search_concat)
        return logits

## VAS without remaining query budget information (number of grid = 48)
class Model_search_Arch(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 96 
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 48, 1)  
        self.pointwise = torch.nn.Conv2d(96, 3, 1, 1)    
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(588, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 48),   
        )

    def forward(self, x, search_info):
        feat_ext = self.agent(x)
        reduced_feat =  F.relu(self.conv1(feat_ext))
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)
        
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        out = combined_feat.view(combined_feat.size(0), -1)
        
        logits = self.linear_relu_stack(out)
        return logits

### FOR number of grid = 36 and TTT
class Model_search_Arch_diffbud_TTT(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_diffbud_TTT, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 96 
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 36, 1)  
        self.pointwise = torch.nn.Conv2d(73, 3, 1, 1)    
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(588, 72),   
            torch.nn.ReLU(),
            torch.nn.Linear(72, 36),    
        )
        self.decoder = torch.nn.Sequential(                 # b, 36, 14, 14
            torch.nn.ConvTranspose2d(36, 36, 3, stride=2),  
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(36, 24, 3, stride=2, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(24, 12, 2, stride=4, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(12, 3, 2, stride=2),  # b, 3, 448, 448
            #torch.nn.ReLU(True),
            torch.nn.Tanh()
        )

    def forward(self, x, search_info, query_left):
        feat_ext = self.agent(x)                       
        reduced_feat =  F.relu(self.conv1(feat_ext))   
        
        recon_inp = self.decoder(reduced_feat)        
        
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 14, 14)
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile, query_info_tile), dim=1)
        
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        out = combined_feat.view(combined_feat.size(0), -1)
        
        logits = self.linear_relu_stack(out)
        return logits, recon_inp    
    
#### FOR DOTA with n=64 AND FM    
class Model_search_Arch_diffbud(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_diffbud, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 96 
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 64, 1)  
        self.pointwise = torch.nn.Conv2d(129, 3, 1, 1)     
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(588, 72), 
            torch.nn.ReLU(),
            torch.nn.Linear(72, 64),   
        )

    def forward(self, x, search_info, query_left):
        feat_ext = self.agent(x)
        reduced_feat =  F.relu(self.conv1(feat_ext))
        
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 14, 14)
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile, query_info_tile), dim=1)
        
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        out = combined_feat.view(combined_feat.size(0), -1)
        
        logits = self.linear_relu_stack(out)
        return logits

# Number of grid = 30    
class Model_search_Arch_diffbudVIS(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_diffbudVIS, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60 
        ## Input feature extractor
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 30, 1) 
        
        self.pointwise = torch.nn.Conv2d(61, 3, 1, 1)
        # final MLP layer to transform combine representation to action space
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(588, 60),   
            torch.nn.ReLU(),
            torch.nn.Linear(60, 30),    
        )

    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)
        # feature squezing using 1x1 conv
        reduced_feat =  F.relu(self.conv1(feat_ext))
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 14, 14)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile, query_info_tile), dim=1)
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        return logits

# number of grid = 36
class Model_search_Arch_diffbudVISx(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_diffbudVISx, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 96 
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 36, 1)  
        
        self.pointwise = torch.nn.Conv2d(73, 3, 1, 1)     
        
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(588, 72),  
            torch.nn.ReLU(),
            torch.nn.Linear(72, 36),    
        )

    def forward(self, x, search_info, query_left):
        feat_ext = self.agent(x)
        reduced_feat =  F.relu(self.conv1(feat_ext))
        
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 14, 14)
        
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile, query_info_tile), dim=1)
        
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        out = combined_feat.view(combined_feat.size(0), -1)
        
        logits = self.linear_relu_stack(out)
        return logits

# other potential approach   
class Model_search_Arch_NEW(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_NEW, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 30, 1)  
        self.pointwise = torch.nn.Conv2d(60, 3, 1, 1)
        
        self.pointwise_p = torch.nn.Conv2d(60, 6, 1, 1)
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(588, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 30),
        )
        
        self.linear_relu_stack_p = torch.nn.Sequential(
            torch.nn.Linear(96, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 30),
        )

    def forward(self, x, search_info):
        feat_ext = self.agent(x)
        reduced_feat =  F.relu(self.conv1(feat_ext))
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)
        
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        out = combined_feat.view(combined_feat.size(0), -1)
        
        logits = F.sigmoid(self.linear_relu_stack(out)) 
        
        x_tile = logits.view(logits.shape[0], logits.shape[1], 1, 1).repeat(1, 1, 4, 4)
        
        s_f_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 4, 4)
        info_cat = torch.cat((x_tile, s_f_tile), dim=1)
        combined_feat_p = F.relu(self.pointwise_p(info_cat))
        
        flatten_p = combined_feat_p.view(combined_feat_p.size(0), -1)
        logit_p = self.linear_relu_stack_p(flatten_p)
        
        return logits, logit_p

### explainable policy  
class Model_search_Arch_NEW_1(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_NEW_1, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 30, 1)  # 1*1 convolution
        self.pointwise = torch.nn.Conv2d(60, 3, 1, 1)
        self.pointwise_1 = torch.nn.Conv2d(60, 30, 1, 1)
        
        self.pointwise_p = torch.nn.Conv2d(90, 3, 1, 1)
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(588, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 30),
        )
        
        self.linear_relu_stack_p = torch.nn.Sequential(
            torch.nn.Linear(96, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 30),
        )

    def forward(self, x, search_info):
        feat_ext = self.agent(x)
        reduced_feat =  F.relu(self.conv1(feat_ext)) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14) 
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1) 
        concat_feat = F.relu(self.pointwise_1(inp_search_concat))
        gap_concat = F.adaptive_avg_pool2d(concat_feat, (1, 1))
        
        pred_logit = F.sigmoid(gap_concat.view(gap_concat.size(0), -1))
        
        pred_tile = pred_logit.view(pred_logit.shape[0], pred_logit.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        info_cat = torch.cat((pred_tile, inp_search_concat), dim=1)
        combined_feat_p = F.relu(self.pointwise_p(info_cat))
        
        flatten_p = combined_feat_p.view(combined_feat_p.size(0), -1)
        logit_p = self.linear_relu_stack(flatten_p)
        
        return pred_logit, logit_p
                        

## policy network using multi-head attention
class Model_search_Arch_MHA(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_MHA, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 30, 1)  # 1*1 convolution
        self.pointwise = torch.nn.Conv2d(60, 3, 1, 1)
        self.v = ViT(
                image_size = 14,
                patch_size = 2,
                num_classes = 30,
                dim = 512,
                depth = 2,
                heads = 4,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1)


    def forward(self, x, search_info):
        feat_ext = self.agent(x)
        reduced_feat =  F.relu(self.conv1(feat_ext))
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)
        
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ### MHA
        logits = self.v(combined_feat)
        
        return logits
    

def build_model():
    #from ResNet import ResNetCifar as ResNet
    print('Building model...')

    def gn_helper(planes):
        return torch.nn.GroupNorm(8, planes)
    norm_layer = gn_helper

    net = ResNet(26, 1, channels=3, classes=30, norm_layer=norm_layer).cuda()
    return net
    
 