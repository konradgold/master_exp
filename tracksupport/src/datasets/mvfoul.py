from random import randint

from torch.utils.data import Dataset
from utils.mv_foul_parser import label2vectormerge, clips2vectormerge
import torch
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from torchvision.io.video import read_video

class MultiViewDataset(Dataset):
    def __init__(self, path, start: int = 0, end: int = 125, fps: int = 25, split: str = "train", num_views: int = 5, mode:str="TCHW", num_frames=16):


        if split != 'Chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity,self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
            print(f"Distribution of offence severity in {split} set: {self.distribution_offence_severity}")
            print(f"Distribution of action in {split} set: {self.distribution_action}")
            print(f"Weights for offence severity in {split} set: {torch.sqrt(self.weights_offence_severity)}")
            print(f"Weights for action in {split} set: {torch.sqrt(self.weights_action)}")
        else:
            self.clips = clips2vectormerge(path, split, num_views, [])

        # INFORMATION ABOUT SELF.LABELS_OFFENCE_SEVERITY
        # self.labels_offence_severity => Tensor of size of the dataset. 
        # each element of self.labels_offence_severity is another tensor of size 4 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if it is not an offence, then the tensor is [1, 0, 0, 0]. 

        # INFORMATION ABOUT SELF.LABELS_ACTION
        # self.labels_action => Tensor of size of the dataset. 
        # each element of self.labels_action is another tensor of size 8 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if the action is a tackling, then the tensor is [1, 0, 0, 0, 0, 0, 0, 0]. 

        # INFORMATION ABOUT SLEF.CLIPS
        # self.clips => list of the size of the dataset
        # each element of the list is another list of size of the number of views. The list contains the paths to all the views of that particular action.

        # The offence_severity groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_offence_severity tensor
        # The type of action groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_action tensor
        
        self.split = split
        self.start = start
        self.end = end
        self.mode = mode
        self.num_views = num_views
        self.num_frames = num_frames

        self.factor = (end - start) / (((end - start) / 25) * fps)

        self.length = len(self.clips)


    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action, 
    def getWeights(self):
        return torch.sqrt(self.weights_offence_severity), torch.sqrt(self.weights_action), 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clips = self.clips[idx]

        for num_view in range(len(clips)):

            index_view = num_view

            if self.split.lower() == "train" and num_view > 0:
                index_view = randint(1, len(clips)-1)

            # As we use a batch size > 1 during training, we always randomly select two views even if we have more than two views.
            # As the batch size during validation and testing is 1, we can 


            final_frames = read_video(clips[index_view], pts_unit="sec", output_format=self.mode)[0][int(self.start):int(self.end)]
            if num_view == 0:
                videos = final_frames.unsqueeze(0)
            else:
                final_frames = final_frames.unsqueeze(0)
                videos = torch.cat((videos, final_frames), 0)
                if self.split.lower() == "train":
                    break
            
        return self.labels_offence_severity[idx], self.labels_action[idx], videos, self.number_of_actions[idx]


