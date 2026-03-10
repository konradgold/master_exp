from utils.tensors import batch_tensor, unbatch_tensor
import torch
from torch import nn

class WeightedAggregate(nn.Module):
    def __init__(self, model, feat_dim, lifting_net=nn.Sequential(), idx=0):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        num_heads = 8
        self.feature_dim = feat_dim
        self.idx = idx

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()
   


    def forward(self, mvimages):
        B, V, T, C, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        # TODO: shift channel / time depending on backbone model
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))

        if aux.ndim == 4:
            aux = aux.view(B, -1, self.feature_dim)
            V = aux.shape[1]

        ##################### VIEW ATTENTION #####################

        # S = source length 
        # N = batch size
        # E = embedding dimension
        # L = target length

        aux = torch.matmul(aux, self.attention_weights)
        # Dimension S, E for two views (2,512)
        aux = aux.squeeze(2)
        aux = aux / (self.feature_dim ** 0.5)

        # Dimension N, S, E
        aux_t = aux.permute(0, 2, 1)

        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))

        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))

        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux
    
class EmbedMaxAggregate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, V, C = x.shape # Batch, Views, Channel, 

        pooled_view = torch.max(x.permute(0, 2, 1), dim=2)[0]
        assert pooled_view.shape == (B, C)
        return pooled_view.squeeze(), x

class EmbedAvgAggregate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, V, C = x.shape # Batch, Views, Channel, 

        pooled_view = torch.mean(x.permute(0, 2, 1), dim=2)
        assert pooled_view.shape == (B, C)
        return pooled_view.squeeze(), x

class EmbedWeightedAggregate(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()

        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()

    def forward(self, x):
        B, V, _ = x.shape # Batch, Views, Channel, 

        if x.ndim == 4:
            x = x.view(B, -1, self.feature_dim)
            V = x.shape[1]
        aux = x

        ##################### VIEW ATTENTION #####################

        # S = source length 
        # N = batch size
        # E = embedding dimension
        # L = target length

        aux = torch.matmul(aux, self.attention_weights)
        # Dimension S, E for two views (2,512)
        aux = aux.squeeze(2)
        aux = aux / (self.feature_dim ** 0.5)

        # Dimension N, S, E
        aux_t = aux.permute(0, 2, 1)

        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))

        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))

        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights

class ViewAvgAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux
    
class ShallowMVAggregate(nn.Module):
    def __init__(self, agr_type="max", feat_dim=400, return_attention=False):
        super().__init__()
        self.agr_type = agr_type
        self.return_attention = return_attention

        if self.agr_type == "max":
            self.aggregation_model = EmbedMaxAggregate()
        elif self.agr_type == "mean":
            self.aggregation_model = EmbedAvgAggregate()
        else:
            self.aggregation_model = EmbedWeightedAggregate(feat_dim=feat_dim)
        
        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 8)
        )
    
    def forward(self, mvimages):
        pooled_view, attention = self.aggregation_model(mvimages)

        pred_action = self.fc_action(pooled_view)
        pred_offence_severity = self.fc_offence(pooled_view)

        if pred_action.ndim == 1:
            pred_action = pred_action.unsqueeze(0)
        if pred_offence_severity.ndim == 1:
            pred_offence_severity = pred_offence_severity.unsqueeze(0)

        if self.return_attention:
            return pred_action, pred_offence_severity, attention
        else:
            return pred_action, pred_offence_severity
    
class EmbedMVAggregate(nn.Module):
    def __init__(self, agr_type="max", feat_dim=400, return_attention=False):
        super().__init__()
        self.return_attention = return_attention

        if agr_type == "max":
            self.aggregation_model = EmbedMaxAggregate()
        elif agr_type == "mean":
            self.aggregation_model = EmbedAvgAggregate()
        elif agr_type == "weighted":
            self.aggregation_model = EmbedWeightedAggregate(feat_dim=feat_dim)
        
        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, feat_dim//4),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim//4),
            nn.Linear(feat_dim//4, feat_dim//8),
            nn.ReLU(),
            nn.Linear(feat_dim//8, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim//4),
            nn.Linear(feat_dim//4, feat_dim//8),
            nn.ReLU(),
            nn.Linear(feat_dim//8, 8)
        )
    

    def forward(self, x):
        pooled_view, attention = self.aggregation_model(x)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)
        if pred_action.ndim == 1:
            pred_action = pred_action.unsqueeze(0)
        if pred_offence_severity.ndim == 1:
            pred_offence_severity = pred_offence_severity.unsqueeze(0)

        if self.return_attention:
            return pred_action, pred_offence_severity, attention
        else:
            return pred_action, pred_offence_severity

class MVAggregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential(), return_attention=False):
        super().__init__()
        self.agr_type = agr_type
        self.return_attention = return_attention

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim//2),
            nn.GELU(),
            nn.Linear(feat_dim//2, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim//2),
            nn.GELU(),
            nn.Linear(feat_dim//2, 4)
        )


        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim//2),
            nn.GELU(),
            nn.Linear(feat_dim//2, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        else:
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)
        if pred_action.ndim == 1:
            pred_action = pred_action.unsqueeze(0)
        if pred_offence_severity.ndim == 1:
            pred_offence_severity = pred_offence_severity.unsqueeze(0)
        if self.return_attention:
            return pred_action, pred_offence_severity, attention
        else:
            return pred_action, pred_offence_severity


HEAD_REGISTRY = {
    "shallow_mv_aggregate": ShallowMVAggregate,
    "embed_mv_aggregate": EmbedMVAggregate,
}

def get_head(model,agr_type="max", feat_dim=400, return_attention=False):
    if model in HEAD_REGISTRY:
        return HEAD_REGISTRY[model](agr_type=agr_type, feat_dim=feat_dim, return_attention=return_attention)
    else:
        raise ValueError(f"Unknown model: {model}")