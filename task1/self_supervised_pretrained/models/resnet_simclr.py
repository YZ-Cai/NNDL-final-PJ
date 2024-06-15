import torch
import torch.nn as nn
import torchvision.models as models
from ..exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, mode='train'):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        if mode == 'train':
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model


    def forward(self, x):
        return self.backbone(x)
    
    
    def load_for_testing(self, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']

        # remove prefix
        for k in list(state_dict.keys()):
            if k.startswith('backbone.') and not k.startswith('backbone.fc'):
                state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
            
        # load weights
        self.backbone.load_state_dict(state_dict, strict=False)
        
        # freeze all layers but the last fc
        for name, param in self.backbone.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
