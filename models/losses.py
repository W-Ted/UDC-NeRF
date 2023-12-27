import torch
from torch import nn


class ColorLoss_Guidance(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        targets = batch["rgbs"].view(-1, 3)
        mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)
        
        loss = self.loss(inputs["rgb_fine"][mask], targets[mask]).mean()
        if "rgb_coarse" in inputs:
            loss += self.loss(inputs["rgb_coarse"][mask], targets[mask]).mean()
        return self.coef * loss


class ColorLoss_Fused(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        targets = batch["rgbs"].view(-1, 3)
        mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)
        
        loss = self.loss(inputs["rgb_instance_fine"][mask], targets[mask]).mean()
        if "rgb_instance_coarse" in inputs:
            loss += self.loss(inputs["rgb_instance_coarse"][mask], targets[mask]).mean()

        return self.coef * loss


class ColorLoss_Merged(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        targets = batch["rgbs"].view(-1, 3)
        mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)
        
        loss = self.loss(inputs["rgb_merged_fine"][mask], targets[mask]).mean()
        if "rgb_merged_coarse" in inputs:
            loss += self.loss(inputs["rgb_merged_coarse"][mask], targets[mask]).mean()

        return self.coef * loss


class ColorLoss_Unmasked(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        valid_mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)

        targets = batch["rgbs"].view(-1, 3) # hw,3
        rgb_unmasked_fine = inputs["rgb_unmasked_fine"] # hw,3,k

        instance_mask = batch["instance_mask"]
        instance_mask_weight = batch["instance_mask_weight"]
        if len(instance_mask.shape) == 3 and instance_mask.shape[0] == 1:
            instance_mask = instance_mask[0] # hw,k
            instance_mask_weight = instance_mask_weight[0] # hw,k
        else:
            instance_mask = instance_mask # hw,k
            instance_mask_weight = instance_mask_weight

        P,K = instance_mask.shape

        loss = (
            self.loss(
            targets[:,:,None].expand([-1,-1,K])[valid_mask] * instance_mask[:,None,:].expand([-1,3,-1])[valid_mask], # hw,3,k
            rgb_unmasked_fine[valid_mask]
            )\
            * instance_mask[:,None,:].expand([-1,3,-1])[valid_mask]
        ).mean()


        if "rgb_unmasked_coarse" in inputs:
            rgb_unmasked_coarse = inputs["rgb_unmasked_coarse"] # hw,3,k
            loss += (self.loss(
                targets[:,:,None].expand([-1,-1,K])[valid_mask] * instance_mask[:,None,:].expand([-1,3,-1])[valid_mask], # hw,3,k
                rgb_unmasked_coarse[valid_mask]
            ) \
            * instance_mask[:,None,:].expand([-1,3,-1])[valid_mask]
            ).mean()

        return self.coef * loss


class ColorLoss_Unmasked_Ps(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        valid_mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)

        targets_ps = batch["rgbs_ps"].view(-1, 3) # hw,3
        rgb_unmasked_fine = inputs["rgb_unmasked_fine"] # hw,3,k

        instance_mask = batch["instance_mask"]
        instance_mask_weight = batch["instance_mask_weight"]
        if len(instance_mask.shape) == 3 and instance_mask.shape[0] == 1:
            instance_mask = instance_mask[0] # hw,k
            instance_mask_weight = instance_mask_weight[0] # hw,k
        else:
            instance_mask = instance_mask # hw,k
            instance_mask_weight = instance_mask_weight

        loss = (
            self.loss(
            targets_ps[:,:,None].expand([-1,-1,1])[valid_mask] * (~instance_mask)[:,None,0:1].expand([-1,3,-1])[valid_mask], # hw,3,1
            rgb_unmasked_fine[:,:,0:1][valid_mask]
            )\
            * (~instance_mask)[:,None,0:1].expand([-1,3,-1])[valid_mask] # object region, version=1
        ).mean()


        if "rgb_unmasked_coarse" in inputs:
            rgb_unmasked_coarse = inputs["rgb_unmasked_coarse"] # hw,3,k

            loss += (self.loss(
                targets_ps[:,:,None].expand([-1,-1,1])[valid_mask] * (~instance_mask)[:,None,0:1].expand([-1,3,-1])[valid_mask], # hw,3,1
                rgb_unmasked_coarse[:,:,0:1][valid_mask]
            ) \
            * (~instance_mask)[:,None,0:1].expand([-1,3,-1])[valid_mask]
            ).mean()

        return self.coef * loss


class RegLoss_ModiSigma(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only
        self.target = 1e-5


    def forward(self, inputs, batch):

        loss = 0
        if "3D_sigmas_modi_coarse" in inputs and len(inputs["3D_sigmas_modi_coarse"])>0:
            loss = self.loss(
                torch.clamp(inputs["3D_sigmas_modi_coarse"],0),
                self.target * torch.ones_like(inputs["3D_sigmas_modi_coarse"])
            ).mean()
        
        if "3D_sigmas_modi_fine" in inputs and len(inputs["3D_sigmas_modi_fine"])>0:
            loss += self.loss(
                torch.clamp(inputs["3D_sigmas_modi_fine"],0),
                self.target * torch.ones_like(inputs["3D_sigmas_modi_fine"])
            ).mean()
        
        return self.coef * loss


class OpacityLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, inputs, batch):
        valid_mask = batch["valid_mask"].view(-1)
        if valid_mask.sum() == 0:  # skip when mask is empty
            print('DEBUG valid mask not valid')
            return None

        instance_mask = batch["instance_mask"]
        instance_mask_weight = batch["instance_mask_weight"]

        if len(instance_mask.shape) == 3 and instance_mask.shape[0] == 1:
            instance_mask = instance_mask[0][valid_mask,:]
            instance_mask_weight = instance_mask_weight[0][valid_mask,:]
        else:
            instance_mask = instance_mask[valid_mask,:]
            instance_mask_weight = instance_mask_weight[valid_mask,:]
        
        instance_mask_weight_bg = instance_mask_weight[:,0:1]
        instance_mask_weight_bg[instance_mask_weight_bg==instance_mask_weight_bg.min()] = 0

        loss = (
            self.loss(
                torch.clamp(inputs["opacity_unmasked_fine"][valid_mask,:], 0, 1)[:,1:],
                instance_mask.float()[:,1:],
            )
            * instance_mask_weight[:,1:]
        ).mean()

        loss += (
            self.loss(
                torch.clamp(inputs["opacity_unmasked_fine"][valid_mask,:], 0, 1)[:,0:1],
                instance_mask.float()[:,0:1],
            )
            * instance_mask_weight_bg
        ).mean()


        if "opacity_unmasked_coarse" in inputs:
            loss += (
                self.loss(
                    torch.clamp(inputs["opacity_unmasked_coarse"][valid_mask,:], 0, 1)[:,1:],
                    instance_mask.float()[:,1:],
                )
                * instance_mask_weight[:,1:]
            ).mean()

            loss += (
            self.loss(
                torch.clamp(inputs["opacity_unmasked_coarse"][valid_mask,:], 0, 1)[:,0:1],
                instance_mask.float()[:,0:1],
            )
                * instance_mask_weight_bg
            ).mean()

        return self.coef * loss



class UDCLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.color_loss_guidance = ColorLoss_Guidance(self.conf["color_loss_guidance_weight"])
        self.color_loss_unmasked = ColorLoss_Unmasked(self.conf["color_loss_unmasked_weight"])
        self.color_loss_unmasked_ps = ColorLoss_Unmasked_Ps(self.conf["color_loss_unmasked_ps_weight"])
        self.color_loss_fused = ColorLoss_Fused(self.conf["color_loss_fused_weight"])

        self.reg_loss_modisigam = RegLoss_ModiSigma(self.conf["reg_loss_modisigam_weight"])
        self.opacity_loss = OpacityLoss(self.conf["opacity_loss_weight"])
        self.color_loss_merged = ColorLoss_Merged(self.conf["color_loss_merged_weight"])

    def forward(self, inputs, batch, epoch=-1):
        loss_dict = dict()

        loss_dict["color_loss_guidance"] = self.color_loss_guidance(inputs, batch)
        loss_dict["color_loss_fused"] = self.color_loss_fused(inputs, batch)
        loss_dict["color_loss_unmasked"] = self.color_loss_unmasked(inputs, batch)
        if self.conf["color_loss_unmasked_ps_weight"] != 0:
            loss_dict["unmasked_ps_color_loss"] = self.color_loss_unmasked_ps(inputs, batch)
        if self.conf["reg_loss_modisigam_weight"] != 0:
            loss_dict["reg_loss_modisigam"] = self.reg_loss_modisigam(inputs, batch)

        loss_dict["opacity_loss"] = self.opacity_loss(inputs, batch)
        loss_dict["color_loss_merged"] = self.color_loss_merged(inputs, batch)

        # remove unused loss
        loss_dict = {k: v for k, v in loss_dict.items() if v != None}
        loss_sum = sum(list(loss_dict.values()))

        # recover loss to orig scale for comparison
        for k, v in loss_dict.items():
            if f"{k}_weight" in self.conf: # color_loss_weight: 1.0
                loss_dict[k] /= self.conf[f"{k}_weight"]

        return loss_sum, loss_dict


def get_udcloss(config):
    return UDCLoss(config.loss)

