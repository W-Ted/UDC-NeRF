from re import T
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, reduce, repeat
from timm.models.layers.helpers import to_2tuple


class UDCNeRF(nn.Module):
    def __init__(
        self,
        model_config,
        coarse
    ):
        super(UDCNeRF, self).__init__()
        self.model_config = model_config
        self.use_voxel_embedding = False
        self.coarse = coarse
        # initialize neural model with config
        self.initialize_model(model_config)

    def initialize_model(self, model_config):
        # model config
        self.N_freq_xyz = model_config["N_freq_xyz"] # 10
        self.N_freq_dir = model_config["N_freq_dir"] # 4
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.in_channels_xyz = xyz_emb_size
        self.in_channels_dir = 3 + 3 * self.N_freq_dir * 2
        self.activation = nn.LeakyReLU(inplace=True)

        self.inst_D = model_config["inst_D"] # 4
        self.inst_W = model_config["inst_W"] # 256
        self.inst_W_code = model_config["inst_W_code"] # 64
        self.inst_skips = model_config["inst_skips"] # [2]
        self.inst_K = model_config["inst_K"] # 6

        # define obj code as learnable parameters
        self.obj_code_type = model_config['obj_code_type'] # 'param', 'embed'
        if self.obj_code_type == 'param':
            self.obj_code = nn.Parameter(torch.zeros(self.inst_K, self.inst_W_code), requires_grad=True)
        elif self.obj_code_type == 'embed':
            self.obj_code_embed = nn.Embedding(self.inst_K, self.inst_W_code)
            self.register_buffer('constant_ind', torch.tensor([i for i in range(self.inst_K)]))

        # involve the object codes by using MLP
        self.inst_channel_in = (
                self.in_channels_xyz
            )
        self.feat_code_fusion_layer = nn.Linear(self.inst_W+self.inst_W_code, self.inst_W)
        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        

        if not self.coarse: # fine model only 
            # bg branch
            i = 0
            # sigma head
            sigma_head = nn.Sequential(
                nn.Linear(self.inst_W, self.inst_W),
                self.activation,
                nn.Linear(self.inst_W, 1)
                )
            setattr(self, f"instance_sigma_head_{i+1}", sigma_head)

            # encoding final layer
            encoding_final = nn.Sequential(nn.Linear(self.inst_W, self.inst_W))
            setattr(self, f"instance_encoding_final_{i+1}", encoding_final)

            # dir encoding layer
            dir_encoding = nn.Sequential(
                nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W),
                self.activation,
                nn.Linear(self.inst_W, self.inst_W),
                self.activation,
                nn.Linear(self.inst_W, self.inst_W // 2),
                self.activation,
            )
            setattr(self, f"instance_dir_encoding_{i+1}", dir_encoding)

            # rgb head
            rgb_head = nn.Sequential( 
                nn.Linear(self.inst_W // 2, 3), 
                nn.Sigmoid()
                )
            setattr(self, f"instance_rgb_head_{i+1}", rgb_head)


            #### fg branches
            # sigma head
            i = 1 
            sigma_head = nn.Sequential(
                nn.Linear(self.inst_W, self.inst_W),
                self.activation,
                nn.Linear(self.inst_W, 1)
                )
            setattr(self, f"instance_sigma_head_{i+1}", sigma_head)

            # encoding final layer
            encoding_final = nn.Sequential(
                nn.Linear(self.inst_W, self.inst_W),
                )
            setattr(self, f"instance_encoding_final_{i+1}", encoding_final)

            # dir encoding layer
            dir_encoding = nn.Sequential(
                nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W),
                self.activation,
                nn.Linear(self.inst_W, self.inst_W),
                self.activation,
                nn.Linear(self.inst_W, self.inst_W // 2),
                self.activation,
            )
            setattr(self, f"instance_dir_encoding_{i+1}", dir_encoding)

            # rgb head
            rgb_head = nn.Sequential(
                nn.Linear(self.inst_W // 2, 3), 
                nn.Sigmoid()
                )
            setattr(self, f"instance_rgb_head_{i+1}", rgb_head)


        if not self.coarse or (self.coarse): # fine OR (coarse)
            # Scene branch
            # scene sigma head: 
            self.sigma = nn.Sequential(
                nn.Linear(self.inst_W, self.inst_W),
                self.activation,
                nn.Linear(self.inst_W, 1)
            )

            # scene final layer
            self.encoding_final = nn.Sequential(
                    nn.Linear(self.inst_W, self.inst_W),
                    )
            
            # scene dir layer
            self.dir_encoding = nn.Sequential(
                    nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W),
                    self.activation,
                    nn.Linear(self.inst_W, self.inst_W),
                    self.activation,
                    nn.Linear(self.inst_W, self.inst_W // 2),
                    self.activation,
                )

            # scene rgb head:
            self.rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())


    def forward(self, inputs, sigma_only=False):
        output_dict = {}
        input_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_) # one linear layer
        output_dict["sigma"] = sigma

        if sigma_only:
            return output_dict

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        output_dict["rgb"] = rgb

        return output_dict
    
    
    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"] # PxC1
        input_dir = inputs.get("emb_dir", None) # PxC2
        # N_rays = inputs["N_rays"]
        # N_samples = inputs["N_samples"]


        P = emb_xyz.shape[0]
        K = self.inst_K
        x_ = emb_xyz

 
        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([emb_xyz, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i+1}")(x_) # (P,x)

        if self.obj_code_type == 'embed':
            self.obj_code = self.obj_code_embed(self.constant_ind)
        x_ = x_[:,None,:].expand(P, self.inst_K, -1).flatten(0,1)
        obj_code = self.obj_code[None,...].expand(P, self.inst_K, -1).flatten(0,1)
        x_ = torch.cat([x_, obj_code], -1)
        x_ = self.feat_code_fusion_layer(x_) # (PK,2W) -> (PK,W)
        x_ = x_.reshape(P, self.inst_K, self.inst_W) # (P,K,W)

        
        if not self.coarse or self.coarse: # fine OR (coarse)
            # scene sigma
            scene_sigma = self.sigma(reduce(x_, 'p k c -> p c', 'mean'))
            output_dict["sigma"] = scene_sigma

            # scene rgb
            x_final = self.encoding_final(reduce(x_, 'p k c -> p c', 'mean'))
            dir_encoding_input = torch.cat([x_final, input_dir], -1)
            dir_encoding = self.dir_encoding(dir_encoding_input)
            scene_rgb = self.rgb(dir_encoding)
            output_dict["rgb"] = scene_rgb


        if not self.coarse: # fine model only
            # inst sigma
            inst_sigma_bg = getattr(self, f"instance_sigma_head_1")(x_[:,0:1,:]) # (P,1,1)
            inst_sigma_fg = getattr(self, f"instance_sigma_head_2")(x_[:,1:,:]) # (P,K-1,1)
            inst_sigma = torch.cat([inst_sigma_bg, inst_sigma_fg], 1)
            output_dict['unmasked_sigma'] = inst_sigma # (P,K,1)    

            # inst rgb
            inst_x_final_bg = getattr(self, f"instance_encoding_final_1")(x_[:,0,:])
            inst_dir_encoding_input_bg = torch.cat([inst_x_final_bg, input_dir], -1)
            inst_dir_encoding_bg = getattr(self, f"instance_dir_encoding_1")(inst_dir_encoding_input_bg)
            rgb_bg = getattr(self, f"instance_rgb_head_1")(inst_dir_encoding_bg) # (P,3)

            inst_x_final_fg = getattr(self, f"instance_encoding_final_2")(x_[:,1:,:].flatten(0,1))
            inst_dir_encoding_input_fg = torch.cat([inst_x_final_fg, input_dir[:,None,:].expand(P,(K-1),-1).flatten(0,1)], -1)
            inst_dir_encoding_fg = getattr(self, f"instance_dir_encoding_2")(inst_dir_encoding_input_fg)
            rgb_fg = getattr(self, f"instance_rgb_head_2")(inst_dir_encoding_fg) # (P(K-1),3)
            
            unmasked_rgb = torch.cat([rgb_bg.reshape(P,1,3), rgb_fg.reshape(P,(K-1),3)], 1)
            
            output_dict['unmasked_rgb'] = unmasked_rgb

        return output_dict

