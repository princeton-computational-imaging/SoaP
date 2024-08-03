import argparse
import commentjson as json
import numpy as np
import os

import tinycudann as tcnn

from utils import utils
from utils.utils import debatch

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

#########################################################################################################
################################################ DATASET ################################################
#########################################################################################################

class BundleDataset(Dataset):
    def __init__(self, args):
        bundle = dict(np.load(args.bundle_path, allow_pickle=True))
        utils.de_item(bundle)
        
        if not args.no_raw:
            raw_frames = torch.tensor(np.array([bundle[f'raw_{i}']['raw'] for i in range(bundle['num_raw_frames'])]).astype(np.int32))[None]  # B,T,H,W
        if args.no_shade_map or args.no_raw:
            pass # no shade map needed
        else:
            shade_map = torch.tensor(np.load(os.path.join(os.getcwd(), "data/shade_map.npy")))[None,None,:,:] # 1,1,H,W, compensation for lens shading
            raw_frames = raw_frames * shade_map
        
        self.motion = bundle['motion']
        if args.no_device_rotations:
            self.frame_timestamps = torch.tensor(np.linspace(0,1, bundle['num_rgb_frames']))
            self.motion_timestamps = torch.tensor(np.linspace(0,1, bundle['num_rgb_frames']))
            self.quaternions = torch.tensor(np.repeat([[0,0,0,1.0]], bundle['num_rgb_frames'], axis=0)).float()
        else:
            self.frame_timestamps = torch.tensor([bundle[f'raw_{i}']['timestamp'] for i in range(bundle['num_rgb_frames'])])
            self.motion_timestamps = torch.tensor(self.motion['timestamp'])
            self.quaternions = torch.tensor(self.motion['quaternion']) # T',4, has different timestamps from frames

        self.reference_quaternion = utils.multi_interp(self.frame_timestamps[0:1], self.motion_timestamps, self.quaternions) # quaternion at frame 0
        self.reference_rotation = utils.convert_quaternions_to_rot(self.reference_quaternion)
        
        self.processed_rgb_volume = torch.tensor(np.array([bundle[f'rgb_{i}']['rgb'] for i in range(bundle['num_rgb_frames'])]))
        self.processed_rgb_volume = (self.processed_rgb_volume[:,:,:,:3].permute(0,3,1,2)).float() # remove alpha, make: T,C,H,W
        self.processed_rgb_volume = self.processed_rgb_volume / self.processed_rgb_volume[0].max() # scale 0-1
        
        intrinsics_ratio = 1.0
        if args.no_phone_depth and not args.no_raw: # intrinsics from RGB, but img from RAW, rescale
            intrinsics_ratio = bundle['raw_0']['height'] / bundle['rgb_0']['height']
        elif not args.no_phone_depth and args.no_raw: # intrinsics from depth, img from processed RGB
            intrinsics_ratio = bundle['rgb_0']['height'] / bundle['raw_0']['height']

        if args.no_phone_depth:    
            self.intrinsics = torch.tensor(np.array([bundle[f'rgb_{i}']['intrinsics'] for i in range(bundle['num_rgb_frames'])])).float() # T,3,3
        else:
            self.intrinsics = torch.tensor(np.array([bundle[f'depth_{i}']['intrinsics'] for i in range(bundle['num_depth_frames'])]))
            
        self.intrinsics[:,:3,:2] = self.intrinsics[:,:3,:2] * intrinsics_ratio
            
        if args.no_raw: # use processed RGB
            self.rgb_volume = (self.processed_rgb_volume).float()
            self.rgb_volume = self.rgb_volume - self.rgb_volume.min()
            self.rgb_volume = self.rgb_volume/self.rgb_volume.max()
            
        else: # use minimally processed RAW
            self.rgb_volume = (utils.raw_to_rgb(raw_frames)).float() # T,C,H,W
            self.rgb_volume = self.rgb_volume - self.rgb_volume.min()
            self.rgb_volume = self.rgb_volume/self.rgb_volume.max()
            if args.dark: # cut off highlights for scaling (long-tail-distribution)
                self.rgb_volume = self.rgb_volume/np.percentile(self.rgb_volume, 98)
                self.rgb_volume = self.rgb_volume.clamp(0,1)
                                                                        
        self.reference_intrinsics = self.intrinsics[0:1]
        
        if args.no_phone_depth:
            self.depth_volume = torch.zeros(bundle['num_rgb_frames'], 1, 64, 64, dtype=torch.float32) # placeholder depth
        else:
            self.depth_volume = torch.tensor(np.array([bundle[f'depth_{i}']['depth'] for i in range(bundle['num_depth_frames'])]))
            self.depth_volume = 1/(self.depth_volume[:,:,:,None].permute(0,3,1,2)).float() # T,C,H,W; lidar has inverse depth
        
        T,C,H,W = self.rgb_volume.shape
        self.num_frames, self.img_channels, self.img_height, self.img_width = T,C,H,W
        
        self.point_batch_size = args.point_batch_size
        self.num_batches = args.num_batches

    def __len__(self):
        return self.num_batches  # arbitrary as we continuously generate random samples
        
    def __getitem__(self, idx):
        # create uniform u,v between 0.025 and 0.975 to preserve edges
        uv = torch.rand(self.point_batch_size, 2) * torch.tensor([[0.95,0.95]]) + torch.tensor([[0.025,0.025]])
        
        # t is time for all frames, looks like [0, 0,... 0, 1/41, 1/41, ..., 1/41, 2/41, 2/41, ..., 2/41, etc.]
        t = torch.linspace(0,1,self.num_frames).repeat_interleave(uv.shape[0])[:,None] # num_frames * point_batch_size, 1
        
        return self.sample_grid(uv, t, frame=0, sample_depth=True, sample_rgb=True, sample_processed_rgb=False)

    def sample_grid(self, uv, t, frame, sample_depth=False, sample_rgb=False, sample_processed_rgb=False):
        """ Return TUV grid, interpolated rotation, intrinsics, depth, rgb samples
        """
        
        lidar_samples, rgb_samples, rgb_processed_samples = -1, -1, -1
        
        # convert to frame times [0-1] -> (seconds)
        t_frame = torch.tensor(np.interp(t, np.linspace(0,1,len(self.frame_timestamps)), self.frame_timestamps)).squeeze()
        # grab linearly interpolated quaternions at those timestamps
        quaternions = utils.multi_interp(t_frame, self.motion_timestamps, self.quaternions)
        # grab linearly interpolated intrinsics at those timestamps
        intrinsics = utils.multi_interp(t_frame, self.frame_timestamps, self.intrinsics.view(-1,9)).reshape(-1,3,3)
        
        if sample_depth:
            # grid_sample uses coordinates [-1,1] whereas MLP uses [0,1], hence rescaling
            grid_uv = ((uv - 0.5) * 2)[None,:,None,:] # 1,point_batch_size,1,2
            lidar_samples = F.grid_sample(self.depth_volume[frame:frame+1], grid_uv, mode="bilinear", padding_mode="border", align_corners=True)
            lidar_samples = lidar_samples.squeeze()[:,None] # point_batch_size, C
        
        if sample_rgb:
            grid_uv = ((uv - 0.5) * 2)[None,:,None,:] # 1,point_batch_size,1,2
            rgb_samples = F.grid_sample(self.rgb_volume[frame:frame+1], grid_uv, mode="bilinear", padding_mode="border", align_corners=True)
            rgb_samples = rgb_samples.squeeze().permute(1,0) # point_batch_size, C
            
        if sample_processed_rgb:
            grid_uv = ((uv - 0.5) * 2)[None,:,None,:] # 1,point_batch_size,1,2
            rgb_processed_samples = F.grid_sample(self.processed_rgb_volume[frame:frame+1], grid_uv, mode="bilinear", padding_mode="border", align_corners=True)
            rgb_processed_samples = rgb_processed_samples.squeeze().permute(1,0) # point_batch_size, C

        return t, uv, quaternions, intrinsics, lidar_samples, rgb_samples, rgb_processed_samples

#########################################################################################################
################################################ MODELS #################$###############################
#########################################################################################################
    
class PlaneModel(pl.LightningModule):
    def __init__(self, depth):
        super().__init__()
        # ax + by + c
        self.plane_coefs = torch.nn.Parameter(data=torch.tensor([1/10,1/10,depth/5]), requires_grad=True)
        # increase effective learning rate of plane without custom lr scheduler
        self.scale_factor = torch.nn.Parameter(data=torch.tensor([5.0,5.0,5.0]), requires_grad=False)
        
    def forward(self, uv):
        uv_homogenous = torch.cat((uv, torch.ones_like(uv[:,:1])), dim=1)
        plane = uv_homogenous * self.plane_coefs * self.scale_factor
        return torch.sum(plane, dim=1, keepdims=True)
        
    
class LearnedRotationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rotation_betas = torch.nn.Parameter(data=torch.zeros(args.control_points_motion, 3, 1, dtype=torch.float32), requires_grad=True)
        
    def forward(self, quaternions, t):
        # use de casteljau algorithm for interpolation
        rotation_deltas = utils.de_casteljau(self.rotation_betas, t)
        rx, ry, rz = rotation_deltas[:,0], rotation_deltas[:,1], rotation_deltas[:,2]
        r1 = torch.ones_like(rx)
        
        # identity rotation eye(3) plus small rotational offsets
        rotations = torch.stack([torch.stack([ r1, -rz,  ry], dim=-1),
                                 torch.stack([ rz,  r1, -rx], dim=-1),
                                 torch.stack([-ry,  rx,  r1], dim=-1)], dim=-1)
        
        return rotations
    
    
class DeviceRotationModel(pl.LightningModule):
    def __init__(self, args, reference_rotation):
        super().__init__()
        self.args = args
        self.reference_rotation = reference_rotation
        self.rotation_betas = torch.nn.Parameter(data=torch.zeros(args.control_points_motion, 3, 1, dtype=torch.float32), requires_grad=True)

    def forward(self, quaternions, t):
        rotations = torch.inverse(self.reference_rotation) @ utils.convert_quaternions_to_rot(quaternions) # from gyro
        
        rotation_deltas = utils.de_casteljau(self.rotation_betas, t)
        rx, ry, rz = rotation_deltas[:,0], rotation_deltas[:,1], rotation_deltas[:,2]
        r0 = torch.zeros_like(rx)
        
        rotation_offsets = torch.stack([torch.stack([ r0, -rz,  ry], dim=-1),
                                        torch.stack([ rz,  r0, -rx], dim=-1),
                                        torch.stack([-ry,  rx,  r0], dim=-1)], dim=-1)
        
        return rotations + self.args.rotation_weight * rotation_offsets
        
        
class TranslationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.translation_betas = torch.nn.Parameter(data=torch.zeros(args.control_points_motion, 3, 1, dtype=torch.float32), requires_grad=True)

    def forward(self, t):
        return self.args.translation_weight * utils.de_casteljau(self.translation_betas, t)
    
class IntrinsicsModel(pl.LightningModule):
    def __init__(self, args, reference_intrinsics):
        super().__init__()
        self.args = args
        self.intrinsic_betas = torch.nn.Parameter(data=torch.zeros(args.control_points_intrinsics, 1, 1, dtype=torch.float32), requires_grad=True)
        self.focal = torch.nn.Parameter(data=torch.tensor([reference_intrinsics[0,0,0]]),  requires_grad=True)
        self.cy = reference_intrinsics[0,2,0]
        self.cx = reference_intrinsics[0,2,1]

    def forward(self, t):
        f_deltas = utils.de_casteljau(self.intrinsic_betas, t)
                
        cy = self.cy * torch.ones_like(t)
        cx = self.cx * torch.ones_like(t)
        f = (self.focal * torch.ones_like(t)) + f_deltas
        f0 = torch.zeros_like(t)
        f1 = torch.ones_like(t)
        
        intrinsics = torch.stack([torch.stack([f,  f0, cy], dim=-1),
                                  torch.stack([f0, f,  cx], dim=-1),
                                  torch.stack([f0, f0, f1], dim=-1)], dim=-1)
        return intrinsics.squeeze(dim=1)
        
#########################################################################################################
################################################ NETWORK ################################################
#########################################################################################################
    
class BundleMLP(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # load network configs
        with open(args.config_path_depth) as config_depth:
            config_depth = json.load(config_depth)
        with open(args.config_path_rgb) as config_rgb:
            config_rgb = json.load(config_rgb)
            
        self.args = args
                    
        self.encoding_depth = tcnn.Encoding(n_input_dims=2, encoding_config=config_depth["encoding"])
        self.network_depth = tcnn.Network(n_input_dims=self.encoding_depth.n_output_dims, n_output_dims=1, network_config=config_depth["network"])
        
        self.encoding_rgb = tcnn.Encoding(n_input_dims=2, encoding_config=config_rgb["encoding"])
        self.network_rgb = tcnn.Network(n_input_dims=self.encoding_rgb.n_output_dims, n_output_dims=3, network_config=config_rgb["network"])
        
        self.model_translation = TranslationModel(args)
        self.model_plane = PlaneModel(depth=1.0)
        
        self.mask = torch.ones(self.encoding_depth.n_output_dims, dtype=torch.float32)
        self.save_hyperparameters()
        
        bundle = BundleDataset(args)
        self.bundle = bundle
        self.rgb_volume = bundle.rgb_volume
        self.processed_rgb_volume = bundle.processed_rgb_volume
        self.reference_intrinsics = bundle.reference_intrinsics
        self.reference_rotation = bundle.reference_rotation
        
        if args.no_device_rotations: # learn rotations from scratch
            self.model_rotation = LearnedRotationModel(args)
        else: # use gyro data
            self.model_rotation = DeviceRotationModel(args, self.reference_rotation)
            
        self.model_intrinsics = IntrinsicsModel(args, self.reference_intrinsics)

    def sample_volume(self, uv, volume, frame=None):
        """ Grid sample from 2D image volume at coordinates (u,v)
            If frame=None, sample from all frames, else single frame
        """
        pbs = self.bundle.point_batch_size
        grid_uv = ((uv - 0.5) * 2)
        
        if frame is None:
            grid_uv = grid_uv.reshape(self.bundle.num_frames, pbs, 1, -1) # frames, pbs, 1, 2
            rgb_samples = F.grid_sample(volume, grid_uv, mode="bilinear", padding_mode="border", align_corners=True)
            rgb_samples = rgb_samples.squeeze().permute(0,2,1).reshape(pbs * self.bundle.num_frames, -1)
        else:
            grid_uv = grid_uv[None,:,None,:] # frames, pbs, 1, 2
            rgb_samples = F.grid_sample(volume[frame:frame+1], grid_uv, mode="bilinear", padding_mode="border", align_corners=True)
            rgb_samples = rgb_samples.squeeze().permute(1,0)
        
        return rgb_samples
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.99), eps=1e-15, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.args.gamma)
        return [optimizer], [scheduler]
        
    def forward(self, t, uv, quaternions, lidar_samples, rgb_samples):
        """ Forward model pass, estimate motion, implicit depth + image.
        """
        translation = self.model_translation(t)
        rotation = self.model_rotation(quaternions, t)
        
        uv_depth = self.encoding_depth(uv)
        uv_rgb = self.encoding_rgb(uv)

        mask = self.mask.to(self.device)[None,:]
        
        if self.args.allow_negative_depth: # no ReLUs, no clamps, depth can go hog-wild
            plane = self.model_plane(uv)
            depth = (plane - self.network_depth(uv_depth * mask))
        else: # clamp depth between 0.01 and 10, depth offset must be *in front* of plane
            plane = F.relu(self.model_plane(uv)).clamp(0.01, 10.0)
            depth = (plane - F.relu(self.network_depth(uv_depth * mask))).clamp(0.01, 10)
            
        if self.args.fixed_image: # just sample static reference frame
            rgb = rgb_samples
        else: # sample from RGB MLP
            rgb = F.relu(0.5 + self.network_rgb(uv_rgb)).float()

        return rgb, depth, plane, rotation, translation
    
    def reproject(self, t, uv, depth, rotation, translation, intrinsics):
        """ Reproject uv coordinates to new refererence frame
        """
        if self.args.no_intrinsics: # use learned model
            intrinsics = self.model_intrinsics(t)
            reference_intrinsics = self.model_intrinsics(torch.zeros_like(t)[0:1])
        else: # used stored intrinsics
            reference_intrinsics = self.reference_intrinsics
            
        uvz = torch.cat((uv, depth), dim=1)
        xyz = utils.uvz_to_xyz(uvz, reference_intrinsics, img_width=self.bundle.img_width, img_height=self.bundle.img_height)
        xyz = (torch.inverse(rotation) @ xyz[:,:,None])[:,:,0] + translation # project to query
        uvz_reprojected = utils.xyz_to_uvz(xyz, intrinsics, img_width=self.bundle.img_width, img_height=self.bundle.img_height)
        tuv_reprojected = torch.cat((t, uvz_reprojected[:,0:2]), dim=1)
        
        return tuv_reprojected
    
    def training_step(self, train_batch, batch_idx):
        N = self.bundle.num_frames
        pbs = self.args.point_batch_size
        
        t, uv, quaternions, intrinsics, lidar_samples, rgb_samples, _ = train_batch # collapse batch + point dimensions
        t, uv, quaternions, intrinsics, lidar_samples, rgb_samples = debatch(t), debatch(uv), debatch(quaternions), debatch(intrinsics), debatch(lidar_samples), debatch(rgb_samples)
        
        rgb, depth, plane, rotation, translation = self.forward(t, uv, quaternions, lidar_samples, rgb_samples)
        uv, rgb, depth, plane = uv.repeat(N,1), rgb.repeat(N,1), depth.repeat(N,1), plane.repeat(N,1)
                
        tuv_plane_reprojected = self.reproject(t, uv, plane, rotation, translation, intrinsics)
        tuv_depth_reprojected = self.reproject(t, uv, depth, rotation, translation, intrinsics)
                
        rgb_plane_reprojected = self.sample_volume(tuv_plane_reprojected[:,1:], self.rgb_volume) # sample all timesteps with u,v
        rgb_depth_reprojected = self.sample_volume(tuv_depth_reprojected[:,1:], self.rgb_volume)
        
        loss = 0.0

        # overall depth loss
        depth_rgb_loss = ((rgb/(rgb.detach() + 0.001)) - (rgb_depth_reprojected/(rgb.detach() + 0.001))) ** 2
        depth_rgb_loss = depth_rgb_loss.mean(dim=1, keepdims=True) # mean over RGB channels
        loss += depth_rgb_loss.mean()

        # plane-only loss
        plane_rgb_loss = ((rgb/(rgb.detach() + 0.001)) - (rgb_plane_reprojected/(rgb.detach() + 0.001))) ** 2
        plane_rgb_loss = plane_rgb_loss.mean(dim=1, keepdims=True) # mean over RGB channels

        # weighted plane loss
        plane_depth_loss = (depth/plane - 1) ** 2
        weighted_plane_depth_loss = plane_rgb_loss/(depth_rgb_loss + 0.001) * plane_depth_loss
        loss += self.args.plane_weight * weighted_plane_depth_loss.mean()
        
        self.log('loss', loss)
        return loss
    
    def make_grid(self, height, width, u_lims, v_lims):
        """ Create (u,v) meshgrid with size (height,width) extent (u_lims, v_lims)
        """
        u = torch.linspace(u_lims[0], u_lims[1], width)
        v = torch.linspace(v_lims[0], v_lims[1], height)
        u_grid, v_grid = torch.meshgrid([u, v], indexing="xy") # u/v grid
        return torch.stack((u_grid.flatten(), v_grid.flatten())).t()
        
    def generate_imgs(self, frame, height=960, width=720, u_lims=[0,1], v_lims=[0,1]):
        """ Produce reference images and depth maps for tensorboard/visualization
        """
        device = self.device
        uv = self.make_grid(height, width, u_lims, v_lims)
        t = torch.tensor(frame/(self.bundle.num_frames - 1)).repeat(uv.shape[0])[:,None] # num_points, 1
        
        batch = self.bundle.sample_grid(uv, t, frame, sample_depth=True, sample_rgb=True, sample_processed_rgb=True)
        batch = [elem.to(device) for elem in batch]
        t, uv, quaternions, intrinsics, lidar_samples, rgb_samples, rgb_processed_samples = batch
        
        rgb_raw = rgb_samples.reshape(height, width, 3).permute(2,0,1) # channel first
        rgb_processed = rgb_processed_samples.reshape(height, width, 3).permute(2,0,1) # channel first
        depth_lidar = lidar_samples.reshape(height, width)
        depth_lidar_img = utils.colorize_tensor(depth_lidar, vmin=lidar_samples.min(), vmax=lidar_samples.max(), cmap="RdYlBu")
            
        return rgb_raw, rgb_processed, depth_lidar, depth_lidar_img
    
    def generate_outputs(self, frame, height=960, width=720, u_lims=[0,1], v_lims=[0,1]):
        """ Use forward model to sample implicit image I(u,v), depth D(u,v) and raw/processed images
            at reprojected u,v, coordinates. Results should be aligned (sampled at (u',v'))
        """
        device = self.device
        uv = self.make_grid(height, width, u_lims, v_lims)
        t = torch.tensor(frame/(self.bundle.num_frames - 1)).repeat(uv.shape[0])[:,None] # num_points, 1
        
        batch = self.bundle.sample_grid(uv, t, frame, sample_depth=True, sample_rgb=True, sample_processed_rgb=True)
        batch = [elem.to(device) for elem in batch]
        t, uv, quaternions, intrinsics, lidar_samples, rgb_samples, rgb_processed_samples = batch
        
        with torch.no_grad():
            rgb, depth, plane, rotation, translation = self.forward(t, uv, quaternions, lidar_samples, rgb_samples)
            tuv_reprojected = self.reproject(t, uv, depth, rotation, translation, intrinsics)
            rgb_raw = self.sample_volume(tuv_reprojected[:,1:], self.rgb_volume, frame=frame)
            rgb_processed = self.sample_volume(tuv_reprojected[:,1:], self.processed_rgb_volume, frame=frame)
            
            rgb = rgb.reshape(height, width, 3).permute(2,0,1) # channel first
            rgb_raw = rgb_raw.reshape(height, width, 3).permute(2,0,1) # channel first
            rgb_processed = rgb_processed.reshape(height, width, 3).permute(2,0,1)
            
            depth = depth.reshape(height, width)
            depth_img = utils.colorize_tensor(depth, vmin=0, vmax=depth.max(), cmap="RdYlBu")
            
        return rgb, rgb_raw, rgb_processed, depth, depth_img
    
#########################################################################################################
############################################### VALIDATION ##############################################
#########################################################################################################
        
class ValidationCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_epoch_start(self, trainer, model):
        args = model.args
        coef = ((model.current_epoch/model.args.max_epochs) * args.mask_k_max) + ((1 - model.current_epoch/model.args.max_epochs) * args.mask_k_min)
        model.mask = torch.sigmoid(torch.linspace(args.mask_k_max, coef, len(model.mask)))
        print("Mask mean:", model.mask.mean())
        
        # let plane train on its own for 10 epochs
        if model.current_epoch == 10:
            # start training depth
            model.encoding_depth.requires_grad_(True)
            model.encoding_depth.train(True)
            model.network_depth.requires_grad_(True)
            model.network_depth.train(True)
            if args.no_intrinsics:
                model.model_intrinsics.requires_grad_(True)
                model.model_intrinsics.train(True)

        for i, frame in enumerate([0]): # can sample more frames
            rgb, rgb_raw, rgb_processed, depth, depth_img = model.generate_outputs(frame)
            model.logger.experiment.add_image(f'pred/{i}_rgb', rgb, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_raw', rgb_raw, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_processed', rgb_processed, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_depth', depth_img, global_step=trainer.global_step)
            
            if model.args.save_video: # save the evolution of the model
                if i == 0: # save first frame
                    np.save(f"video/{model.args.name}/{model.current_epoch}_depth.npy", depth.detach().cpu().numpy())
                    np.save(f"video/{model.args.name}/{model.current_epoch}_rgb.npy", rgb.detach().cpu().numpy())
            
            # zoomed images
            # rgb, rgb_raw, rgb_processed, depth, depth_img, depth_lidar, depth_lidar_img = model.generate_imgs(frame, u_lims=[0.4,0.6], v_lims=[0.4,0.6])
            # model.logger.experiment.add_image(f'pred/{i}_rgb_zoom', rgb, global_step=trainer.global_step)
            # model.logger.experiment.add_image(f'pred/{i}_depth_zoom', depth_img, global_step=trainer.global_step)
            
    def on_train_start(self, trainer, model):
        pl.seed_everything(42)
        
        # pl doesn't put non-parameters on the right device
        model.rgb_volume = model.rgb_volume.to(model.device)
        model.processed_rgb_volume = model.processed_rgb_volume.to(model.device)
        model.reference_intrinsics = model.reference_intrinsics.to(model.device)
        if not model.args.no_device_rotations:
            model.model_rotation.reference_rotation = model.model_rotation.reference_rotation.to(model.device)
        model.model_intrinsics.focal = model.model_intrinsics.focal.to(model.device)
        
        model.logger.experiment.add_text("args", str(model.args))
        
        rgb_raw, rgb_processed, depth_lidar, depth_lidar_img = model.generate_imgs(0)
        model.logger.experiment.add_image('gt/lidar', depth_lidar_img, global_step=trainer.global_step)
        
        for i, frame in enumerate([0]):
            rgb_raw, rgb_processed, depth_lidar, depth_lidar_img = model.generate_imgs(frame)
            model.logger.experiment.add_image(f'gt/{i}_rgb_raw', rgb_raw, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'gt/{i}_rgb_processed', rgb_processed, global_step=trainer.global_step)
            # zoomed images
            # rgb, rgb_raw, rgb_processed, depth, depth_img, depth_lidar, depth_lidar_img = model.generate_imgs(frame, u_lims=[0.4,0.6], v_lims=[0.4,0.6])
            # model.logger.experiment.add_image(f'gt/{i}_rgb_raw_zoom', rgb_raw, global_step=trainer.global_step)
            # model.logger.experiment.add_image(f'gt/{i}_rgb_processed_zoom', rgb_processed, global_step=trainer.global_step)
        
        if model.args.save_video:
            os.makedirs(f"video/{model.args.name}", exist_ok=True)
            
    def on_train_end(self, trainer, model):
        checkpoint_dir = os.path.join("checkpoints", args.name, "last.ckpt")
        trainer.save_checkpoint(checkpoint_dir)

if __name__ == "__main__":
    
    # argparse
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--point_batch_size', type=int, default=1024, help="Number of points to sample per dataloader index.")
    parser.add_argument('--num_batches', type=int, default=256, help="Number of training batches.")
    parser.add_argument('--no_shade_map', action='store_true', help="Don't use shade map, useful for low-light captures.")
    parser.add_argument('--no_raw', action='store_true', help="No RAW data available, use RGB volume instead.")
    parser.add_argument('--no_device_rotations', action='store_true', help="Learn rotations from scratch, useful if no gyro data available.")
    parser.add_argument('--no_intrinsics', action='store_true', help="Learn camera intrinsics from scratch, useful if no camera intrinsics available.")
    parser.add_argument('--no_phone_depth', action='store_true', help="No phone depth data in bundle.")
    parser.add_argument('--allow_negative_depth', action='store_true', help="Allow negative depth solutions, useful for weird or digitally stabilized data.")
    parser.add_argument('--dark', action='store_true', help="Low-light capture, automatically also turns off shade map.")
    
    # model
    parser.add_argument('--control_points_motion', type=int, default=21, help="Spline control points for translation/rotation model.")
    parser.add_argument('--control_points_intrinsics', type=int, default=4, help="Spline control points for intrinsics model.")
    parser.add_argument('--config_path_depth', type=str, default="config/config_depth.json", help="Depth model config.")
    parser.add_argument('--config_path_rgb', type=str, default="config/config_rgb.json", help="RGB model config.")
    parser.add_argument('--plane_weight', type=float, default=1e-4, help="Depth regularization.")
    parser.add_argument('--rotation_weight', type=float, default=1e-1, help="Scale learned rotation.")
    parser.add_argument('--translation_weight', type=float, default=1e-1, help="Scale learned translation.")
    parser.add_argument('--mask_k_min', type=float, default=-100, help="Mask weight evolution parameter.")
    parser.add_argument('--mask_k_max', type=float, default=100, help="Mask weight evolution parameter.")
    parser.add_argument('--fixed_image', action='store_true', help="Fix I(u,v) to be the zero-th frame during training.")

    
    # training
    parser.add_argument('--bundle_path', type=str, required=True, help="Path to frame_bundle.npz")
    parser.add_argument('--name', type=str, required=True, help="Experiment name for logs and checkpoints.")
    parser.add_argument('--max_epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--gamma', type=float, default=0.98, help="Learning rate decay gamma.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--save_video', action='store_true', help="Store training outputs at each epoch for visualization.")


    args = parser.parse_args()
    if args.dark:
        args.no_shade_map = True
    
    print(args)
    
    # dataset
    bundle_dataset = BundleDataset(args)
    train_loader = DataLoader(bundle_dataset, batch_size=1, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)

    # model
    model = BundleMLP(args)
    # let plane train on its own at the start
    model.network_depth.requires_grad_(False)
    model.encoding_depth.requires_grad_(False)
    model.model_intrinsics.requires_grad_(False)

    # training
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join("checkpoints", args.name), save_top_k=1, save_last=True, monitor="loss")
    lr_callback = pl.callbacks.LearningRateMonitor()
    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=args.name, name="lightning_logs")
    validation_callback = ValidationCallback()
    trainer = pl.Trainer(accelerator="auto", strategy="auto", max_epochs=args.max_epochs,
                         logger=logger, callbacks=[validation_callback, lr_callback], enable_checkpointing=False)
    trainer.fit(model, train_loader)
