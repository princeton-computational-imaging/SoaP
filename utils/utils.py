import matplotlib.pyplot as plt
import numpy as np
import torch

def de_casteljau(betas, t):
    """ castle interpolation, for knights
        see: https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
        assumes t in [0,1]
    """
    t = t[None,None,:,0] # 1,1,T
    
    out = betas.clone()
    N = betas.shape[0] # number of points
    for i in range(1, N):
        out = out[:-1,:] * (1-t) + out[1:,:] * t
    return out.squeeze(0).permute(1,0)

def uvz_to_xyz(uvz, intrinsics, img_width, img_height):
    """ Get xyz coordinates in meters from uv coordinates [0-1]
     iPhone poses are right-handed system, +x is right towards power button, +y is up towards front camera, +z is towards user's face
     images are opencv convention right-handed, x to the right, y down, and z into the world (away from face)
    """
    u = uvz[:,0:1] * img_width
    v = uvz[:,1:2] * img_height
    z = uvz[:,2:3]
    
    # intrinsics are for landscape sensor, top row: y, middle row: x, bottom row: z
    fy, cy, fx, cx = intrinsics[:,0,0,None], intrinsics[:,2,0,None], intrinsics[:,1,1,None], intrinsics[:,2,1,None]
    
    x = (u - cx) * (z/fx)
    y = (v - cy) * (z/fy)

    # rotate around the camera's x-axis by 180 degrees
    # now point cloud is in y up and z towards face convention
    y = -y
    z = -z
                     
    # match pose convention (y,x,z)
    return torch.cat((x,y,z), dim=1)

def xyz_to_uvz(uvz, intrinsics, img_width, img_height):
    """ Get uv coordinates [0-1] from coordinates rays in meters
    """    
    fy, cy, fx, cx = intrinsics[:,0,0,None], intrinsics[:,2,0,None], intrinsics[:,1,1,None], intrinsics[:,2,1,None]
    x, y, z = uvz[:,0:1], uvz[:,1:2], uvz[:,2:3]
    
    # undo rotation from convert_px_rays_to_m
    y = -y
    z = -z
    
    u  = (x * (fx/z) + cx) / img_width
    v =  (y * (fy/z) + cy) / img_height
    
    return torch.cat((u,v,z), dim=1)

def convert_quaternions_to_rot(quaternions):
    """ Convert quaternions (xyzw) to 3x3 rotation matrices.
        Adapted from: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix
    """

    qx, qy, qz, qw = quaternions[:,0], quaternions[:,1], quaternions[:,2], quaternions[:,3]
    
    R00 = 2 * ((qw * qw) + (qx * qx)) - 1
    R01 = 2 * ((qx * qy) - (qw * qz))
    R02 = 2 * ((qx * qz) + (qw * qy))
     
    R10 = 2 * ((qx * qy) + (qw * qz))
    R11 = 2 * ((qw * qw) + (qy * qy)) - 1
    R12 = 2 * ((qy * qz) - (qw * qx))
     
    R20 = 2 * ((qx * qz) - (qw * qy))
    R21 = 2 * ((qy * qz) + (qw * qx))
    R22 = 2 * ((qw * qw) + (qz * qz)) - 1
     
    R = torch.stack([R00, R01, R02, R10, R11, R12, R20, R21, R22], dim=-1)
    R = R.reshape(-1,3,3)
               
    return R

def multi_interp(x, xp, fp):
    """ Simple extension of np.interp for independent
        linear interpolation of all axes of fp
    """
    if torch.is_tensor(fp):
        out = [torch.tensor(np.interp(x, xp, fp[:,i]), dtype=fp.dtype) for i in range(fp.shape[-1])]
        return torch.stack(out, dim=-1)
    else:
        out = [np.interp(x, xp, fp[:,i]) for i in range(fp.shape[-1])]
        return np.stack(out, axis=-1)

def raw_to_rgb(raw_frames):
    """ Convert RAW mosaic into three-channel RGB volume
        by only in-filling empty pixels.
        Returns volume of shape: (T, C, H, W)
    """ 
    
    B = raw_frames[:,:,0::2,1::2].float()
    G1 = raw_frames[:,:,0::2,0::2].float()
    G2 = raw_frames[:,:,1::2,1::2].float()
    R = raw_frames[:,:,1::2,0::2].float()

    # Blue
    B_upsampled = torch.zeros_like(B).repeat(1,1,2,2)
    B_left = torch.roll(B, 1, dims=3)
    B_down = torch.roll(B, -1, dims=2)
    B_diag = torch.roll(B, [-1,1], dims=[2,3])

    B_upsampled[:,:,0::2,1::2] = B
    B_upsampled[:,:,0::2,0::2] = (B + B_left)/2
    B_upsampled[:,:,1::2,1::2] = (B + B_down)/2
    B_upsampled[:,:,1::2,0::2] = (B + B_down + B_left + B_diag)/4

    # Green
    G_upsampled = torch.zeros_like(G1).repeat(1,1,2,2)
    G1_right = torch.roll(G1, -1, dims=3)
    G1_down = torch.roll(G1, -1, dims=2)

    G2_left = torch.roll(G2, 1, dims=3)
    G2_up = torch.roll(G2, 1, dims=2)

    G_upsampled[:,:,0::2,0::2] = G1
    G_upsampled[:,:,0::2,1::2] = (G1 + G1_right + G2 + G2_up)/4
    G_upsampled[:,:,1::2,0::2] = (G1 + G1_down + G2 + G2_left)/4
    G_upsampled[:,:,1::2,1::2] = G2
    G_upsampled = G_upsampled

    # Red
    R_upsampled = torch.zeros_like(R).repeat(1,1,2,2)
    R_right = torch.roll(R, -1, dims=3)
    R_up = torch.roll(R, 1, dims=2)
    R_diag = torch.roll(R, [1,-1], dims=[2,3])

    R_upsampled[:,:,1::2,0::2] = R
    R_upsampled[:,:,1::2,1::2] = (R + R_right)/2
    R_upsampled[:,:,0::2,0::2] = (R + R_up)/2
    R_upsampled[:,:,0::2,1::2] = (R + R_up + R_right + R_diag)/4

    rgb_volume = torch.concat([R_upsampled, G_upsampled, B_upsampled], dim=0).permute(1,0,2,3) # T, C, H, W
    
    return rgb_volume
    
def de_item(bundle):
    """ Call .item() on all dictionary items
        removes unnecessary extra dimension
    """

    bundle['motion'] = bundle['motion'].item()
    
    if 'num_rgb_frames' not in bundle:
        return # motion bundle
    
    for i in range(bundle['num_rgb_frames']):
        bundle[f'rgb_{i}'] = bundle[f'rgb_{i}'].item()
        
    for i in range(bundle['num_raw_frames']):
        bundle[f'raw_{i}'] = bundle[f'raw_{i}'].item()
        
    for i in range(bundle['num_depth_frames']):
        bundle[f'depth_{i}'] = bundle[f'depth_{i}'].item()
        
def debatch(x):
    """ Collapse batch and channel dimension together
    """

    if len(x.shape) <=1:
        raise Exception("This tensor is to small to debatch.")
    elif len(x.shape) == 2:
        return x.reshape(x.shape[0] * x.shape[1])
    else:
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
    
def colorize_tensor(value, vmin=None, vmax=None, cmap=None, colorbar=False, height=9.6, width=7.2):
    """ Convert tensor to 3 channel RGB array according to colors from cmap
        similar usage as plt.imshow
    """
    assert len(value.shape) == 2 # H x W
    
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(width,height)
    a = ax.imshow(value.detach().cpu(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_axis_off()
    if colorbar:
        cbar = plt.colorbar(a, fraction=0.05)
        cbar.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.close()
    
    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    
    return torch.tensor(img).permute(2,0,1).float()