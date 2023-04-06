import argparse
import numpy as np
import struct
from matplotlib import gridspec
import matplotlib.pyplot as plt
from glob import glob
import os
from os.path import join
from os.path import split
from natsort import natsorted
from skimage.transform import resize
import re
from tqdm import tqdm

""" Code to process depth/image/pose binaries the ios DepthBundleRecorder app into more useable .npz files.
    Tested for iPhone 12, 13, 14 pro.
    Usage: python ConvertBinaries.py -d folder_with_bundles
    Output: a folder processed_folder_with_bundles containing the processed depth bundles
"""

def cut(x): # return value after the ":"
    return x.split(":")[1]

def process_raw(npz_file, raw_name):
    global is_iphone12
    with open(raw_name, mode='rb') as file:
        raw = file.read()

    raw_split = raw.split(b"<BEGINHEADER>")
    num_raw_frames = 0

    for i, raw_frame in tqdm(enumerate(raw_split[1:])):
        if len(raw_frame) < 100: # skip weird outliers
            continue

        raw_header, raw_image = raw_frame.split(b"<ENDHEADER>")
        raw_header = re.sub("\[|\]|\(|\)|\s|\'", "", str(raw_header)) # Strip all delims but <> and commas
        raw_header = re.sub(r"\s+", "", raw_header) # Strip spaces
        rw = raw_header.split(",")

        frame_count = int(cut(rw[1])) # skip description
        timestamp = float(cut(rw[2]))
        height = int(cut(rw[3]))
        width = int(cut(rw[4]))
        ISO = int(cut(rw[6]))
        exposure_time = float(cut(rw[7]))
        aperture = float(cut(rw[8]))
        brightness = float(cut(rw[9]))
        shutter_speed = float(cut(rw[10]))
        black_level = int(cut(rw[13]))
        white_level = int(cut(rw[14]))
        

        raw_image = struct.unpack('H'* ((len(raw_image)//2)), raw_image)
        raw_image = np.reshape(raw_image, (height, width))
        raw_image = np.flip(raw_image.swapaxes(0,1), 1) # make vertical
        
        if is_iphone12:
            # Make everything same bayer format:
            # G B
            # R G
            A1 = raw_image[::2,1::2].copy()
            A2 = raw_image[1::2,::2].copy()
            raw_image[::2,1::2] = A2
            raw_image[1::2,::2] = A1

        raw_frame = {'frame_count' : frame_count,
                     'timestamp' : timestamp,
                     'height' : height,
                     'width' : width,
                     'ISO' : ISO,
                     'exposure_time' : exposure_time,
                     'aperture' : aperture,
                     'brightness' : brightness,
                     'shutter_speed' : shutter_speed,
                     'black_level' : black_level,
                     'white_level' : white_level,
                     'raw' : raw_image
        }

        num_raw_frames += 1
        npz_file[f'raw_{i}'] = raw_frame
    npz_file['num_raw_frames'] = num_raw_frames

def process_rgb(npz_file, rgb_name):
    global is_iphone12
    with open(rgb_name, mode='rb') as file:
        rgb = file.read()

    rgb_split = rgb.split(b"<BEGINHEADER>")
    num_rgb_frames = 0

    for i, rgb_frame in tqdm(enumerate(rgb_split[1:])):
        if len(rgb_frame) < 100: # skip weird outliers
            continue

        rgb_header, rgb_image = rgb_frame.split(b"<ENDHEADER>")
        rgb_header = re.sub("\[|\]|\(|\)|\s|\'", "", str(rgb_header)) # Strip all delims but <> and commas
        rgb_header = re.sub(r"\s+", "", rgb_header) # Strip spaces
        r = rgb_header.split(",")

        frame_count = int(cut(r[1])) # skip description
        timestamp = float(cut(r[2]))
        height = int(cut(r[3]))
        width = int(cut(r[4]))
        intrinsics = np.array([[cut(r[6]), r[7],  r[8]],
                                   [r[9],  r[10], r[11]],
                                   [r[12], r[13], r[14]]], dtype=np.float32)
        
        rgb_image = struct.unpack('B'* ((len(rgb_image))), rgb_image)
        
        try:
            rgb_image = np.reshape(rgb_image, (1440, 1920, 4)) # 12 extra bytes at the end of each row, mystery
            rgb_image = rgb_image[:,:,[2,1,0,3]] # cut extra bytes, go from BGRA to RGBA
            rgb_image = np.flip(rgb_image.swapaxes(0,1), 1) # make vertical
        except:
            raise Exception("RGB format not understood.")

        rgb_frame = {'frame_count' : frame_count,
                     'timestamp' : timestamp,
                     'height' : height,
                     'width' : width,
                     'intrinsics' : intrinsics,
                     'rgb' : rgb_image
        }

        num_rgb_frames += 1
        npz_file[f'rgb_{i}'] = rgb_frame
    npz_file['num_rgb_frames'] = num_rgb_frames

def process_depth(npz_file, depth_name):
    with open(depth_name, mode='rb') as file:
        depth = file.read()

    depth_split = depth.split(b"<BEGINHEADER>")
    num_depth_frames = 0

    for i, depth_frame in tqdm(enumerate(depth_split[1:])):
        if len(depth_frame) < 100: # skip weird outliers
            continue

        depth_header, depth_image = depth_frame.split(b"<ENDHEADER>")
        depth_header = re.sub("\[|\]|\(|\)|\s|\'", "", str(depth_header)) # Strip all delims but <> and commas
        depth_header = re.sub(r"\s+", "", depth_header) # Strip spaces
        d = depth_header.split(",")

        frame_count = int(cut(d[1])) # skip description
        timestamp = float(cut(d[2]))
        height = int(cut(d[3]))
        width = int(cut(d[4]))
        intrinsic_width = int(float(cut(d[6])))
        intrinsic_height = int(float(cut(d[7]))) # it has a decimal for some reason
        intrinsics = np.array([[cut(d[8]), d[9],  d[10]],
                                   [d[11], d[12], d[13]],
                                   [d[14], d[15], d[16]]], dtype=np.float32)
        lens_distortion = np.array([cut(d[17]), *d[18:59]], dtype=np.float32)
        lens_undistortion = np.array([cut(d[59]), *d[60:101]], dtype=np.float32) # 42 numbers, heh
        depth_accuracy = int(cut(d[101]))

        depth_image = struct.unpack('e'* ((len(depth_image)) // 2), depth_image)
        depth_image = np.reshape(depth_image, (height, width))
        depth_image = np.flip(depth_image.swapaxes(0,1), 1) # make vertical

        depth_frame = {'frame_count' : frame_count,
                       'timestamp' : timestamp,
                       'height' : height,
                       'width' : width,
                       'intrinsic_height' : intrinsic_height,
                       'intrinsic_width' : intrinsic_width,
                       'intrinsics' : intrinsics,
                       'lens_distortion' : lens_distortion,
                       'lens_undistortion' : lens_undistortion,
                       'depth_accuracy' : depth_accuracy,
                       'depth' : depth_image
        }

        num_depth_frames += 1
        npz_file[f'depth_{i}'] = depth_frame
    npz_file['num_depth_frames'] = num_depth_frames

def process_motion(npz_file, motion_name):
    with open(motion_name, mode='rb') as file:
        motion = str(file.read())

    motion_split = motion.split("<BEGINHEADER>")
    num_motion_frames = 0

    frame_count = []
    timestamp = []
    quaternion = []
    roll_pitch_yaw = []
    rotation_rate = []
    acceleration = []
    gravity = []

    for i, motion_frame in tqdm(enumerate(motion_split)):
        if len(motion_frame) < 100: # skip weird outliers
            continue

        motion_frame = motion_frame.strip().replace("<ENDHEADER>", "")
        motion_frame = re.sub("\[|\]|\(|\)|\s|\'", "", motion_frame) # Strip all delims but <> and commas
        motion_frame = re.sub(r"\s+", "", motion_frame) # Strip spaces
        m = motion_frame.split(",")

        frame_count.append(int(cut(m[0])))
        timestamp.append(float(cut(m[1])))
        # quaternion x,y,z,w
        quaternion.append(np.array([cut(m[2]), cut(m[3]), cut(m[4]), cut(m[5])], dtype=np.float32))
        rotation_rate.append(np.array([cut(m[6]), cut(m[7]), cut(m[8])], dtype=np.float32))
        roll_pitch_yaw.append(np.array([cut(m[9]), cut(m[10]), cut(m[11])], dtype=np.float32))
        acceleration.append(np.array([cut(m[12]), cut(m[13]), cut(m[14])], dtype=np.float32))
        gravity.append(np.array([cut(m[15]), cut(m[16]), cut(m[17])], dtype=np.float32))

        num_motion_frames += 1

    motion_frame = {'frame_count' : np.array(frame_count),
                'timestamp' : np.array(timestamp), 
                'quaternion' : np.array(quaternion), 
                'rotation_rate' : np.array(rotation_rate),
                'roll_pitch_yaw' : np.array(roll_pitch_yaw),
                'acceleration' : np.array(acceleration),
                'gravity' : np.array(gravity),
                'num_motion_frames': np.array(num_motion_frames)}


    npz_file["motion"] = motion_frame
    
def match_timestamps(npz_file):
    raw_timestamps = np.array([npz_file[f'raw_{i}']['timestamp'] for i in range(npz_file['num_raw_frames'])])
    raw_timestamps = np.around(raw_timestamps, 3)
    rgb_timestamps = np.array([npz_file[f'rgb_{i}']['timestamp'] for i in range(npz_file['num_rgb_frames'])])
    rgb_timestamps = np.around(rgb_timestamps, 3)
    depth_timestamps = np.array([npz_file[f'depth_{i}']['timestamp'] for i in range(npz_file['num_depth_frames'])])
    depth_timestamps = np.around(depth_timestamps, 3)
    assert (rgb_timestamps == depth_timestamps).all()
    
    matches = np.array([np.where(rgb_timestamps == raw_timestamps[i])[0][0] for i in range(npz_file['num_raw_frames'])])
    assert len(matches) == npz_file['num_raw_frames'] # all frames have a match
    
    for i in range(npz_file['num_raw_frames']):
        match_idx = matches[i]
        npz_file[f'rgb_{i}'] = npz_file[f'rgb_{match_idx}']
        npz_file[f'depth_{i}'] = npz_file[f'depth_{match_idx}']

    for i in range(npz_file['num_raw_frames'], npz_file['num_rgb_frames']):
        del npz_file[f'rgb_{i}']
        del npz_file[f'depth_{i}']

    npz_file['num_rgb_frames'] = npz_file['num_depth_frames'] = npz_file['num_raw_frames']
    
    raw_timestamps = np.array([npz_file[f'raw_{i}']['timestamp'] for i in range(npz_file['num_raw_frames'])])
    raw_timestamps = np.around(raw_timestamps, 3)
    rgb_timestamps = np.array([npz_file[f'rgb_{i}']['timestamp'] for i in range(npz_file['num_rgb_frames'])])
    rgb_timestamps = np.around(rgb_timestamps, 3)
    depth_timestamps = np.array([npz_file[f'depth_{i}']['timestamp'] for i in range(npz_file['num_depth_frames'])])
    depth_timestamps = np.around(depth_timestamps, 3)
    assert (rgb_timestamps == raw_timestamps).all()

def main():
    global is_iphone12
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=None, type=str, required=True, help='Data directory')
    parser.add_argument('-iphone12', action='store_true', help='Flag that this is an iPhone 12 to rotate bayer array.')
    args = parser.parse_args()
    is_iphone12 = args.iphone12
    
    if "bundle-" not in args.d:
        bundle_names = natsorted(glob(join(args.d, "bundle*")))
    else:
        bundle_names = [args.d]
    
    for bundle_name in bundle_names:
        print(f"Processing {split(bundle_name)[-1]}.")
        
        if "processed-" in bundle_name:
            continue # already processed, skip
        
        if "-motion" not in bundle_name:
            # Process image + depth bundle
            motion_name = join(bundle_name, "motion.bin") 
            rgb_name = join(bundle_name, "imageRGB.bin")
            raw_name = join(bundle_name, "imageRAW.bin")
            depth_name = join(bundle_name, "depth.bin") 
            
            save_path = join(split(bundle_name)[0], "processed-" + split(bundle_name)[1])
            os.makedirs(save_path, exist_ok=True)

            npz_file = {}
            
            process_depth(npz_file, depth_name)
            process_motion(npz_file, motion_name)
            process_rgb(npz_file, rgb_name)
            process_raw(npz_file, raw_name)
            match_timestamps(npz_file)

            # Save first frame preview
            fig = plt.figure(figsize=(14, 30)) 
            gs = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0, width_ratios=[1,1,1.12])
            ax1 = plt.subplot(gs[0,0])
            ax1.imshow(npz_file['rgb_0']['rgb'])
            ax1.axis('off')
            ax1.set_title("RGB")
            ax2 = plt.subplot(gs[0,1])
            ax2.imshow(npz_file['raw_0']['raw'], cmap="gray")
            ax2.axis('off')
            ax2.set_title("RAW")
            ax3 = plt.subplot(gs[0,2])
            d = ax3.imshow(npz_file['depth_0']['depth'], cmap="Spectral", vmin=0, vmax=5)
            ax3.axis('off')
            ax3.set_title("Depth")
            fig.colorbar(d, fraction=0.055, label="Depth [m]")
            plt.savefig(join(save_path, "frame_first.png"), bbox_inches='tight', pad_inches=0.05, facecolor='white')
            plt.close()

            # Save last frame preview
            fig = plt.figure(figsize=(14, 30)) 
            gs = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0, width_ratios=[1,1,1.12])
            ax1 = plt.subplot(gs[0,0])
            ax1.imshow(npz_file[f'rgb_{npz_file["num_raw_frames"] - 1}']['rgb'])
            ax1.axis('off')
            ax1.set_title("RGB")
            ax2 = plt.subplot(gs[0,1])
            ax2.imshow(npz_file[f'raw_{npz_file["num_raw_frames"] - 1}']['raw'], cmap="gray")
            ax2.axis('off')
            ax2.set_title("RAW")
            ax3 = plt.subplot(gs[0,2])
            d = ax3.imshow(npz_file[f'depth_{npz_file["num_raw_frames"] - 1}']['depth'], cmap="Spectral", vmin=0, vmax=5)
            ax3.axis('off')
            ax3.set_title("Depth")
            fig.colorbar(d, fraction=0.055, label="Depth [m]")
            plt.savefig(join(save_path, "frame_last.png"), bbox_inches='tight', pad_inches=0.05, facecolor='white')
            plt.close()

            # Save bundle
            np.savez(join(save_path, "frame_bundle"), **npz_file)
        
        else:
            # Process only motion bundle
            
            motion_name = join(bundle_name, "motion.bin")
            
            save_path = bundle_name.replace("bundle-", "bundle_processed-")
            os.makedirs(save_path, exist_ok=True)

            npz_file = {}
            
            process_motion(npz_file, motion_name)

            # Save bundle
            np.savez(join(save_path, "motion_bundle"), **npz_file)
            
if __name__ == '__main__':
    is_iphone12 = False
    main()
