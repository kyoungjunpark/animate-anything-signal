import time
import numpy as np
import io
import os
from PIL import Image
import cv2
import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
random.seed(125)
np.random.seed(125)

def run_model(model, rgbs, N, sw, gif_name):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)

    # try to pick a point on the dog, so we get an interesting trajectory
    # x = torch.randint(-10, 10, size=(1, N), device=torch.device('cuda')) + 468
    # y = torch.randint(-10, 10, size=(1, N), device=torch.device('cuda')) + 118
    x = torch.ones((1, N), device=torch.device('cuda')) * 450.0
    y = torch.ones((1, N), device=torch.device('cuda')) * 100.0
    xy0 = torch.stack([x, y], dim=-1) # B, N, 2
    _, S, C, H, W = rgbs.shape

    trajs_e = torch.zeros((B, S, N, 2), dtype=torch.float32, device='cuda')
    for n in range(N):
        # print('working on keypoint %d/%d' % (n+1, N))
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cuda')
        traj_e[:,0] = xy0[:,n] # B, 1, 2  # set first position 
        feat_init = None
        while not done:
            end_frame = cur_frame + 8

            rgb_seq = rgbs[:,cur_frame:end_frame]
            S_local = rgb_seq.shape[1]
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,8-S_local,1,1,1)], dim=1)

            outs = model(traj_e[:,cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
            preds = outs[0]
            vis = outs[2] # B, S, 1
            feat_init = outs[3]
            
            vis = torch.sigmoid(vis) # visibility confidence
            xys = preds[-1].reshape(1, 8, 2)
            traj_e[:,cur_frame:end_frame] = xys[:,:S_local]

            found_skip = False
            thr = 0.9
            si_last = 8-1 # last frame we are willing to take
            si_earliest = 1 # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0,si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e
    
    pad = 50
    rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
    trajs_e = trajs_e + pad

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    
    if sw is not None and sw.save_this:
        linewidth = 2

        for n in range(N):
            # print('visualizing kp %d' % n)
            kp_vis = sw.summ_traj2ds_on_rgbs('video_%d/kp_%d_trajs_e_on_rgbs' % (sw.global_step, n), trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=linewidth)

            # write to disk, in case that's more convenient
            kp_list = list(kp_vis.unbind(1))
            kp_list = [kp[0].permute(1,2,0).cpu().numpy() for kp in kp_list]
            kp_list = [Image.fromarray(kp) for kp in kp_list]
            out_fn = './chain_out_%s.gif' % gif_name
            kp_list[0].save(out_fn, save_all=True, append_images=kp_list[1:])
            print('saved %s' % out_fn)
            
        # sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        # sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')
        

    return trajs_e

def load_video_to_tensor(video_path, num_frames=8, similarity_threshold=0.99):
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    frames = []
    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add the frame to the list of frames if it's unique
        frames.append(frame_rgb)

        # Set the current frame as the previous frame for the next iteration
        prev_frame = frame_rgb

    cap.release()

    # Now that we have all unique frames, we can select the 8 frames
    total_frames = len(frames)

    # If there are fewer than num_frames, select as many as possible
    if total_frames < num_frames:
        print(f"Warning: Only {total_frames} unique frames were found instead of {num_frames}.")

    # Select 8 frames from the unique frames
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    # print(total_fr`amesindices)
    # selected_frames = frames[indices]
    print(indices)
    selected_frames = [frames[i] for i in indices]

    # Convert selected frames to a tensor (F, C, H, W)
    frames_tensor = [torch.tensor(frame).permute(2, 0, 1) for frame in
                     selected_frames]  # Convert (H, W, C) to (C, H, W)

    # Stack selected frames into a tensor (F, C, H, W)
    video_tensor = torch.stack(frames_tensor)

    # Add a batch dimension to make the shape (1, F, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor

def main():

    # the idea in this file is to chain together pips from a long sequence, and return some visualizations
    
    exp_name = '00' # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    ## choose hyps
    B = 1
    S = 25
    # N = 1 # number of points to track
    N = 16**2 # number of points to track


    log_freq = 1 # when to produce visualizations 

    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    log_dir = 'logs_chain_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=4).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()
    

    read_start_time = time.time()

    global_step += 1

    sw_t = utils.improc.Summ_writer(
        writer=writer_t,
        global_step=global_step,
        log_freq=log_freq,
        fps=12,
        scalar_freq=int(log_freq/2),
        just_gif=True)
    read_time = time.time() - read_start_time
    iter_start_time = time.time()
    real_video = load_video_to_tensor("demo_videos/real.mp4",num_frames=25)
    fake_video2 = load_video_to_tensor("demo_videos/fake2.mp4", num_frames=25)
    fake_video = load_video_to_tensor("demo_videos/fake.mp4", num_frames=25)

    total_distance1 = 0
    total_distance2 = 0

    with torch.no_grad():
        # print(trajs_e.size())  # torch.Size([1, 8, 256, 2])
        trajs_real = run_model(model, real_video, N, sw_t, "real")
        trajs_fake = run_model(model, fake_video, N, sw_t, "fake")
        trajs_fake2 = run_model(model, fake_video2, N, sw_t, "fake2")

        for i in range(trajs_real.size(2)):  # 256 dimension (index 2)
            slice_real = trajs_real[0, :, i, :]
            slice_fake = trajs_fake[0, :, i, :]
            slice_fake2 = trajs_fake2[0, :, i, :]
            print(slice_real.size(), slice_fake.size(), slice_fake2.size())
            slice_real = slice_real.cpu().numpy()
            slice_fake = slice_fake.cpu().numpy()
            slice_fake2 = slice_fake2.cpu().numpy()

            distance, path = fastdtw(slice_real, slice_fake, dist=euclidean)
            total_distance1 += distance

            distance2, path = fastdtw(slice_real, slice_fake2, dist=euclidean)
            total_distance2 += distance2
    print(total_distance1 / 256, total_distance2 / 256)

    writer_t.close()

if __name__ == '__main__':
    main()
