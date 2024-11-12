import time
import numpy as np
import io
import os
from PIL import Image
import cv2
from utils.pips import saverloader
import imageio.v2 as imageio
from utils.pips.nets.pips import Pips
from utils.pips.utils.improc import preprocess_color
import random
import glob
from utils.pips.utils.basic import print_, print_stats, meshgrid2d
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

random.seed(125)
np.random.seed(125)


def run_model(model, rgbs, N, sw, gif_name, masked_coord):
    rgbs = rgbs.cuda().float()  # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B * S, C, H, W)
    H_, W_ = 360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1) / float(N_ - 1) * (H - 16)
    grid_x = 8 + grid_x.reshape(B, -1) / float(N_ - 1) * (W - 16)
    xy = torch.stack([grid_x, grid_y], dim=-1)  # B, N_*N_, 2

    coordinates_list = list(masked_coord)
    # Convert the list to a tensor
    coordinates_tensor = torch.tensor(coordinates_list, dtype=torch.int64)
    # coordinates_tensor += 8
    # coordinates_tensor = coordinates_tensor[:, [1, 0]]
    # Pad or truncate to fit the shape (1, 256, 2)

    # Reshape the tensor to shape (1, 256, 2)
    coordinates_tensor = coordinates_tensor.unsqueeze(0)

    # The final tensor shape should be (1, 256, 2)
    # print(xy)
    # print(coordinates_tensor)
    xy = coordinates_tensor.cuda()

    _, S, C, H, W = rgbs.shape

    print_stats('rgbs', rgbs)  # min = 0.00, mean = 106.25, max = 255.00 torch.Size([1, 8, 3, 360, 640])
    preds, preds_anim, vis_e, stats = model(xy, rgbs, iters=6)
    trajs_e = preds[-1]
    print_stats('trajs_e', trajs_e)  # min = -63.76, mean = 230.17, max = 632.00 torch.Size([1, 8, 256, 2])

    pad = 50
    rgbs = F.pad(rgbs.reshape(B * S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H + pad * 2,
                                                                                            W + pad * 2)
    trajs_e = trajs_e + pad

    # B,S,N,2, where, S is the sequence length and N is the number of particles, and 2 is the x and y coordinates.
    # exit(1)
    if sw is not None and sw.save_this:
        linewidth = 2

        # visualize the input
        o1 = sw.summ_rgbs('inputs/rgbs', preprocess_color(rgbs[0:1]).unbind(1))
        # visualize the trajs overlaid on the rgbs
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], preprocess_color(rgbs[0:1]),
                                     cmap='spring', linewidth=linewidth)
        # visualize the trajs alone
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1]) * -0.5,
                                     cmap='spring', linewidth=linewidth)
        # concat these for a synced wide vis
        wide_cat = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

        # write to disk, in case that's more convenient
        wide_list = list(wide_cat.unbind(1))
        wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        wide_list = [Image.fromarray(wide) for wide in wide_list]
        out_fn = './out_%s.gif' % gif_name
        wide_list[0].save(out_fn, save_all=True, append_images=wide_list[1:])
        print('saved %s' % out_fn)

        # alternate vis
        sw.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs2', trajs_e[0:1], vis_e[0:1],
                                 preprocess_color(rgbs[0:1]))

        # animation of inference iterations
        rgb_vis = []
        for trajs_e_ in preds_anim:
            trajs_e_ = trajs_e_ + pad
            rgb_vis.append(
                sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], torch.mean(preprocess_color(rgbs[0:1]), dim=1),
                                       cmap='spring', linewidth=linewidth, only_return=True))
        sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return trajs_e


def frames_are_equal(frame1, frame2, threshold=0.95):
    # Calculate the mean squared error between two frames
    mse = np.sum((frame1 - frame2) ** 2) / float(frame1.shape[0] * frame1.shape[1])
    # If the frames are nearly identical, we consider them equal
    # Higher threshold means frames need to be more similar to be considered equal
    if mse < threshold:
        return True
    return False


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
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add the frame to the list of frames if it's unique
        frames.append(frame_rgb)
        # Set the current frame as the previous frame for the next iteration
        prev_frame = gray_frame

    cap.release()

    # Now that we have all unique frames, we can select the 8 frames
    total_frames = len(frames)

    # If there are fewer than num_frames, select as many as possible
    # if total_frames < num_frames:
    #     print(f"Warning: Only {total_frames} unique frames were found instead of {num_frames}.")

    # Select 8 frames from the unique frames
    # indices = np.linspace(0, total_frames - 1, num_frames, dtype=torch.long)
    if len(frames) != 25:
        print("Change videos to first 25 frames", len(frames))
        frame_step = 3
        frame_range = list(range(0, len(frames), frame_step))
        # print(total_framesindices)
        frames = [frames[i] for i in frame_range]
        frames = frames[:25]
        assert len(frames) == 25, (len(frames), frame_range)
        # print("frame_range", frame_range)

    # indices = np.linspace(0, total_frames - 1, num_frames, dtype=torch.long)
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    frames = [frames[i] for i in indices]
    # print("indices", indices)
    # Convert selected frames to a tensor (F, C, H, W)
    frames_tensor = [torch.tensor(frame).permute(2, 0, 1) for frame in
                     frames]  # Convert (H, W, C) to (C, H, W)

    # Stack selected frames into a tensor (F, C, H, W)
    video_tensor = torch.stack(frames_tensor)

    # Add a batch dimension to make the shape (1, F, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor


def main():
    # the idea in this file is to run the model on some demo images, and return some visualizations

    exp_name = '00'  # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    ## choose hyps
    B = 1
    S = 8
    N = 16 ** 2  # number of points to track

    log_freq = 1  # when to produce visualizations

    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    log_dir = 'logs_demo'
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
        fps=5,
        scalar_freq=int(log_freq),
        just_gif=True)

    read_time = time.time() - read_start_time
    iter_start_time = time.time()
    real_video = load_video_to_tensor("demo_videos/real.mp4", num_frames=8)
    fake_video2 = load_video_to_tensor("demo_videos/fake2.mp4", num_frames=8)
    fake_video = load_video_to_tensor("demo_videos/fake.mp4", num_frames=8)

    total_distance1 = 0
    total_distance2 = 0

    with torch.no_grad():
        # print(trajs_e.size())  # torch.Size([1, 8, 256, 2])
        trajs_real, mask_real = run_model(model, real_video, N, sw_t, "real")
        trajs_fake, mask = run_model(model, fake_video, N, sw_t, "fake")
        trajs_fake2 = run_model(model, fake_video2, N, sw_t, "fake2")

        for i in range(trajs_real.size(2)):  # 256 dimension (index 2)
            slice_real = trajs_real[0, :, i, :]
            slice_fake = trajs_fake[0, :, i, :]
            slice_fake2 = trajs_fake2[0, :, i, :]

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
