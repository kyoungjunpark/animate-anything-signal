import numpy as np
import torch
from tqdm import tqdm
from common_metrics_on_video_quality.fvd.styleganv.fvd import get_fvd_feats as get_stylegan_fvd_feats, frechet_distance as stylegan_frechet_distance, load_i3d_pretrained as load_styleganv
from common_metrics_on_video_quality.fvd.videogpt.fvd import load_i3d_pretrained as load_videogpt
from common_metrics_on_video_quality.fvd.videogpt.fvd import get_fvd_logits as get_videogpt_fvd_feats
from common_metrics_on_video_quality.fvd.videogpt.fvd import frechet_distance as videogpt_frechet_distance

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x


def calculate_fvd(videos1, videos2, device, method='styleganv'):

    if method == 'styleganv':
        i3d = load_styleganv(device=device)

    elif method == 'videogpt':
        i3d = load_videogpt(device=device)
    else:
         raise Exception
    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}

    # for calculate FVD, each clip_timestamp must >= 10
    for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
       
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, : clip_timestamp]
        videos_clip2 = videos2[:, :, : clip_timestamp]

        # get FVD features
        if method == 'styleganv':
            feats1 = get_stylegan_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_stylegan_fvd_feats(videos_clip2, i3d=i3d, device=device)
            fvd_results[clip_timestamp] = stylegan_frechet_distance(feats1, feats2)

        elif method == 'videogpt':
            feats1 = get_videogpt_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_videogpt_fvd_feats(videos_clip2, i3d=i3d, device=device)
            fvd_results[clip_timestamp] = videogpt_frechet_distance(feats1, feats2)

        else:
            raise Exception

        # calculate FVD when timestamps[:clip]

    result = {
        "value": fvd_results,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, heigth, width",
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    import json
    result = calculate_fvd(videos1, videos2, device, method='videogpt')
    print(json.dumps(result, indent=4))

    result = calculate_fvd(videos1, videos2, device, method='styleganv')
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
