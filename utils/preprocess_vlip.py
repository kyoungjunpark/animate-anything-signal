import torch
import torchvision
import json
import os
import random
import numpy as np
import argparse
import decord
import cv2

from einops import rearrange
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from glob import glob

decord.bridge.set_bridge('torch')


class PreProcessVideos:
    def __init__(
            self,
            config_name,
            config_save_name,
            video_directory,
            random_start_frame,
            clip_frame_data,
            max_frames,
            beam_amount,
            prompt_amount,
            min_prompt_length,
            max_prompt_length,
            save_dir
    ):

        # Paramaters for parsing videos
        self.prompt_amount = prompt_amount
        self.video_directory = video_directory
        self.random_start_frame = random_start_frame
        self.clip_frame_data = clip_frame_data
        self.max_frames = max_frames
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")

        # Parameters for BLIP2
        self.processor = None
        self.blip_model = None
        self.beam_amount = beam_amount
        self.min_length = min_prompt_length
        self.max_length = max_prompt_length

        # Helper parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir

        # Config parameters
        self.config_name = config_name
        self.config_save_name = config_save_name

    # Base dict to hold all the data.
    # {base_config}
    def build_base_config(self):
        return {
            "name": self.config_name,
            "data": []
        }

    # Video dict for individual videos.
    # {base_config: data -> [{video_path, num_frames, data}]}
    def build_video_config(self, video_path: str, signal_path: str, num_frames: int):
        return {
            "video_path": video_path,
            "camera_pose_path": video_path.replace("output.mp4", "camera_pose.npy"),
            "tx_path": signal_path.replace("channels.pt", "tx.txt"),
            "signal_path": signal_path,
            "initial_signal_path": signal_path.replace("channels.pt", "initial_channels.pt"),
            "num_frames": num_frames,
            "data": []
        }

    # Dict for video frames and prompts / captions.
    # Gets the frame index, then gets a caption for the that frame and stores it.
    # {base_config: data -> [{name, num_frames, data: {frame_index, prompt}}]}
    def build_video_data(self, frame_index: int, prompt: str):
        return {
            "frame_index": frame_index,
            "prompt": prompt
        }

    def check_frames_same(self, video_path):
        # Load video
        cap = cv2.VideoCapture(video_path)

        # Read the first frame
        ret, first_frame = cap.read()

        if not ret:
            print("Error reading the first frame")
            return False

        # Convert first frame to grayscale for easier comparison
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, frame = cap.read()

            # If no more frames, exit loop
            if not ret:
                break

            # Convert current frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check if current frame is the same as the first frame
            if not np.array_equal(first_frame_gray, frame_gray):
                return False

        print("All frames are the same.")
        return True

    # Load BLIP2 for processing
    def load_blip(self):
        print("Loading BLIP2")

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        model.to(self.device)

        self.processor = processor
        self.blip_model = model

    # Process the frames to get the length and image.
    # The limit parameter ensures we don't get near the max frame length.
    def video_processor(
            self,
            video_reader: VideoReader,
            num_frames: int,
            random_start_frame=True,
            frame_num=0
    ):

        frame_number = (
            random.randrange(0, int(num_frames)) if random_start_frame else frame_num
        )
        frame = video_reader[frame_number].permute(2, 0, 1)
        image = transforms.ToPILImage()(frame).convert("RGB")
        return frame_number, image

    def get_frame_range(self, derterministic):
        return range(self.prompt_amount) if self.random_start_frame else derterministic

    def process_blip(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.blip_model.generate(
            **inputs,
            num_beams=self.beam_amount,
            min_length=self.min_length,
            max_length=self.max_length
        )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True)[0].strip()

        return generated_text

    def get_out_paths(self, prompt, frame_number):
        out_name = f"{prompt}_{str(frame_number)}"
        save_path = f"{self.save_dir}/{self.config_save_name}"
        save_filepath = f"{save_path}/{out_name}.mp4"

        return out_name, save_path, save_filepath

    def save_train_config(self, config: dict):
        os.makedirs(self.save_dir, exist_ok=True)

        save_json = json.dumps(config, indent=4)
        save_dir = f"{self.save_dir}/{self.config_save_name}"

        with open(f"{save_dir}.json", 'w') as f:
            f.write(save_json)

    def save_video(self, save_path, save_filepath, frames):
        os.makedirs(save_path, exist_ok=True)
        torchvision.io.write_video(save_filepath, frames, fps=30)

    # Main loop for processing all videos.
    def process_videos(self):
        # self.load_blip()
        config = self.build_base_config()

        if not os.path.exists(self.video_directory):
            raise ValueError(f"{self.video_directory} does not exist.")

        for video_path in glob(os.path.join(self.video_directory, '**', '*.mp4'), recursive=True):

            print(video_path)
            video_reader = None
            derterministic_range = None
            video_len = 0
            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                video_len = len(video_reader)
                frame_step = abs(video_len // self.prompt_amount)
                derterministic_range = range(1, abs(video_len - 1), frame_step)
            except:
                print(f"Error loading {video_path}. Video may be unsupported or corrupt.")
                continue
            if self.check_frames_same(video_path):
                print(f"Frames are all same for {video_path}.")
                continue

            signal_path = video_path.replace("locomotion_video_000", "locomotion_signal_2to6_multi").replace("output.mp4",
                                                                                                           "channels.pt")

            camera_pose_path = video_path.replace("output.mp4", "camera_pose.npy")
            tx_path = signal_path.replace("channels.pt", "tx.txt")

            if not os.path.exists(camera_pose_path) or not os.path.exists(tx_path):
                print(f"No camera or tx {camera_pose_path}")
                continue

            camera_pose = np.load(camera_pose_path)

            tx_pos = np.loadtxt(tx_path)

            if len(camera_pose) != 4 or len(tx_pos) != 3:
                print(len(camera_pose), len(tx_pos))
                print(camera_pose, tx_pos)
                continue

            if not os.path.exists(signal_path):
                print(f"No signal data {signal_path}")
                continue

            # Another try catch block because decord isn't perfect.
            try:
                num_frames = int(len(video_reader))
                channels = torch.load(signal_path)

                if num_frames != channels.size(0):
                    print(f"Not matching #: {num_frames}, {channels.size(0)}")
                    continue
                # Convert the PyTorch tensor to a NumPy array
                # numpy_array = channels.numpy()

                # Save the NumPy array as a .npy file
                # signal_path = signal_path.replace("frame1.pt", "channels.npy")
                # np.save(signal_path, numpy_array)

                video_config = self.build_video_config(video_path, signal_path, num_frames)

                # Secondary loop that process a specified amount of prompts, selects a random frame, then appends it.
                for i in tqdm(
                        self.get_frame_range(derterministic_range),
                        desc=f"Processing {os.path.basename(video_path)}"
                ):
                    frame_number, image = self.video_processor(
                        video_reader,
                        num_frames,
                        self.random_start_frame,
                        frame_num=i
                    )

                    # prompt = self.process_blip(image)
                    prompt = "test"
                    video_data = self.build_video_data(frame_number, prompt)

                    if self.clip_frame_data:

                        # Minimum value, frame number, max value (length of entire video)
                        max_range = abs(len(video_reader) - 1)
                        frame_number = i
                        frame_number = sorted((1, frame_number, max_range))[1]

                        frame_range = range(frame_number, max_range)
                        frame_range_nums = list(frame_range)

                        frames = video_reader.get_batch(frame_range_nums[:self.max_frames])

                        out_name, save_path, save_filepath = self.get_out_paths(prompt, frame_number)

                        self.save_video(save_path, save_filepath, frames)

                        video_data['clip_path'] = save_filepath
                        video_config["data"].append(video_data)

                    else:
                        video_config["data"].append(video_data)

                config['data'].append(video_config)

            except Exception as e:
                print(e)
                continue

        print(f"Done. Saving train config to {self.save_dir}.")
        self.save_train_config(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_name', help="The name of the configuration.", type=str, default='My Config')
    parser.add_argument('--config_save_name', help="The name of the config file that's saved.", type=str,
                        default='my_config')
    parser.add_argument('--video_directory', help="The directory where your videos are located.", type=str,
                        default='./videos')
    parser.add_argument(
        '--random_start_frame',
        help="Use random start frame when processing videos. Good for long videos where frames have different scenes and meanings.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--clip_frame_data',
        help="Save the frames as video clips to HDD/SDD. Videos clips are saved in the same folder as your json directory.",
        action='store_true',
        default=False
    )
    parser.add_argument('--max_frames', help="Maximum frames for clips when --clip_frame_data is enabled.", type=int,
                        default=60)
    parser.add_argument('--beam_amount', help="Amount for BLIP beam search.", type=int, default=7)
    parser.add_argument('--prompt_amount', help="The amount of prompts per video that is processed.", type=int,
                        default=25)
    parser.add_argument('--min_prompt_length', help="Minimum words required in prompt.", type=int, default=15)
    parser.add_argument('--max_prompt_length', help="Maximum words required in prompt.", type=int, default=30)
    parser.add_argument('--save_dir', help="The directory to save the config to.", type=str,
                        default=f"{os.getcwd()}/train_data")

    args = parser.parse_args()

    processor = PreProcessVideos(**vars(args))
    processor.process_videos()
