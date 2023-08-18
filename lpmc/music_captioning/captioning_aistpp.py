import argparse
import os, glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.eval_utils import load_pretrained
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST
from omegaconf import OmegaConf
from tqdm import tqdm


def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MSD Training')
    parser.add_argument('--gpu', default=1, type=int,
                        help='GPU id to use.')
    parser.add_argument("--framework", default="transfer", type=str)
    parser.add_argument("--caption_type", default="lp_music_caps", type=str)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--num_beams", default=5, type=int)
    parser.add_argument("--model_type", default="last", type=str)
    # parser.add_argument("--audio_path", default="gBR_sBM_cXX_dXX_mBR0_chXX.wav", type=str)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    save_dir = f"exp/transfer/lp_music_caps/"
    config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))
    model = BartCaptionModel(max_length = config.max_length)
    model, save_epoch = load_pretrained(args, save_dir, model, mdp=config.multiprocessing_distributed)
    model.eval()


    audio_names = glob.glob(os.path.join(r"\\SEUVCL-DATA-03\Data03Training_SATA01\2308_music2frame_zzm\aistpp\wavs", "*.wav"))
    
    audio_cap_all = {}
    for audio_name in tqdm(audio_names):
        audio_id = audio_name.split("\\")[-1]
        audio_tensor = get_audio(audio_path = audio_name)
        audio_tensor = audio_tensor.cuda()

        audio_cap = {}
        with torch.no_grad():
            output = model.generate(samples=audio_tensor,num_beams=args.num_beams)
            
            number_of_chunks = range(audio_tensor.shape[0])
            for chunk, text in zip(number_of_chunks, output):
                time = f"{chunk * 10}:00-{(chunk + 1) * 10}:00"
                # item = {"text":text,"time":time}
                # inference[chunk] = item
                # print(item)
                audio_cap[time] = text
            
        print(audio_cap)
        audio_cap_all[audio_id] = audio_cap
    json.dump(audio_cap_all, open("test_cap.json", 'w'))