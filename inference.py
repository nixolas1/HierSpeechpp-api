import os
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import utils
from Mels_preprocess import MelSpectrogramFixed

from hierspeechpp_speechsynthesizer import (
    SynthesizerTrn
)
from ttv_v1.text import text_to_sequence
from ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2V
from speechsr24k.speechsr import SynthesizerTrn as AudioSR
from speechsr48k.speechsr import SynthesizerTrn as AudioSR48
from denoiser.generator import MPNet
from denoiser.infer import denoise


from cog import Path, Input, BasePredictor

seed = 420691337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def load_text(fp):
    with open(fp, 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    return filelist
def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

def add_blank_token(text):
    text_norm = intersperse(text, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

class Predictor(BasePredictor):
    def setup(self):
        self.output_dir = "output"
        self.ckpt = "./logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth"
        self.ckpt_text2w2v = "./logs/ttv_libritts_v1/ttv_lt960_ckpt.pth"
        self.ckpt_sr = "./speechsr24k/G_340000.pth"
        self.ckpt_sr48 = "./speechsr48k/G_100000.pth"
        self.denoiser_ckpt = "denoiser/g_best"
        self.scale_norm = "max"
        self.output_sr = 16000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hps = utils.get_hparams_from_file(os.path.join(os.path.split(self.ckpt)[0], 'config.json'))
        self.hps_t2w2v = utils.get_hparams_from_file(os.path.join(os.path.split(self.ckpt_text2w2v)[0], 'config.json'))
        self.h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(self.ckpt_sr)[0], 'config.json'))
        self.h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(self.ckpt_sr48)[0], 'config.json'))
        self.hps_denoiser = utils.get_hparams_from_file(os.path.join(os.path.split(self.denoiser_ckpt)[0], 'config.json'))
        self.model_load()

    def model_load(self):
        self.mel_fn = MelSpectrogramFixed(
            sample_rate=self.hps.data.sampling_rate,
            n_fft=self.hps.data.filter_length,
            win_length=self.hps.data.win_length,
            hop_length=self.hps.data.hop_length,
            f_min=self.hps.data.mel_fmin,
            f_max=self.hps.data.mel_fmax,
            n_mels=self.hps.data.n_mel_channels,
            window_fn=torch.hann_window
        ).cuda()
    
        self.net_g = SynthesizerTrn(self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).cuda()
        self.net_g.load_state_dict(torch.load(self.ckpt))
        self.net_g.eval()

        self.text2w2v = Text2W2V(self.hps.data.filter_length // 2 + 1,
        self.hps.train.segment_size // self.hps.data.hop_length,
        **self.hps_t2w2v.model).cuda()
        self.text2w2v.load_state_dict(torch.load(self.ckpt_text2w2v))
        self.text2w2v.eval()

        if self.output_sr == 48000:
            self.audiosr = AudioSR48(self.h_sr48.data.n_mel_channels,
                self.h_sr48.train.segment_size // self.h_sr48.data.hop_length,
                **self.h_sr48.model).cuda()
            utils.load_checkpoint(self.ckpt_sr48, self.audiosr, None)
            self.audiosr.eval()

        elif self.output_sr == 24000:
            self.audiosr = AudioSR(self.h_sr.data.n_mel_channels,
            self.h_sr.train.segment_size // self.h_sr.data.hop_length,
            **self.h_sr.model).cuda()
            utils.load_checkpoint(self.ckpt_sr, self.audiosr, None)
            self.audiosr.eval()

        else:
            self.audiosr = None

        self.denoiser = MPNet(self.hps_denoiser).cuda()
        state_dict = load_checkpoint(self.denoiser_ckpt, self.device)
        self.denoiser.load_state_dict(state_dict['generator'])
        self.denoiser.eval()
        self.hierspeech = self.net_g, self.text2w2v, self.audiosr, self.denoiser, self.mel_fn

    def tts(self, text):
        net_g, text2w2v, audiosr, denoiser, mel_fn = self.hierspeech
        os.makedirs(self.output_dir, exist_ok=True)
        text = text_to_sequence(str(text), ["english_cleaners2"])
        token = add_blank_token(text).unsqueeze(0).cuda()
        token_length = torch.LongTensor([token.size(-1)]).cuda()
        audio, sample_rate = torchaudio.load(self.input_prompt)
        audio = audio[:1,:]
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000, resampling_method="kaiser_window")
        if self.scale_norm == 'prompt':
            prompt_audio_max = torch.max(audio.abs())
        ori_prompt_len = audio.shape[-1]
        p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
        audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
        if self.denoise_ratio == 0:
            audio = torch.cat([audio.cuda(), audio.cuda()], dim=0)
        else:
            with torch.no_grad():
                denoised_audio = denoise(audio.squeeze(0).cuda(), denoiser, self.hps_denoiser)
            audio = torch.cat([audio.cuda(), denoised_audio[:,:audio.shape[-1]]], dim=0)
        audio = audio[:,:ori_prompt_len]
        src_mel = mel_fn(audio.cuda())
        src_length = torch.LongTensor([src_mel.size(2)]).to(self.device)
        src_length2 = torch.cat([src_length,src_length], dim=0)
        with torch.no_grad():
            w2v_x, pitch = text2w2v.infer_noise_control(token, token_length, src_mel, src_length2,
                                                         noise_scale=self.noise_scale_ttv, denoise_ratio=self.denoise_ratio)
            src_length = torch.LongTensor([w2v_x.size(2)]).cuda()
            pitch[pitch<torch.log(torch.tensor([55]).cuda())] = 0
            converted_audio = net_g.voice_conversion_noise_control(w2v_x, src_length, src_mel, src_length2, pitch,
                                                                   noise_scale=self.noise_scale_vc, denoise_ratio=self.denoise_ratio)
            if self.output_sr == 48000 or 24000:
                converted_audio = audiosr(converted_audio)
        converted_audio = converted_audio.squeeze()
        if self.scale_norm == 'prompt':
            converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * prompt_audio_max
        else:
            converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * 0.999
        converted_audio = converted_audio.cpu().numpy().astype('int16')
        return converted_audio


    def predict(self, text: str = Input(description="Text to convert to speech"),
                input_prompt: str = Input(description="URL to reference voice", default="http://localhost:5001/reference_voice.wav"),
                noise_scale_ttv: float = Input(0.333, "Noise scale for Text-to-Vocal", ge=0, le=1),
                noise_scale_vc: float = Input(0.333, "Noise scale for Voice Conversion", ge=0, le=1),
                denoise_ratio: float = Input(0, "Denoising ratio (more than 0 needs lots of VMEM)", ge=0, le=1),
                output_sample_rate: int = Input(
                    description="Sample rate of the output audio file. More than 16k needs lots of VMEM.",
                    choices=[16000, 24000, 48000], default=16000),
                ) -> Path:
        self.input_prompt = input_prompt
        self.noise_scale_ttv = noise_scale_ttv
        self.noise_scale_vc = noise_scale_vc
        self.denoise_ratio = denoise_ratio
        self.output_sr = output_sample_rate

        wavs = []
        sentences = utils.split_and_recombine_text(text, 370, 420)
        for sent in sentences:
            print("--- " + sent)
            wav = self.tts(text)
            wavs.append(wav)
        wav = np.concatenate(wavs)

        output_file = os.path.join("/tmp/", "out.wav")
        write(output_file, self.output_sr, wav)
        return Path(output_file)