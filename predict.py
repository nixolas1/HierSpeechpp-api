from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:

        global device, hps, hps_t2w2v,h_sr,h_sr48, hps_denoiser
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        hps = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt)[0], 'config.json'))
        hps_t2w2v = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_text2w2v)[0], 'config.json'))
        h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr)[0], 'config.json') )
        h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr48)[0], 'config.json') )
        hps_denoiser = utils.get_hparams_from_file(os.path.join(os.path.split(a.denoiser_ckpt)[0], 'config.json'))
        
        self.hierspeech = model_load(a) 
    
    def predict(
        self,
        text: str = Input(description="Text to convert to speech"),
        reference: Path = Input(description="Reference speech to copy style from", default="reference_voice.wav"),
        noise_scale_ttv: float = Input(description="", ge=0, le=1, default=0.333),
        noise_scale_vc: float = Input(description="", ge=0, le=1, default=0.333),
        seed: int = Input(description="Seed for reproducibility", default=1337)
    ) -> Path:
        """Run a single prediction on the model"""
        speech = tts(text, a, hierspeech)
        out_path = "/tmp/out.mp3"
        audio = ipd.Audio(wav, rate=24000, normalize=True)
        audio = AudioSegment(audio.data, frame_rate=24000, sample_width=2, channels=1)
        audio.export(out_path, format="mp3", bitrate="64k")
        return Path(out_path)
