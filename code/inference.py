"""Usage

echo "One sentence. \nAnother sentence. | python inference.py checkpoint1 checkpoint2 --device cuda --audio_folder ~/audio
cat text.txt | python inference.py checkpoint1 checkpoint2 --device cuda
Audios are by default saved to ~/audio

Does not handle numbers - write everything in words.
"""
import argparse, sys, os, time
import torch
from librosa.output import write_wav

from fertility_model import FertilityModel
from melgan.model.generator import Generator
from melgan.utils.hparams import HParam
from hparam import HPStft, HPText
from utils.text import TextProcessor
from functional import mask

parser = argparse.ArgumentParser()
parser.add_argument("fertility_checkpoint", type=str, help="Checkpoint file for Fertility model")
parser.add_argument("melgan_checkpoint", type=str, help="Checkpoint file for MelGan.")
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',  help="What device to use.")
parser.add_argument("--audio_folder", type=str, default="audio", help="Where to save audios")
args = parser.parse_args()

print('Loading model checkpoints')
m = FertilityModel(
    device=args.device
).load(args.fertility_checkpoint)
m.eval()

# TODO: relativize the paths
checkpoint = torch.load('/media/jan/Data/models/nvidia_tacotron2_LJ11_epoch3200.pt')#args.melgan_checkpoint)
hp = HParam('/home/jan/convolutional_tts/code/melgan/config/default.yaml')#"code/melgan/config/default.yaml")
melgan = Generator(hp.audio.n_mel_channels).to(args.device)
melgan.load_state_dict(checkpoint["model_g"])
melgan.eval(inference=False)

print('Processing text')
txt_processor = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
text = [t.strip() for t in sys.stdin.readlines()]

phonemes, plen = txt_processor(text)
# append more zeros - avoid cutoff at the end of the largest sequence
phonemes = torch.cat((phonemes, torch.zeros(len(phonemes), 5).long() ), dim=-1)
phonemes = phonemes.to(args.device)

print('Synthesizing')
# generate spectrograms
with torch.no_grad():
    spec, durations = m((phonemes, plen))


# invert to log(mel-spectrogram)
spec = m.collate.norm.inverse(spec)

# mask with pad value expected by MelGan
msk = mask(spec.shape, durations.sum(dim=-1).long(), dim=1).to(args.device)
spec = spec.masked_fill(~msk, -11.5129)

# Append more pad frames to improve end of the longest sequence
spec = torch.cat((spec.transpose(2,1), -11.5129*torch.ones(len(spec), HPStft.n_mel, 5).to(args.device)), dim=-1)

# generate audio
with torch.no_grad():
    audio = melgan(spec).squeeze(1)

print('Saving audio')
# TODO: cut audios to proper length
for i,a in enumerate(audio.detach().cpu().numpy()):
    write_wav(os.path.join(args.audio_folder,f'{i}.wav'), a, HPStft.sample_rate, norm=False)