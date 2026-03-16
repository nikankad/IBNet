#imports
import torchaudio
from torch.utils.data import DataLoader
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence


#dataset
dataset = torchaudio.datasets.LIBRISPEECH(
    root="/home/student/GOATS422/Notarius/Datasets",
    url="train-clean-100",
    download=False, 
)

#transform waveform to spectogram in decibals
spec_transform = nn.Sequential(
    torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160),
    torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
)

# collate function: pad waveforms and keep transcripts as targets
def collate_fn(batch):
    waveforms, sample_rates, transcripts, speaker_ids, chapter_ids, utt_ids = zip(*batch)

    # [B, 1, T_max]
    tensors = pad_sequence([w.squeeze(0) for w in waveforms], batch_first=True).unsqueeze(1)

    tensors_db = spec_transform(tensors)
    # keep text targets as a list (do not torch.tensor strings)
    targets = list(transcripts)

    return tensors_db, targets, tensors, sample_rates[0]
#initialize dataloader
df = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)

class TSCConv(nn.Module):
    def __init__(self, inChannel, outChannel, kernelSize, stride):
        super().__init__()
        padding = (kernelSize - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(inChannel, outChannel, kernelSize, stride, padding=padding, groups=inChannel),
            nn.Conv1d(inChannel, outChannel, kernel_size=1),
            nn.BatchNorm1d(outChannel),
            nn.ReLU()  
        )

    def forward(self,x):
        return self.net(x)
