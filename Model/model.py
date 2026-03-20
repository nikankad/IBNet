#imports
import torchaudio
import os
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
from torch.nn.utils.rnn import pad_sequence
torch.set_num_threads(24)

# Support for Vertex AI - use env var if available, else use default
root = os.getenv("DATASET_PATH", "home/student/GOATS422")

train_ds = torchaudio.datasets.LIBRISPEECH(root=root, url="train-clean-100", download=False)
val_ds   = torchaudio.datasets.LIBRISPEECH(root=root, url="dev-clean",       download=False)
test_ds  = torchaudio.datasets.LIBRISPEECH(root=root, url="test-clean",      download=False)


spec_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(n_fft=400, sample_rate=16000,  hop_length=160, n_mels=64),
    torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
)

# define vocabulary
chars = "abcdefghijklmnopqrstuvwxyz '"  # 26 + space + apostrophe = 28 chars
blank = len(chars)                       # 28 = CTC blank token
num_classes = len(chars) + 1            # 29 total

# char to index
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

def encode(transcript):
    transcript = transcript.lower()
    return [char2idx[c] for c in transcript if c in char2idx]

def decode(indices):
    return ''.join([idx2char[i] for i in indices if i != blank])

# collate function: pad waveforms and keep transcripts as targets while also returning the orignal lengths of data
def collate_fn(batch):
    waveforms, _, transcripts, *_ = zip(*batch)

    waveform_lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    input_lengths = (waveform_lengths // 160) + 1 #160 because hop length is 160

    tensors = pad_sequence(
        [w.squeeze(0) for w in waveforms],
        batch_first=True
    ).unsqueeze(1)

    tensors = spec_transform(tensors)

    # encode transcripts to integer tokens
    encoded = [torch.tensor(encode(t), dtype=torch.long) for t in transcripts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = pad_sequence(encoded, batch_first=True, padding_value=0)

    return tensors, targets, input_lengths, target_lengths
#initialize dataloader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn, num_workers=24, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=24)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=24)

class TSCConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel),
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()  
        )
    def forward(self,x):
        return self.net(x)
    
class QuartNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, R=5):
        super().__init__()
    
        self.net = nn.Sequential(
            TSCConv(in_channel, out_channel, kernel_size),
            *[TSCConv(out_channel, out_channel, kernel_size) for _ in range(R-1)]
        )

        #residual 
        self.residual = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel)
        )
    def forward(self, x):
        return torch.relu(self.net(x) + self.residual(x))

class QuartzNetBxR(nn.Module):
    def __init__(self, n_mels=64, n_classes=29, B=5, R=5):
        super().__init__()
        self.net = nn.Sequential(
            #c1
            nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #B1-5
            #B1
            QuartNetBlock(256, 256, kernel_size=33, R=5),
            #B2
            QuartNetBlock(256, 256, kernel_size=39, R=5),
            #B3
            QuartNetBlock(256, 512, kernel_size=51, R=5),
            #B4
            QuartNetBlock(512, 512, kernel_size=63, R=5),
            #B5
            QuartNetBlock(512, 512, kernel_size=75, R=5),
            #C2
            nn.Conv1d(512, 512, kernel_size=87, padding=43),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #C3
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            #C4
            nn.Conv1d(1024, n_classes, dilation=2, kernel_size=1),  
        )
    def forward(self, x):
        x = x.squeeze(1) # now: (batch, n_mels, time)
        return self.net(x)


