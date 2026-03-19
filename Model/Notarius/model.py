#imports
import torch.nn as nn 
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, SpeedPerturbation
from torch.nn.utils.rnn import pad_sequence
#settings
torch.set_num_threads(24)
torch.backends.cudnn.benchmark = True


# define vocabulary
chars = "abcdefghijklmnopqrstuvwxyz '"  # 26 + space + apostrophe = 28 chars
blank = len(chars)                       # 28 = CTC blank token
num_classes = len(chars) + 1            # 29 total

# char to index
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

spec_transform = nn.Sequential(
    MelSpectrogram(n_fft=400, sample_rate=16000,  hop_length=160, n_mels=64),
    AmplitudeToDB(stype="power", top_db=80)
)

speed_perturb = SpeedPerturbation(
    orig_freq=16000, 
    factors=[0.9, 1.0, 1.1]
)

def encode(transcript):
    transcript = transcript.lower()
    return [char2idx[c] for c in transcript if c in char2idx]

def decode(indices):
    return ''.join([idx2char[i] for i in indices if i != blank])

# collate function: pad waveforms and keep transcripts as targets while also returning the orignal lengths of data
def collate_fn(batch):
    waveforms, _, transcripts, *_ = zip(*batch)
    #pertubaed waveforms making them 10% slower or faster
    perturbed_waveforms = [speed_perturb(w)[0] for w in waveforms]
    waveform_lengths = torch.tensor([w.shape[-1] for w in perturbed_waveforms], dtype=torch.long)
    input_lengths = (waveform_lengths // 160) + 1 #160 because hop length is 160

    tensors = pad_sequence(
        [w.squeeze(0) for w in perturbed_waveforms],
        batch_first=True
    ).unsqueeze(1)

    tensors = spec_transform(tensors).squeeze(1)

    # encode transcripts to integer tokens
    encoded = [torch.tensor(encode(t), dtype=torch.long) for t in transcripts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = pad_sequence(encoded, batch_first=True, padding_value=0)

    return tensors, targets, input_lengths, target_lengths
class TSCConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, use_relu=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv1d(in_channel, in_channel, kernel_size, padding, groups=in_channel),
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
        ]
        if use_relu:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)
    
class QuartNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, R=5):
        super().__init__()
    
        self.net = nn.Sequential(
            #On our Rth block, 
            *[TSCConv(in_channel if i==0 else out_channel, out_channel, kernel_size, use_relu=(i!=R-1)) for i in range(R)]
        )
        #residual 
        self.residual = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel)
        )
    def forward(self, x):
        return torch.relu(self.net(x) + self.residual(x))

class QuartzNetBxR(nn.Module):
    def __init__(self, n_mels=64, n_classes=29,R=5, B=5):
        super().__init__()
        assert B%5==0, "B Must be a mutiple of 5"
        self.net = nn.Sequential(
            #c1
            nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #B1-5
            #B1
            *[QuartNetBlock(256, 256, kernel_size=33, R=R) for _ in range(B//5)],
            #B2
            *[QuartNetBlock(256, 256, kernel_size=39, R=R)  for _ in range(B//5)],
            #B3
            QuartNetBlock(256, 512, kernel_size=51, R=R),                                    # first: transitions 256→512
            *[QuartNetBlock(512, 512, kernel_size=51, R=R) for _ in range(B//5 - 1)],       # rest: stays at 512            #B4
            *[QuartNetBlock(512, 512, kernel_size=63, R=R) for _ in range(B//5)],
            #B5
            *[QuartNetBlock(512, 512, kernel_size=75, R=R) for _ in range(B//5)],
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
        x = self.net(x)  # (batch, classes, time)
        x = x.permute(2, 0, 1)      # (time, batch, n_classes)
        return x.log_softmax(dim=2)


