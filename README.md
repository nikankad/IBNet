# Notarius
English speech to text deep learning model

## Save and reuse training progress

Training now writes checkpoints to `outputs/checkpoints`:

- `last.pt`: most recent epoch
- `best.pt`: best validation loss so far
- `epoch_XXX.pt`: periodic snapshot (default every 10 epochs)
- `final_model.pt`: final weights after training ends

Run training:

```bash
python model/train.py
```

Resume a stopped run from a checkpoint:

```python
from model.train import train_model

train_model(B=5, R=5, num_epochs=10, resume_from="last.pt")
```

Transcribe a real wav file using a saved checkpoint:

```bash
python model/transcribe.py --audio /path/to/audio.wav --checkpoint outputs/checkpoints/best.pt
```
