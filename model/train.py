import torch.nn as nn 
import torch
from torchaudio.datasets import LIBRISPEECH
from helpers import collate_fn, collate_fn_test
from model import QuartzNetBxR
# from dataset import LocalLibriSpeechDataset
from torch.utils.data import DataLoader
from torch_optimizer import NovoGrad
from torch.utils.data import Subset
root = "C:/Users/nikan/Desktop/coding/Notarius/datasets"
# settings
# torch.set_num_threads(12)
torch.backends.cudnn.benchmark = True

train_ds = LIBRISPEECH(root=root, url="test-clean", download=False)
val_ds   = LIBRISPEECH(root=root, url="test-clean", download=False)
test_ds  = LIBRISPEECH(root=root, url="test-clean", download=False)

small_train_ds = Subset(train_ds, range(200))
small_val_ds = Subset(val_ds, range(50))
small_test_ds = Subset(test_ds, range(50))

#initialize dataloader
train_loader = DataLoader(small_train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn_test, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(small_val_ds,   batch_size=64, shuffle=False, collate_fn=collate_fn_test, num_workers=2, persistent_workers=True)
test_loader  = DataLoader(small_test_ds,  batch_size=64, shuffle=False, collate_fn=collate_fn_test, num_workers=2, persistent_workers=True)


def train_model(B=5, R=5, num_epochs=10):
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Dataset sizes | train: {len(small_train_ds)} | val: {len(small_train_ds)} | test: {len(small_test_ds)}"
    )
    print(
        f"Dataloader batches | train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}"
    )
    print(f"Starting training for {num_epochs} epochs with QuartzNet B={B}, R={R}")

    model = QuartzNetBxR(n_mels=64, n_classes=29, B=B, R=R).to(device)
    optimizer = NovoGrad(model.parameters(), lr=0.01, betas=(0.95, 0.5), weight_decay=0.001)
    criterion = nn.CTCLoss(blank=28, zero_infinity=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Training started")

        for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            if batch_idx == 0:
                print(
                    f"First batch shapes | inputs: {tuple(inputs.shape)} | targets: {tuple(targets.shape)}"
                )
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            adjusted_lengths = ((input_lengths - 1) // 2) + 1
            loss = criterion(outputs, targets, adjusted_lengths, target_lengths)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % 5 == 0:
                running_avg_loss = total_train_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f} | "
                    f"Running Avg Loss: {running_avg_loss:.4f}",
                    end='\r'
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Training finished | Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        print(f"[Epoch {epoch+1}/{num_epochs}] Validation started")
        with torch.no_grad():
            for val_batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                adjusted_lengths = ((input_lengths - 1) // 2) + 1
                val_loss = criterion(outputs, targets, adjusted_lengths, target_lengths)
                total_val_loss += val_loss.item()

                if val_batch_idx % 5 == 0:
                    running_val_avg_loss = total_val_loss / (val_batch_idx + 1)
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Val Batch {val_batch_idx}/{len(val_loader)} | "
                        f"Val Loss: {val_loss.item():.4f} | "
                        f"Running Avg Val Loss: {running_val_avg_loss:.4f}",
                        end='\r'
                    )

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} complete | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )


    return model, train_losses, val_losses


if __name__ == "__main__":
    train_model(B=5 , R=2 ,num_epochs=10)
    