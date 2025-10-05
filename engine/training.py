import argparse
import os
from typing import List

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess.pgn

# --- Import model and utilities safely ---
try:
    from .cnn import ChessEvaluator
    from .board_utils import board_to_tensor
except ImportError:
    # fallback if running as a standalone script
    from cnn import ChessEvaluator
    from board_utils import board_to_tensor


# --- Dataset Definition ---
class ChessDataset(Dataset):
    def __init__(self, pgn_path: str, max_games: int = None, skip_openings: int = 0):
        self.positions: List[torch.Tensor] = []
        self.labels: List[float] = []

        if not os.path.exists(pgn_path):
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")

        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            games_read = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games_read += 1
                if max_games is not None and games_read > max_games:
                    break

                result = game.headers.get("Result", "*")
                if result == "1-0":
                    outcome = 1.0
                elif result == "0-1":
                    outcome = -1.0
                elif result == "1/2-1/2":
                    outcome = 0.0
                else:
                    continue  # skip unplayed/incomplete games

                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    if ply >= skip_openings:
                        t = board_to_tensor(board)
                        if not isinstance(t, torch.Tensor):
                            t = torch.tensor(t, dtype=torch.float32)
                        else:
                            t = t.to(dtype=torch.float32)
                        self.positions.append(t)
                        self.labels.append(outcome)
                    board.push(move)
                    ply += 1

        if len(self.positions) == 0:
            raise ValueError("No positions were extracted from the PGN file.")

        # Turn lists into tensors
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.labels[idx]


# --- Data Loader Collation ---
def collate_fn(batch):
    boards, labels = zip(*batch)
    boards = torch.stack(boards, dim=0)
    labels = torch.stack(labels, dim=0)
    return boards, labels


# --- Device Selection ---
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


# --- Training Loop ---
def train(pgn_path, out_path="chess_model.pth", epochs=5, batch_size=32, lr=1e-4, max_games=None, skip_openings=0):
    device = get_device()
    print("Using device:", device)

    dataset = ChessDataset(pgn_path, max_games=max_games, skip_openings=skip_openings)
    print(f"Loaded dataset with {len(dataset)} positions.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(2, os.cpu_count())
    )

    model = ChessEvaluator(input_planes=18).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for i, (boards, labels) in enumerate(dataloader, start=1):
            boards, labels = boards.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(boards)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 0 or i == len(dataloader):
                print(f"Epoch {epoch} | Step {i}/{len(dataloader)} | Avg Loss: {running_loss / i:.6f}")

        avg_epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.6f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_epoch_loss
        }, out_path)
        print(f"Checkpoint saved: {out_path}")

    print("Training complete.")


# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a chess CNN evaluator on PGN data.")
    parser.add_argument("--pgn", required=True, help="Path to PGN file with games")
    parser.add_argument("--out", default="chess_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games for quick testing")
    parser.add_argument("--skip-openings", type=int, default=0, help="Skip first N plies of every game")
    args = parser.parse_args()

    train(
        args.pgn,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_games=args.max_games,
        skip_openings=args.skip_openings
    )
