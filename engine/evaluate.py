import torch
import chess
from board_utils import board_to_tensor  
from cnn import ChessEvaluator
import os

_device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else
                       "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "chess_cnn.pth")

checkpoint = torch.load(model_path, map_location=_device)
_model = ChessEvaluator(input_planes=18).to(_device)

if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    _model.load_state_dict(checkpoint["model_state"])
else:
    _model.load_state_dict(checkpoint)
_model.eval()

def evaluate_board(board: chess.Board) -> float:
    t = board_to_tensor(board).unsqueeze(0).to(_device) 
    with torch.no_grad():
        score = _model(t).item()
    return score
