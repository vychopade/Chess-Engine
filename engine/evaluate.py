import os
import torch
import chess
from board_utils import board_to_tensor
from cnn import ChessEvaluator

_device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else
                       "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "chess_cnn.pth")

#load 
checkpoint = torch.load(model_path, map_location=_device)
_model = ChessEvaluator(input_planes=18).to(_device)

if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    _model.load_state_dict(checkpoint["model_state"])
else:
    _model.load_state_dict(checkpoint)
_model.eval()

#arbitrary??
_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def evaluate_board(board: chess.Board) -> float:
    t = board_to_tensor(board).unsqueeze(0).to(_device)
    with torch.no_grad():
        score = _model(t).item()  #alr scaled
    return float(score)

def material_eval(board: chess.Board) -> float:
    material = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        material += sign * _PIECE_VALUES.get(piece.piece_type, 0)
    scaled = material / 3000.0  # heuristic scaling
    if scaled > 1.0:
        scaled = 1.0
    if scaled < -1.0:
        scaled = -1.0
    return float(scaled)

def blended_eval(board: chess.Board, net_weight: float = 0.8) -> float:
    net = evaluate_board(board)
    mat = material_eval(board)
    return float(net_weight * net + (1.0 - net_weight) * mat)
