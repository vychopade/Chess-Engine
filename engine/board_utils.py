import numpy as np
import chess
import random
import torch

def board_to_tensor(board: chess.Board):
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        idx = piece_map[piece.piece_type] + (0 if piece.color else 6)
        planes[idx, row, col] = 1

    planes[12, :, :] = 1 if board.turn == chess.WHITE else 0

    if board.has_kingside_castling_rights(chess.WHITE): planes[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE): planes[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK): planes[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK): planes[16, :, :] = 1

    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        planes[17, row, col] = 1

    return torch.tensor(planes)

def tensor_to_board(tensor: torch.Tensor):
    if isinstance(tensor, torch.Tensor):
        planes = tensor.cpu().numpy()
    else:
        planes = tensor

    board = chess.Board.empty()  

    piece_map = {
        0: chess.PAWN,
        1: chess.KNIGHT,
        2: chess.BISHOP,
        3: chess.ROOK,
        4: chess.QUEEN,
        5: chess.KING,
    }

    for idx in range(12):
        piece_type = piece_map[idx % 6]
        color = chess.WHITE if idx < 6 else chess.BLACK
        for row in range(8):
            for col in range(8):
                if planes[idx, row, col] == 1:
                    square = row * 8 + col
                    board.set_piece_at(square, chess.Piece(piece_type, color))

    if planes[12, 0, 0] == 1:  
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK

    rights = 0
    if planes[13, 0, 0] == 1: rights |= chess.BB_H1  # White kingside
    if planes[14, 0, 0] == 1: rights |= chess.BB_A1  # White queenside
    if planes[15, 0, 0] == 1: rights |= chess.BB_H8  # Black kingside
    if planes[16, 0, 0] == 1: rights |= chess.BB_A8  # Black queenside
    board.castling_rights = rights

    # En passant
    ep_square = None
    if planes[17].sum() > 0:
        row, col = np.argwhere(planes[17] == 1)[0]
        ep_square = row * 8 + col
    board.ep_square = ep_square

    return board

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    value = 0
    for piece_type in piece_values:
        value += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return value

def userMove(board: chess.Board):
    while True:
        try:
            user_input = input("Enter move: ")
            move = chess.Move.from_uci(user_input)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move. Try again.")
        except Exception as e:
            print(f"Error: {e}. Please enter a valid move.")
