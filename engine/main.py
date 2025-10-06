from engine.search import best_move 
from board_utils import userMove
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import random


board = chess.Board()

moves = 0
while not board.is_game_over():
    moves+=1
    print(f"Move {moves}:")
    print(board)
    board.push(best_move(board, 3))
    print("\n")
