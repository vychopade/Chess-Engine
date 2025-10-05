import chess
from evaluate import blended_eval
import time
from typing import Optional, Dict, Tuple


_transposition_table: Dict[Tuple[int,int], Tuple[float,str]] = {}

QUIESCENCE = True       
QUIESCENCE_MAX = 6     
NET_WEIGHT = 0.85     
MAX_NODES = 2000000      #node limit

_nodes_searched = 0

def _sort_moves(board: chess.Board, moves):
    scored = []
    for move in moves:
        board.push(move)
        score = blended_eval(board, net_weight=NET_WEIGHT)
        board.pop()
        scored.append((score, move))
    if board.turn == chess.WHITE:
        scored.sort(key=lambda x: x[0], reverse=True)
    else:
        scored.sort(key=lambda x: x[0])
    return [m for (_, m) in scored]

def quiescence_search(board: chess.Board, alpha: float, beta: float, ply: int) -> float:
    global _nodes_searched
    _nodes_searched += 1
    stand_pat = blended_eval(board, net_weight=NET_WEIGHT)
    if stand_pat >= beta:
        return stand_pat
    if alpha < stand_pat:
        alpha = stand_pat

    if ply >= QUIESCENCE_MAX:
        return stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    if not captures:
        return stand_pat

    captures = _sort_moves(board, captures)
    for move in captures:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, ply + 1)  
        board.pop()
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha

def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
    global _transposition_table, _nodes_searched
    _nodes_searched += 1
    # Node limit
    if _nodes_searched > MAX_NODES:
        return blended_eval(board, net_weight=NET_WEIGHT)

    # terminal
    if depth == 0:
        if QUIESCENCE:
            return quiescence_search(board, alpha, beta, 0)
        return blended_eval(board, net_weight=NET_WEIGHT)

    if board.is_game_over():
        if board.is_checkmate():
            #if in checkmate cant make this move
            return -1.0 if board.turn == chess.WHITE else 1.0
        else:
            return 0.0

    zkey = board.transposition_key()
    tt_value = _transposition_table.get((zkey, depth))
    if tt_value is not None:
        val, flag = tt_value
        if flag == "EXACT":
            return val
        elif flag == "LOWERBOUND":
            alpha = max(alpha, val)
        elif flag == "UPPERBOUND":
            beta = min(beta, val)
        if alpha >= beta:
            return val

    legal_moves = list(board.legal_moves)
    # move ordering
    ordered_moves = _sort_moves(board, legal_moves)

    if maximizing_player:
        value = float('-inf')
        for move in ordered_moves:
            board.push(move)
            child_val = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if child_val > value:
                value = child_val
            alpha = max(alpha, child_val)
            if alpha >= beta:
                break
    else:
        value = float('inf')
        for move in ordered_moves:
            board.push(move)
            child_val = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if child_val < value:
                value = child_val
            beta = min(beta, child_val)
            if beta <= alpha:
                break

    #simple bounds
    flag = "EXACT"
    if value <= alpha:
        flag = "UPPERBOUND"
    elif value >= beta:
        flag = "LOWERBOUND"
    _transposition_table[(zkey, depth)] = (value, flag)

    return value

def best_move(board: chess.Board, depth: int, net_weight: float = NET_WEIGHT, time_limit: Optional[float] = None) -> chess.Move:
    global _transposition_table, _nodes_searched
    _transposition_table.clear()
    _nodes_searched = 0

    maximizing = True if board.turn == chess.WHITE else False

    best_mv = None
    if maximizing:
        best_value = float('-inf')
    else:
        best_value = float('inf')

    legal_moves = list(board.legal_moves)
    ordered_moves = _sort_moves(board, legal_moves)

    start_time = time.time()
    for move in ordered_moves:
        board.push(move)
        val = minimax(board, depth - 1, float('-inf'), float('inf'), not maximizing)
        board.pop()

        if maximizing:
            if val > best_value:
                best_value = val
                best_mv = move
        else:
            if val < best_value:
                best_value = val
                best_mv = move

        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

    return best_mv
