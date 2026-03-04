from extension.board_utils import list_legal_moves_for
from extension.board_rules import get_result
from extension.board_utils import print_board_ascii, copy_piece_move
import random
from functools import lru_cache
import time

"""
Killer moves show big improvements in pruning efficiency for higher depths

Added MVA-LVA heuristic for captures in quiescence search
They cause slow down but improve pruning efficiency significantly
Without checking for win vs loss they lead to stupid fivolds etc.

Added check for win vs loss in minimax terminal state evaluation
Improves "smartness" significantly as agent avoids losing lines better

"""


piece_values = {
    'King': 1000,
    'Queen': 9,
    'Bishop': 4,
    'Knight': 4,
    'Pawn': 1,
    'Right': 10
}


# Global Variables
DEPTH = 11
INF = 10**9
DEADLINE = None

# Killer moves to improve move ordering => alpha-beta pruning
# KILLER_MOVES[depth] = [move_id1, move_id2]
# Avooid errors with depth greater than initial size
KILLER_MOVES = [[None, None] for _ in range(50)]

# Testing variables
NODES = 0
MAX_DEPTH_REACHED = 0
PRUNE_COUNT = 0


def opponent(board, player, var):
    piece, move_opt = None, None
    
    if var:
        ply = var[0]
        budget = var[1]
        print(f"Ply: {ply}")
    else:
        ply = 10
        budget = 14  # Default thinking time budget

    piece_count = sum(1 for _ in board.get_pieces()) + 1
    depth = 10

    print(f"{depth} computed depth")

    legal = list_legal_moves_for(board, player)
    if legal:
         piece, move_opt = choose_Move(board, player, thinking_budget = budget,  max_depth=depth)
        
    return piece, move_opt
    
    return None

def evaluate(board):
    # Simple evaluation by piece values
    p0 = board.players[0]
    score = 0
    
    for piece in board.get_pieces():
        val = piece_values[piece.name]
        if piece.player == p0:
            score += val
        else:
            score -= val
    return score


def minimax(board, depth, alpha, beta, maximizing_player, ply):
    '''
    Minimax with board as a state representation
    Alpha Beta pruning for decreasing search tree
    Quiescence search to check for stupid captures
    '''
    global NODES, PRUNE_COUNT, KILLER_MOVES

    # Only checking time every 100 nodes
    if NODES % 100 == 0 and time.time() >= DEADLINE:
        raise TimeoutError("Thinking Budget Exceeded")

    result = get_result(board)
    if result:
        # Check if the result string indicates a Win, Loss, or Draw
        # We add/subtract 'depth' to prefer faster wins and slower losses
        if "wins" in result:
            if "white" in result: # White wins
                return INF + depth 
            else:              # Black wins
                return -INF - depth
                
        elif "loses" in result:
            if "white" in result: # White loses
                return -INF - depth
            else:              # Black loses
                return INF + depth
                
        elif "Draw" in result or "Stalemate" in result:
             # Draw is 0. Better than losing, worse than winning.
             return 0
             
        # Fallback for unknown strings
        return evaluate(board)

    if depth == 0:
        return quiescence(board, alpha, beta, maximizing_player)


    if maximizing_player:
        maxEval = -INF
        legal = move_order(board, board.players[0], ply)  # white is maximizing player

        for piece, move_opt in legal:
            NODES += 1

            temp_board = board.clone()
            temp_board, temp_piece, temp_move_opt = copy_piece_move(temp_board, piece, move_opt)
            
            temp_piece.move(temp_move_opt)
            
            eval = minimax(temp_board, depth - 1, alpha, beta, False, ply+1)
            
            if eval > maxEval:
                maxEval = eval
            
            alpha = max(alpha, eval)
            if beta <= alpha:
                PRUNE_COUNT += 1
                # Store unique killer moves
                store_killer(ply, piece, move_opt)
                
                break
        best_value = maxEval
    
    else:
        minEval = INF
        legal = move_order(board, board.players[1], ply)  # black is minimizing player
        for piece, move_opt in legal:
            NODES += 1
            
            temp_board = board.clone()
            temp_board, temp_piece, temp_move_opt = copy_piece_move(temp_board, piece, move_opt)

            temp_piece.move(temp_move_opt)
            
            eval = minimax(temp_board, depth - 1, alpha, beta, True, ply+1)
            
            if eval < minEval:
                minEval = eval

            beta = min(beta, eval)
            if beta <= alpha:
                PRUNE_COUNT += 1
                store_killer(ply, piece, move_opt)
                break
        best_value = minEval

    return best_value


def choose_Move(board, player, thinking_budget, max_depth, ply=0):
    """
    Chooses move using iterative deepening on minimax
    """
    global NODES, PRUNE_COUNT, MAX_DEPTH_REACHED, DEADLINE, KILLER_MOVES

    KILLER_MOVES = [[None, None] for _ in range(50)]
    NODES = 0
    MAX_DEPTH_REACHED = 0
    PRUNE_COUNT = 0

    is_white = (player.name == board.players[0].name)
    legal = move_order(board, player, ply)
    # In case we terminate without single iteration of minimax
    global_best_move = legal[0]

    start = time.time()
    deadline = start + 0.80 * thinking_budget
    DEADLINE = deadline

    for depth in range(1, max_depth + 1):
        # Testing variables
        
        current_depth_best_move = None
        current_depth_best_value = -INF if is_white else INF
        

        depth_completed = True
        try:
            for piece, move_opt in legal:
                if time.time() >= deadline:
                    depth_completed = False
                    break   

                temp_board = board.clone()
                temp_board, temp_piece, temp_move_opt = copy_piece_move(temp_board, piece, move_opt)

                temp_piece.move(temp_move_opt)
                NODES += 1

                
                board_value = minimax(temp_board, depth - 1, -INF, INF, not is_white, ply=1)

                if is_white:
                    if board_value > current_depth_best_value:
                        current_depth_best_value = board_value
                        current_depth_best_move = (piece, move_opt)
                else:
                    if board_value < current_depth_best_value:
                        current_depth_best_value = board_value
                        current_depth_best_move = (piece, move_opt)
        except TimeoutError:
            print("Timeout during depth", depth)
            depth_completed = False
            break
        

        if depth_completed and current_depth_best_move:
            MAX_DEPTH_REACHED = depth
            global_best_move = current_depth_best_move
            print(f"Depth {depth} finished. Current depth best score: {current_depth_best_value}")
            
            # PV-Sorting: Sort moves so next depth checks the best move first
            legal.sort(key=lambda pm: 0 if pm == global_best_move else 1)
            store_killer(ply, global_best_move[0], global_best_move[1])
        else:
            print(f"Depth {depth} timed out. Using result from Depth {depth-1}")
            break
    print(f"Nodes searched: {NODES}, Max Depth Reached: {MAX_DEPTH_REACHED}, Prunes: {PRUNE_COUNT}, Prune Rate: {PRUNE_COUNT / (NODES+1) :.2%}")

    return global_best_move


def quiescence(board, alpha, beta, maximizing_player, depth = 0):
    """
    Simple Quiescence search with depth limit
    Helps to avoid stupid captures 
    """
    global NODES, PRUNE_COUNT

    best_value = evaluate(board)

    if depth > 1: 
        return best_value
        
    if NODES % 100 == 0 and time.time() >= DEADLINE:
        raise TimeoutError("Timeout during quiescence search")

    if maximizing_player:
        if best_value >= beta:
            # PRUNE_COUNT += 1
            return beta 
        if best_value > alpha:
            alpha = best_value

        current_player = board.players[0]

    else:
        if best_value <= alpha:
            # PRUNE_COUNT += 1
            return alpha 
        if best_value < beta:
            beta = best_value

        current_player = board.players[1]

    # We only check captures in quiescence search
    legal = list_legal_moves_for(board, current_player)
    captures = [(p, m) for p, m in legal if m.captures]
    
    if not captures:
        return best_value


    captures.sort(key=lambda m: move_score (m, board), reverse=True)

    for piece, move_opt in captures:
        # NODES += 1
        
        temp_board = board.clone()
        temp_board, temp_piece, temp_move_opt = copy_piece_move(temp_board, piece, move_opt)
        temp_piece.move(temp_move_opt)
        
        score = quiescence(temp_board, alpha, beta, not maximizing_player, depth + 1)

        if maximizing_player:
            if score >= beta:
                # PRUNE_COUNT += 1
                return beta
            if score > alpha:
                alpha = score
        else:
            if score <= alpha:
                # PRUNE_COUNT += 1
                return alpha
            if score < beta:
                beta = score
    
    return alpha if maximizing_player else beta


def get_move_id(piece, move_option):
    """
    Id for each move for killer move heuristic, regardless if it's clone of board
    Returns (start_x, start_y, end_x, end_y)
    """
    return (piece.position.x, piece.position.y, move_option.position.x, move_option.position.y)


def store_killer(ply, piece, move_option):
    """
    Helper function to store killer moves
    If new killer is different from primary, shift primary to secondary
    """
    move_id = get_move_id(piece, move_option)

    # Only store unique killer moves
    if KILLER_MOVES[ply][0] != move_id:
        KILLER_MOVES[ply][1] = KILLER_MOVES[ply][0]
        KILLER_MOVES[ply][0] = move_id


def move_order(board, player, ply=None):
    """
    Simple move ordering based on move score
    """
    legal = list_legal_moves_for(board, player)
    legal.sort(key=lambda m: move_score (m, board, ply), reverse=True)

    return legal
    

def move_score(move_tuple, board, ply=None):
    """
    for move score  function to improve alpha-beta pruning efficiency.
    Extra promote > capture > check > others
    + Killer move heuristic
    """
    piece, move_option = move_tuple
    score = 0
    
    # MVV-LVA to sort captures
    if move_option.captures:
        aggressor_val = piece_values[piece.name]
        victim_val = 0
        for pos in move_option.captures:
            # Direct access to _squares [y][x]
            try:
                # The source code defines _squares as list[list[Square | None]]
                target_square = board._squares[pos.y][pos.x]
                
                # Check if square exists and has a piece
                if target_square is not None and target_square.piece is not None:
                    v_val = piece_values[target_square.piece.name]
                    if v_val > victim_val:
                        victim_val = v_val
                        
            except (AttributeError, IndexError):
                # For edge cases or non-standard boards
                pass
        score += 2000 + (victim_val * 10 ) - aggressor_val
    
    # prioritize promotions
    if move_option.extra.get('promote'):
        score += 5000

    if move_option.extra.get('check'):
        score += 500

    # Killer move heuristic
    current_id = get_move_id(piece, move_option)

    if ply: 
        if current_id == KILLER_MOVES[ply][0]:
            score += 900 
        elif current_id == KILLER_MOVES[ply][1]:
            score += 800
    
    return score