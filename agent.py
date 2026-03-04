import random
import time
from typing import Any, Dict, Optional, Tuple, List
from extension.board_utils import list_legal_moves_for
from extension.piece_right import Right

# Monkey patch to fix Right.name attribute error
try:
    # Check if Right.name is behaving as a property object instead of returning the string
    if isinstance(Right.__dict__.get('name'), (property, classmethod)) or isinstance(Right.name, property):
        # Force it to be a simple property that returns "Right"
        Right.name = property(lambda self: "Right")
except Exception:
    pass

# ---- Bitboard Constants & Helpers ----

# Board size
ROWS = 5
COLS = 5
SQUARES = 25

# Piece codes
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5
RIGHT = 6  # Custom piece: Knight + Rook

WHITE = 0
BLACK = 1

# Masks
MASK_ALL = (1 << SQUARES) - 1
FILE_A = 0x0108421  # Columns 0, 5, 10, 15, 20 -> 1 | 32 | 1024 ...
FILE_E = 0x1084210  # Columns 4, 9, 14, 19, 24
RANK_1 = 0x000001F  # 0-4
RANK_5 = 0x1F00000  # 20-24

# Precomputed attacks
KNIGHT_ATTACKS = [0] * SQUARES
KING_ATTACKS = [0] * SQUARES
PAWN_ATTACKS = [[0] * SQUARES, [0] * SQUARES] # [color][sq]

def _init_tables():
    for sq in range(SQUARES):
        y, x = divmod(sq, COLS)
        
        # Knight
        k_mask = 0
        for dy, dx in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < ROWS and 0 <= nx < COLS:
                k_mask |= (1 << (ny * COLS + nx))
        KNIGHT_ATTACKS[sq] = k_mask
        
        # King
        k_mask = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < ROWS and 0 <= nx < COLS:
                    k_mask |= (1 << (ny * COLS + nx))
        KING_ATTACKS[sq] = k_mask
        
        # Pawn Attacks (captures only)
        # White (UP, y+1)
        w_mask = 0
        if y + 1 < ROWS:
            if x - 1 >= 0: w_mask |= (1 << ((y + 1) * COLS + (x - 1)))
            if x + 1 < COLS: w_mask |= (1 << ((y + 1) * COLS + (x + 1)))
        PAWN_ATTACKS[WHITE][sq] = w_mask
        
        # Black (DOWN, y-1)
        b_mask = 0
        if y - 1 >= 0:
            if x - 1 >= 0: b_mask |= (1 << ((y - 1) * COLS + (x - 1)))
            if x + 1 < COLS: b_mask |= (1 << ((y - 1) * COLS + (x + 1)))
        PAWN_ATTACKS[BLACK][sq] = b_mask

_init_tables()

# Ray lookups for sliding pieces (Rook, Bishop, Queen)
RAYS = [[0] * 8 for _ in range(SQUARES)] # [sq][dir_idx]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] # N, S, W, E, NW, NE, SW, SE

def _init_rays():
    for sq in range(SQUARES):
        y, x = divmod(sq, COLS)
        for i, (dy, dx) in enumerate(DIRS):
            mask = 0
            cy, cx = y + dy, x + dx
            while 0 <= cy < ROWS and 0 <= cx < COLS:
                mask |= (1 << (cy * COLS + cx))
                cy += dy
                cx += dx
            RAYS[sq][i] = mask

_init_rays()

def get_sliding_attacks(sq, occ, is_diag):
    """
    Calculate sliding attacks for a square given occupancy.
    is_diag: True for Bishop, False for Rook. Queen uses both.
    """
    attacks = 0
    start = 4 if is_diag else 0
    end = 8 if is_diag else 4
    
    for i in range(start, end):
        ray = RAYS[sq][i]
        if not ray: continue
        blockers = ray & occ
        if not blockers:
            attacks |= ray
        else:
            # Find first blocker using bit manipulation
            if i % 2 == 0: # Decreasing (N, W, NW, SW) -> Scan for MSB (highest bit < sq)
                # Blocker is the largest bit in ray & occ
                b_idx = (blockers).bit_length() - 1
                attacks |= (ray & ~RAYS[b_idx][i])
            else: # Increasing (S, E, NE, SE) -> Scan for LSB (lowest bit > sq)
                # LSB: b = blockers & -blockers
                b_val = blockers & -blockers
                b_idx = b_val.bit_length() - 1
                attacks |= (ray & ~RAYS[b_idx][i])
                
    return attacks

# Zobrist Hashing
ZOBRIST_TABLE = {}
ZOBRIST_SIDE = 0
def _init_zobrist():
    global ZOBRIST_SIDE
    rng = random.Random(42)
    for sq in range(SQUARES):
        for pc in range(7): # 7 piece types
            for color in [WHITE, BLACK]:
                ZOBRIST_TABLE[(color, pc, sq)] = rng.getrandbits(64)
    ZOBRIST_SIDE = rng.getrandbits(64)

_init_zobrist()

class Bitboard:
    __slots__ = ('pieces', 'occupied', 'side_to_move', 'hash')
    def __init__(self):
        self.pieces = [[0] * 7 for _ in range(2)] # [color][piece_type]
        self.occupied = [0] * 2 # [color]
        self.side_to_move = WHITE
        self.hash = 0
        
    def copy(self):
        new_b = Bitboard()
        new_b.pieces = [p[:] for p in self.pieces]
        new_b.occupied = self.occupied[:]
        new_b.side_to_move = self.side_to_move
        new_b.hash = self.hash
        return new_b
        
    def get_piece_at(self, sq):
        mask = 1 << sq
        if not ((self.occupied[0] | self.occupied[1]) & mask):
            return None, None
        color = WHITE if (self.occupied[WHITE] & mask) else BLACK
        for ptype in range(7):
            if self.pieces[color][ptype] & mask:
                return color, ptype
        return None, None

# ---- Search Structures ----

class TTEntry:
    __slots__ = ('depth', 'score', 'flag', 'move')
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2
    
    def __init__(self, depth, score, flag, move):
        self.depth = depth
        self.score = score
        self.flag = flag
        self.move = move

class SearchState:
    def __init__(self, time_limit, max_nodes=5000000):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes = 0
        self.abort = False
        self.max_nodes = max_nodes
        self.tt = {} # Transposition Table
        self.history = {} # History Heuristic [side][from][to] -> score

    def time_up(self):
        if self.abort: return True
        if self.nodes % 2048 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.abort = True
        return self.abort

# ---- Agent Logic ----

def parse_board(board_obj) -> Bitboard:
    bb = Bitboard()
    bb.side_to_move = WHITE if board_obj.current_player.name.lower() == "white" else BLACK
    
    # Map piece names to codes
    name_map = {
        "pawn": PAWN, "knight": KNIGHT, "bishop": BISHOP, 
        "rook": ROOK, "right": RIGHT, "queen": QUEEN, "king": KING
    }
    
    for y in range(ROWS):
        for x in range(COLS):
            sq_obj = board_obj._squares[y][x]
            piece = sq_obj._piece
            if piece:
                color = WHITE if piece.player.name.lower() == "white" else BLACK
                
                # Safe name access
                p_name = ""
                try:
                    p_name = piece.name.lower()
                except AttributeError:
                    # Fallback for broken property in Right piece
                    if "Right" in str(type(piece)):
                        p_name = "right"
                    else:
                        p_name = "unknown"
                
                ptype = None
                for k, v in name_map.items():
                    if k in p_name:
                        ptype = v
                        break
                
                if ptype is not None:
                    sq = y * COLS + x
                    bb.pieces[color][ptype] |= (1 << sq)
                    bb.occupied[color] |= (1 << sq)
                    bb.hash ^= ZOBRIST_TABLE[(color, ptype, sq)]
    
    if bb.side_to_move == BLACK:
        bb.hash ^= ZOBRIST_SIDE
        
    return bb

def generate_moves_bitboard(bb: Bitboard):
    """Generate all pseudo-legal moves for the side to move."""
    moves = [] # (from_sq, to_sq, promotion_type)
    us = bb.side_to_move
    them = 1 - us
    occ_us = bb.occupied[us]
    occ_them = bb.occupied[them]
    occ_all = occ_us | occ_them
    
    # Pawns
    pawns = bb.pieces[us][PAWN]
    if pawns:
        # Push
        if us == WHITE:
            # Single push: +5
            single_push = (pawns << 5) & ~occ_all & MASK_ALL
            
            # Iterate single pushes
            p_temp = single_push
            while p_temp:
                to_sq = (p_temp & -p_temp).bit_length() - 1
                p_temp &= p_temp - 1
                from_sq = to_sq - 5
                
                # Promotion?
                if to_sq >= 20: # Rank 5 (indices 20-24)
                    moves.append((from_sq, to_sq, QUEEN))
                else:
                    moves.append((from_sq, to_sq, None))
                    # Double push from Rank 2 (indices 5-9) -> Rank 4 (15-19)
                    # If single push landed on Rank 3 (10-14), check next step
                    # Wait, Rank 1 (0-4) -> Rank 2 (5-9).
                    # Double push is from Rank 2 (indices 5-9) in 1-based indexing?
                    # No, standard chess is Rank 2. Here it's 5x5.
                    # Let's assume Rank 2 (index 1) is the pawn start rank.
                    # Indices 5-9.
                    if 5 <= from_sq <= 9:
                        to_sq2 = to_sq + 5
                        if not ((occ_all >> to_sq2) & 1):
                            moves.append((from_sq, to_sq2, None))

        else: # BLACK
            # Single push: -5
            single_push = (pawns >> 5) & ~occ_all
            
            p_temp = single_push
            while p_temp:
                to_sq = (p_temp & -p_temp).bit_length() - 1
                p_temp &= p_temp - 1
                from_sq = to_sq + 5
                
                # Promotion?
                if to_sq <= 4: # Rank 1 (indices 0-4)
                    moves.append((from_sq, to_sq, QUEEN))
                else:
                    moves.append((from_sq, to_sq, None))
                    # Double push from Rank 4 (indices 15-19) -> Rank 2
                    if 15 <= from_sq <= 19:
                        to_sq2 = to_sq - 5
                        if not ((occ_all >> to_sq2) & 1):
                            moves.append((from_sq, to_sq2, None))
                                
        # Captures
        p_temp = pawns
        while p_temp:
            from_sq = (p_temp & -p_temp).bit_length() - 1
            p_temp &= p_temp - 1
            
            attacks = PAWN_ATTACKS[us][from_sq] & occ_them
            while attacks:
                to_sq = (attacks & -attacks).bit_length() - 1
                attacks &= attacks - 1
                # Promotion?
                if (us == WHITE and to_sq >= 20) or (us == BLACK and to_sq <= 4):
                    moves.append((from_sq, to_sq, QUEEN))
                else:
                    moves.append((from_sq, to_sq, None))

    # Knights
    knights = bb.pieces[us][KNIGHT]
    while knights:
        from_sq = (knights & -knights).bit_length() - 1
        knights &= knights - 1
        attacks = KNIGHT_ATTACKS[from_sq] & ~occ_us
        while attacks:
            to_sq = (attacks & -attacks).bit_length() - 1
            attacks &= attacks - 1
            moves.append((from_sq, to_sq, None))

    # King
    kings = bb.pieces[us][KING]
    if kings:
        from_sq = (kings & -kings).bit_length() - 1
        attacks = KING_ATTACKS[from_sq] & ~occ_us
        while attacks:
            to_sq = (attacks & -attacks).bit_length() - 1
            attacks &= attacks - 1
            moves.append((from_sq, to_sq, None))

    # Sliding Pieces
    # Bishop
    bishops = bb.pieces[us][BISHOP]
    while bishops:
        from_sq = (bishops & -bishops).bit_length() - 1
        bishops &= bishops - 1
        attacks = get_sliding_attacks(from_sq, occ_all, True) & ~occ_us
        while attacks:
            to_sq = (attacks & -attacks).bit_length() - 1
            attacks &= attacks - 1
            moves.append((from_sq, to_sq, None))
            
    # Rook
    rooks = bb.pieces[us][ROOK]
    while rooks:
        from_sq = (rooks & -rooks).bit_length() - 1
        rooks &= rooks - 1
        attacks = get_sliding_attacks(from_sq, occ_all, False) & ~occ_us
        while attacks:
            to_sq = (attacks & -attacks).bit_length() - 1
            attacks &= attacks - 1
            moves.append((from_sq, to_sq, None))
            
    # Queen
    queens = bb.pieces[us][QUEEN]
    while queens:
        from_sq = (queens & -queens).bit_length() - 1
        queens &= queens - 1
        attacks = (get_sliding_attacks(from_sq, occ_all, True) | get_sliding_attacks(from_sq, occ_all, False)) & ~occ_us
        while attacks:
            to_sq = (attacks & -attacks).bit_length() - 1
            attacks &= attacks - 1
            moves.append((from_sq, to_sq, None))
            
    # Right (Knight + Rook)
    rights = bb.pieces[us][RIGHT]
    while rights:
        from_sq = (rights & -rights).bit_length() - 1
        rights &= rights - 1
        attacks = (KNIGHT_ATTACKS[from_sq] | get_sliding_attacks(from_sq, occ_all, False)) & ~occ_us
        while attacks:
            to_sq = (attacks & -attacks).bit_length() - 1
            attacks &= attacks - 1
            moves.append((from_sq, to_sq, None))
            
    return moves

def apply_move(bb: Bitboard, move):
    """Returns a NEW bitboard with move applied."""
    new_bb = bb.copy()
    from_sq, to_sq, promo = move
    us = bb.side_to_move
    them = 1 - us
    
    # Identify piece moving
    ptype = None
    for p in range(7):
        if new_bb.pieces[us][p] & (1 << from_sq):
            ptype = p
            break
            
    # Remove from source
    mask_from = 1 << from_sq
    new_bb.pieces[us][ptype] &= ~mask_from
    new_bb.occupied[us] &= ~mask_from
    new_bb.hash ^= ZOBRIST_TABLE[(us, ptype, from_sq)]
    
    # Handle capture
    mask_to = 1 << to_sq
    if new_bb.occupied[them] & mask_to:
        for p in range(7):
            if new_bb.pieces[them][p] & mask_to:
                new_bb.pieces[them][p] &= ~mask_to
                new_bb.hash ^= ZOBRIST_TABLE[(them, p, to_sq)]
                break
        new_bb.occupied[them] &= ~mask_to
        
    # Place at dest
    final_type = promo if promo is not None else ptype
    new_bb.pieces[us][final_type] |= mask_to
    new_bb.occupied[us] |= mask_to
    new_bb.hash ^= ZOBRIST_TABLE[(us, final_type, to_sq)]
    
    # Switch side
    new_bb.side_to_move = them
    new_bb.hash ^= ZOBRIST_SIDE
    
    return new_bb

# Evaluation
PIECE_VALUES = [100, 320, 330, 500, 900, 20000, 800] # P, N, B, R, Q, K, Right
# PSQTs (flattened)
# ... (Implement simplified PSQTs)

def evaluate_bb(bb: Bitboard):
    score = 0
    
    # Material & Position
    for color in [WHITE, BLACK]:
        sign = 1 if color == WHITE else -1
        for ptype in range(7):
            pieces = bb.pieces[color][ptype]
            val = PIECE_VALUES[ptype]
            while pieces:
                sq = (pieces & -pieces).bit_length() - 1
                pieces &= pieces - 1
                
                # Simple centrality bonus
                y, x = divmod(sq, COLS)
                dist = abs(y - 2) + abs(x - 2)
                pos_bonus = (4 - dist) * 5
                
                score += sign * (val + pos_bonus)
                
    # Perspective
    if bb.side_to_move == BLACK:
        score = -score
    return score

# ---- Search Logic ----

def quiescence_bb(bb: Bitboard, alpha, beta, state: SearchState):
    state.nodes += 1
    if state.time_up(): return 0

    stand_pat = evaluate_bb(bb)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # Generate captures only
    moves = generate_moves_bitboard(bb)
    them = 1 - bb.side_to_move
    occ_them = bb.occupied[them]
    
    captures = []
    for m in moves:
        from_sq, to_sq, promo = m
        if (1 << to_sq) & occ_them:
            captures.append(m)
            
    # MVV-LVA Sorting
    def mvv_lva(move):
        from_sq, to_sq, _ = move
        # Victim
        victim_val = 0
        for p in range(7):
            if bb.pieces[them][p] & (1 << to_sq):
                victim_val = PIECE_VALUES[p]
                break
        # Attacker
        attacker_val = 0
        us = bb.side_to_move
        for p in range(7):
            if bb.pieces[us][p] & (1 << from_sq):
                attacker_val = PIECE_VALUES[p]
                break
        return victim_val * 10 - attacker_val

    captures.sort(key=mvv_lva, reverse=True)
    
    for move in captures:
        new_bb = apply_move(bb, move)
        score = -quiescence_bb(new_bb, -beta, -alpha, state)
        
        if state.abort: return 0
        
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
            
    return alpha

def alphabeta_bb(bb: Bitboard, depth, alpha, beta, state: SearchState):
    state.nodes += 1
    if state.time_up(): return 0
    
    # TT Probe
    if bb.hash in state.tt:
        entry = state.tt[bb.hash]
        if entry.depth >= depth:
            if entry.flag == TTEntry.EXACT: return entry.score
            if entry.flag == TTEntry.LOWERBOUND: alpha = max(alpha, entry.score)
            if entry.flag == TTEntry.UPPERBOUND: beta = min(beta, entry.score)
            if alpha >= beta: return entry.score

    if depth <= 0:
        return quiescence_bb(bb, alpha, beta, state)
        
    moves = generate_moves_bitboard(bb)
    if not moves:
        # Checkmate or Stalemate?
        # We need to check if King is in check.
        # For now, assume loss if no moves (stalemate = loss in this variant?)
        return -20000 + depth # Prefer later loss
        
    # Move Ordering
    # 1. TT Move
    # 2. Captures (MVV-LVA)
    # 3. History?
    
    tt_move = None
    if bb.hash in state.tt:
        tt_move = state.tt[bb.hash].move
        
    them = 1 - bb.side_to_move
    occ_them = bb.occupied[them]
    
    def score_move(move):
        if move == tt_move: return 1000000
        
        from_sq, to_sq, promo = move
        score = 0
        
        # Capture?
        if (1 << to_sq) & occ_them:
            # Victim
            victim_val = 0
            for p in range(7):
                if bb.pieces[them][p] & (1 << to_sq):
                    victim_val = PIECE_VALUES[p]
                    break
            # Attacker
            attacker_val = 0
            us = bb.side_to_move
            for p in range(7):
                if bb.pieces[us][p] & (1 << from_sq):
                    attacker_val = PIECE_VALUES[p]
                    break
            score = 10000 + victim_val * 10 - attacker_val
        
        # Promotion?
        if promo is not None:
            score += 9000
            
        return score

    moves.sort(key=score_move, reverse=True)
    
    best_val = -float('inf')
    best_move = None
    flag = TTEntry.UPPERBOUND
    original_alpha = alpha
    
    for move in moves:
        new_bb = apply_move(bb, move)
        val = -alphabeta_bb(new_bb, depth - 1, -beta, -alpha, state)
        
        if state.abort: return 0
        
        if val > best_val:
            best_val = val
            best_move = move
            
        alpha = max(alpha, val)
        if alpha >= beta:
            flag = TTEntry.LOWERBOUND
            break
            
    # TT Store
    if not state.abort:
        if best_val > original_alpha and best_val < beta:
            flag = TTEntry.EXACT
        elif best_val >= beta:
            flag = TTEntry.LOWERBOUND
        else:
            flag = TTEntry.UPPERBOUND
            
        state.tt[bb.hash] = TTEntry(depth, best_val, flag, best_move)
        
    return best_val

def iterative_deepening_bb(bb: Bitboard, state: SearchState, max_depth=20):
    best_move = None
    
    for depth in range(1, max_depth + 1):
        if state.time_up(): break
        
        # Aspiration Windows? Maybe later.
        val = alphabeta_bb(bb, depth, -float('inf'), float('inf'), state)
        
        if state.abort: break
        
        # Get best move from TT
        if bb.hash in state.tt:
            best_move = state.tt[bb.hash].move
            
        # print(f"Depth {depth} finished. Score: {val}, Move: {best_move}")
        
    return best_move

# Wrapper
def agent(board, player, var):
    # Parse time
    time_limit = 0.5
    if isinstance(var, (list, tuple)):
        time_limit = float(var[1]) if len(var) > 1 else 0.5
    else:
        time_limit = float(var) if var else 0.5
        
    # Safety margin
    search_time = max(0.05, time_limit - 0.05)
    
    state = SearchState(time_limit=search_time)
    
    # Convert to Bitboard
    bb = parse_board(board)
    
    # Run Search
    best_move_bb = iterative_deepening_bb(bb, state)
    
    # Map back to object
    if best_move_bb:
        from_sq, to_sq, promo = best_move_bb
        
        # Find piece at from_sq
        fy, fx = divmod(from_sq, COLS)
        ty, tx = divmod(to_sq, COLS)
        
        found_piece = None
        found_move = None
        
        # Note: board.get_player_pieces(player)
        for p in board.get_player_pieces(player):
            if p.position.x == fx and p.position.y == fy:
                found_piece = p
                break
        
        if found_piece:
            # Find the move option
            for opt in found_piece.get_move_options():
                if opt.position.x == tx and opt.position.y == ty:
                    found_move = opt
                    break
            
            if found_piece and found_move:
                return found_piece, found_move

    # Fallback
    legal = list_legal_moves_for(board, player)
    if legal:
        return legal[0][0], list(legal[0][1])[0]
    return None, None
