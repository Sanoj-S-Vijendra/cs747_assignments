from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
import math

class Task1Agent(BaseAgent):
    def __init__(self, name="Task1Agent"):
        super().__init__(name)
        self.piece_val = {0: 100, 1: 320, 2: 330, 3: 500, 4: 900, 5: 20000}
        self.trans_tbl = {}
        self.pawn_w = [[0,0,0,0],[30,30,30,30],[20,20,20,20],[10,10,10,10],[0,0,0,0]]
        self.pawn_b = [[0,0,0,0],[-10,-10,-10,-10],[-20,-20,-20,-20],[-30,-30,-30,-30],[0,0,0,0]]
        self.knight = [[1,2,2,1],[2,4,4,2],[2,4,4,2],[1,2,2,1],[0,1,1,0]]
        self.bishop = [[1,2,2,1],[2,3,3,2],[2,3,3,2],[1,2,2,1],[1,1,1,1]]

    def move(self, chess_obj:Chess):
        ### Your code goes here ###
        ''' right now the behaves similar to a random agent, picking moves uniformly randomly, change it to 
        design your agent
        '''
        # moves, proms = chess_obj.legal_moves()
        # legal_moves = piece_matrix_to_legal_moves(moves, proms)
        # move = random.choice(legal_moves)
        # return move
        search_depth = 2
        best_move = self.find_best_move(chess_obj, search_depth)
        return best_move

    ### Any other utility functions you want to define for your agent.

    def evaluate_board(self, chess_obj):
        """
        Evaluates the current board state.
        Returns a score from White's perspective (+ is good for White, - for Black).
        """
        result = chess_obj.game_result()
        if result is not None:
            if(result==1): return 1000000  # White win
            if(result==-1): return -1000000 # Black win
            if(result==0): return 0          # Draw
        score = 0
        for i in range(5):
            for j in range(4):
                piece,colour = chess_obj.any_piece_at(i,j)
                if(piece!=-1):
                    piece_val = self.piece_val[piece]
                    pos_val = 0
                    if(piece==0):
                        pos_val = self.pawn_w[i][j] if colour == 1 else self.pawn_b[i][j]
                    elif(piece == 1):
                        pos_val = self.knight[i][j]
                    elif(piece==2):
                        pos_val = self.bishop[i][j]
                    if colour == 1:
                        score+=(piece_val+pos_val)
                    else:
                        score-=(piece_val-pos_val)
        return score
    
    def move_score(self, chess_obj, move):
        (i,j),(dx,dy),promo = move
        # Promotion highest priority
        if(promo!=-1):
            return 1000+self.piece_val[promo]
        victim, _ = chess_obj.any_piece_at(i+dx,j+dy)
        if(victim!=-1):
            return 500+self.piece_val[victim]
        return 0

    def minimax(self, chess_obj, depth, alpha, beta):
        """
        Recursive Minimax function with Alpha-Beta Pruning
        and Transposition Table.
        """
        # unique hashable key for the current state
        tt_key = (chess_obj.bitboards.tobytes(), chess_obj.turn, depth)
        if tt_key in self.trans_tbl:
            return self.trans_tbl[tt_key]
        game_res = chess_obj.game_result()
        if game_res is not None:
            if(game_res==1): return 1000000-(10-depth) # prefer faster wins
            if(game_res==-1): return -1000000+(10-depth) # prefer slower losses
            if(game_res==0): return 0
            
        if depth == 0:
            return self.evaluate_board(chess_obj)
        moves,proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves,proms)
        if not legal_moves:
            return 0
        sorted_moves = sorted(legal_moves,key=lambda m: self.move_score(chess_obj,m),reverse=True)
        is_max_player = (chess_obj.turn==1)
        if is_max_player:
            max_eval = -math.inf
            for move in sorted_moves:
                (i,j),(dx,dy),promo = move
                next_state = chess_obj.copy()
                next_state.make_move(i,j,dx,dy,promo)
                eval = self.minimax(next_state,depth-1,alpha,beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if(beta<=alpha):
                    break
            self.trans_tbl[tt_key] = max_eval
            return max_eval
        else:
            min_eval = math.inf
            for move in sorted_moves:
                (i,j),(dx,dy),promo = move
                next_state = chess_obj.copy()
                next_state.make_move(i,j,dx,dy,promo)
                eval = self.minimax(next_state,depth-1,alpha,beta)
                min_eval = min(min_eval,eval)
                beta = min(beta,eval)
                if(beta<=alpha):
                    break
            self.trans_tbl[tt_key] = min_eval
            return min_eval

    def find_best_move(self, chess_obj, depth):
        """
        Iterates through root moves and calls minimax.
        """
        moves,proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        if not legal_moves:
            return None
        sorted_moves = sorted(legal_moves,key=lambda m: self.move_score(chess_obj,m),reverse=True)
        best_move = sorted_moves[0]
        is_max_player = (chess_obj.turn == 1)
        if is_max_player:
            best_eval = -math.inf
            alpha = -math.inf
            beta = math.inf
            for move in sorted_moves:
                (i,j),(dx,dy),promo = move
                next_state = chess_obj.copy()
                next_state.make_move(i,j,dx,dy,promo)
                eval = self.minimax(next_state,depth-1,alpha,beta)
                if(eval>best_eval):
                    best_eval = eval
                    best_move = move
                alpha = max(alpha,eval)
        else:
            best_eval = math.inf
            alpha = -math.inf
            beta = math.inf
            for move in sorted_moves:
                (i,j),(dx,dy),promo = move
                next_state = chess_obj.copy()
                next_state.make_move(i,j,dx,dy,promo)
                eval = self.minimax(next_state,depth-1,alpha,beta)
                if(eval<best_eval):
                    best_eval = eval
                    best_move = move
                beta = min(beta,eval) 
        return best_move
    
    def reset(self,):
        self.trans_tbl.clear()