import copy
import random
import time
import math

# --- Configuration ---
GAMES_PER_MATCHUP = 50  # Games played per pair of agents
BOARD_ROWS = 6
BOARD_COLS = 7
EMPTY = 0
P1_PIECE = 1
P2_PIECE = 2
K_FACTOR = 32 # Elo K-factor

class Connect4Sim:
    def __init__(self, p1_config, p2_config):
        self.players = [p1_config, p2_config]
        self.board = [[EMPTY for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.turn = 0
        self.winner = None
        self.moves_history = []

    def is_valid_location(self, board, col):
        return board[0][col] == EMPTY

    def get_valid_locations(self, board):
        return [c for c in range(BOARD_COLS) if self.is_valid_location(board, c)]

    def get_next_open_row(self, board, col):
        for r in range(BOARD_ROWS - 1, -1, -1):
            if board[r][col] == EMPTY:
                return r
        return None

    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    def check_win(self, board, piece):
        # Check horizontal locations
        for c in range(BOARD_COLS - 3):
            for r in range(BOARD_ROWS):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True
        # Check vertical locations
        for c in range(BOARD_COLS):
            for r in range(BOARD_ROWS - 3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True
        # Check positively sloped diagonals
        for c in range(BOARD_COLS - 3):
            for r in range(BOARD_ROWS - 3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True
        # Check negatively sloped diagonals
        for c in range(BOARD_COLS - 3):
            for r in range(3, BOARD_ROWS):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        return False

    def is_terminal_node(self, board):
        return self.check_win(board, P1_PIECE) or self.check_win(board, P2_PIECE) or len(self.get_valid_locations(board)) == 0

    # --- AI Evaluation Functions ---

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = P1_PIECE if piece == P2_PIECE else P2_PIECE

        if window.count(piece) == 4:
            score += 1000
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 80 # Strongly penalize letting the opponent get 3 in a row

        return score

    def score_position(self, board, piece):
        score = 0
        
        # Center column preference (highly valuable in Connect 4)
        center_array = [board[r][BOARD_COLS//2] for r in range(BOARD_ROWS)]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Horizontal
        for r in range(BOARD_ROWS):
            row_array = board[r]
            for c in range(BOARD_COLS - 3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, piece)

        # Vertical
        for c in range(BOARD_COLS):
            col_array = [board[r][c] for r in range(BOARD_ROWS)]
            for r in range(BOARD_ROWS - 3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, piece)

        # Positive Diagonal
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        # Negative Diagonal
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score

    # --- Minimax ---

    def run_minimax(self, player_idx):
        agent = self.players[player_idx]
        depth = agent['depth']
        piece = P1_PIECE if player_idx == 0 else P2_PIECE
        
        valid_locations = self.get_valid_locations(self.board)
        
        # Fallback if board is full (should be caught by terminal checks)
        if not valid_locations: return None
        
        # If depth is 0 (Random Agent), just pick a random move
        if depth == 0:
            return random.choice(valid_locations)

        # Otherwise, run the search
        col, score = self.minimax_recursive(self.board, depth, -math.inf, math.inf, True, piece)
        
        # Fallback if all moves look equally bad
        if col is None:
            col = random.choice(valid_locations)
            
        return col

    def minimax_recursive(self, board, depth, alpha, beta, is_maximizing, piece):
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)
        
        opp_piece = P1_PIECE if piece == P2_PIECE else P2_PIECE

        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_win(board, piece):
                    return (None, 100000000000000)
                elif self.check_win(board, opp_piece):
                    return (None, -10000000000000)
                else: # Game is over, no more valid moves
                    return (None, 0)
            else: # Depth is zero
                return (None, self.score_position(board, piece))

        if is_maximizing:
            value = -math.inf
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = [r[:] for r in board] # Deep copy board
                self.drop_piece(b_copy, row, col, piece)
                new_score = self.minimax_recursive(b_copy, depth - 1, alpha, beta, False, piece)[1]
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_col, value

        else: # Minimizing player
            value = math.inf
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = [r[:] for r in board] # Deep copy board
                self.drop_piece(b_copy, row, col, opp_piece)
                new_score = self.minimax_recursive(b_copy, depth - 1, alpha, beta, True, piece)[1]
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_col, value

    def run_turn(self):
        col = self.run_minimax(self.turn)
        if col is None:
            self.winner = "Draw"
            return

        row = self.get_next_open_row(self.board, col)
        piece = P1_PIECE if self.turn == 0 else P2_PIECE
        self.drop_piece(self.board, row, col, piece)
        
        self.moves_history.append(f"P{self.turn+1} -> Col {col}")

        # Check Win
        if self.check_win(self.board, piece):
            self.winner = self.turn
        elif len(self.get_valid_locations(self.board)) == 0:
            self.winner = "Draw"
        else:
            self.turn = 1 - self.turn

# --- Elo Logic ---

def update_elo(rating1, rating2, result):
    """
    result: 1 if player 1 wins, 0 if player 2 wins, 0.5 for a draw
    """
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
    
    new_rating1 = rating1 + K_FACTOR * (result - expected1)
    new_rating2 = rating2 + K_FACTOR * ((1 - result) - expected2)
    
    return new_rating1, new_rating2

# --- Simulation Runner ---

def main():
    # Define our pool of agents. Depth 0 is random.
    agents = [
        {'id': 0, 'name': "Depth 1 (Novice)", 'depth': 1, 'elo': 1200},
        {'id': 1, 'name': "Depth 2 (Amateur)", 'depth': 2, 'elo': 1200},
        {'id': 2, 'name': "Depth 3 (Pro)", 'depth': 3, 'elo': 1200},
        {'id': 3, 'name': "Depth 4 (Master)", 'depth': 4, 'elo': 1200}
    ]

    print("Starting Connect 4 Minimax Elo Tournament")
    print("-" * 75)
    print(f"{'Matchup':<40} | {'Winner':<20} | {'Turns'}")
    print("-" * 75)

    start_time_total = time.time()

    # Round Robin: Every agent plays every other agent
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            p1 = agents[i]
            p2 = agents[j]
            
            # Play N games per matchup (alternating who goes first)
            for game_num in range(GAMES_PER_MATCHUP):
                # Alternate who is P1 (who drops first)
                if game_num % 2 == 0:
                    current_p1, current_p2 = p1, p2
                else:
                    current_p1, current_p2 = p2, p1

                sim = Connect4Sim(current_p1, current_p2)
                turns = 0
                
                while sim.winner is None:
                    sim.run_turn()
                    turns += 1
                
                # Determine Elo Result (1 for p1 win, 0 for p2 win, 0.5 for draw)
                if sim.winner == 0:
                    winner_name = current_p1['name']
                    p1_res, p2_res = 1, 0
                elif sim.winner == 1:
                    winner_name = current_p2['name']
                    p1_res, p2_res = 0, 1
                else:
                    winner_name = "Draw"
                    p1_res, p2_res = 0.5, 0.5

                # Update Elos in the master list
                new_elo1, new_elo2 = update_elo(current_p1['elo'], current_p2['elo'], p1_res)
                current_p1['elo'] = new_elo1
                current_p2['elo'] = new_elo2

                matchup_str = f"{current_p1['name']} vs {current_p2['name']}"
                print(f"{matchup_str:<40} | {winner_name:<20} | {turns}")

    end_time_total = time.time()
    
    # Sort agents by final Elo
    agents.sort(key=lambda x: x['elo'], reverse=True)

    print("\n" + "=" * 60)
    print("TOURNAMENT COMPLETE")
    print(f"Time Taken: {end_time_total - start_time_total:.2f} seconds")
    print("=" * 60)
    print(f"{'Rank':<5} | {'Agent Name':<25} | {'Search Depth':<15} | {'Final Elo':<10}")
    print("-" * 60)
    
    for idx, agent in enumerate(agents):
        print(f"{idx+1:<5} | {agent['name']:<25} | {agent['depth']:<15} | {int(agent['elo']):<10}")

if __name__ == "__main__":
    main()