import copy
import random
import time
from collections import deque

# --- Configuration ---
GAMES_TO_PLAY = 10
SEARCH_DEPTH = 2
BOARD_SIZE = 9

class QuoridorSim:
    def __init__(self):
        self.players = [
            {'id': 0, 'x': 4, 'y': 8, 'walls': 10, 'goal_y': 0, 'name': "P1 (Distance Only)"},
            {'id': 1, 'x': 4, 'y': 0, 'walls': 10, 'goal_y': 8, 'name': "P2 (Wall Conserv)"}
        ]
        self.walls = []  # List of {'x', 'y', 'type': 'h'/'v'}
        self.turn = 0
        self.winner = None
        self.moves_history = []

    def is_valid_coord(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def is_path_blocked(self, x1, y1, x2, y2, walls):
        if y1 == y2:  # Horizontal
            gap_x = min(x1, x2)
            gap_y = y1
            for w in walls:
                if w['type'] == 'v' and w['x'] == gap_x and (w['y'] == gap_y or w['y'] == gap_y - 1):
                    return True
        elif x1 == x2:  # Vertical
            gap_x = x1
            gap_y = min(y1, y2)
            for w in walls:
                if w['type'] == 'h' and w['y'] == gap_y and (w['x'] == gap_x or w['x'] == gap_x - 1):
                    return True
        return False

    def bfs_distance(self, start_x, start_y, target_y, current_walls):
        queue = deque([(start_x, start_y, 0)])
        visited = set([(start_x, start_y)])
        
        while queue:
            cx, cy, dist = queue.popleft()
            if cy == target_y:
                return dist
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if self.is_valid_coord(nx, ny) and (nx, ny) not in visited:
                    if not self.is_path_blocked(cx, cy, nx, ny, current_walls):
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        return float('inf')

    def is_valid_wall(self, x, y, orientation, current_walls):
        if not (0 <= x < BOARD_SIZE - 1 and 0 <= y < BOARD_SIZE - 1): return False
        for w in current_walls:
            if w['x'] == x and w['y'] == y: return False
            if orientation == 'h':
                if w['type'] == 'h' and w['y'] == y and abs(w['x'] - x) <= 1: return False
            else:
                if w['type'] == 'v' and w['x'] == x and abs(w['y'] - y) <= 1: return False
                
        # Connectivity check
        temp_walls = current_walls + [{'x': x, 'y': y, 'type': orientation}]
        if self.bfs_distance(self.players[0]['x'], self.players[0]['y'], self.players[0]['goal_y'], temp_walls) == float('inf'): return False
        if self.bfs_distance(self.players[1]['x'], self.players[1]['y'], self.players[1]['goal_y'], temp_walls) == float('inf'): return False
        return True

    def get_possible_moves(self, player_idx, players, walls):
        moves = []
        p = players[player_idx]
        opp = players[1 - player_idx]
        
        # 1. Pawn Moves
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = p['x'] + dx, p['y'] + dy
            if self.is_valid_coord(nx, ny):
                if not self.is_path_blocked(p['x'], p['y'], nx, ny, walls):
                    if nx == opp['x'] and ny == opp['y']:
                        # Jump
                        jump_x, jump_y = nx + dx, ny + dy
                        if self.is_valid_coord(jump_x, jump_y) and not self.is_path_blocked(nx, ny, jump_x, jump_y, walls):
                            moves.append({'type': 'move', 'x': jump_x, 'y': jump_y})
                        else:
                            diagonals = [(-1, 0), (1, 0)] if dx == 0 else [(0, -1), (0, 1)]
                            for ddx, ddy in diagonals:
                                diag_x, diag_y = nx + ddx, ny + ddy
                                if self.is_valid_coord(diag_x, diag_y) and not self.is_path_blocked(nx, ny, diag_x, diag_y, walls):
                                    moves.append({'type': 'move', 'x': diag_x, 'y': diag_y})
                    else:
                        moves.append({'type': 'move', 'x': nx, 'y': ny})

        # 2. Wall Moves (Heuristic Pruning: Only near players to save time)
        if players[player_idx]['walls'] > 0:
            # Optimization: Only check walls near both players
            relevant_spots = set()
            for pl in players:
                r = 2
                min_x, max_x = max(0, pl['x'] - r), min(BOARD_SIZE - 2, pl['x'] + r)
                min_y, max_y = max(0, pl['y'] - r), min(BOARD_SIZE - 2, pl['y'] + r)
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        relevant_spots.add((x, y))
            
            for (x, y) in relevant_spots:
                if self.is_valid_wall(x, y, 'h', walls): moves.append({'type': 'wall', 'x': x, 'y': y, 'orientation': 'h'})
                if self.is_valid_wall(x, y, 'v', walls): moves.append({'type': 'wall', 'x': x, 'y': y, 'orientation': 'v'})
                
        return moves

    # --- AI Evaluation Functions ---

    def evaluate_p1(self, players, walls):
        """P1 Heuristic: Distance Only"""
        p1_dist = self.bfs_distance(players[0]['x'], players[0]['y'], players[0]['goal_y'], walls)
        p2_dist = self.bfs_distance(players[1]['x'], players[1]['y'], players[1]['goal_y'], walls)
        # P1 wants to Minimize P1_Dist and Maximize P2_Dist
        return p2_dist - p1_dist

    def evaluate_p2(self, players, walls):
        """P2 Heuristic: Distance + Wall Conservation"""
        p1_dist = self.bfs_distance(players[0]['x'], players[0]['y'], players[0]['goal_y'], walls)
        p2_dist = self.bfs_distance(players[1]['x'], players[1]['y'], players[1]['goal_y'], walls)
        # P2 wants to Minimize P2_Dist and Maximize P1_Dist
        score = p1_dist - p2_dist
        # Bonus for keeping walls
        # score += (players[1]['walls'] * 0.5)
        return score

    # --- Minimax ---

    def run_minimax(self, player_idx):
        """
        Runs minimax for the specific player.
        Note: Each player runs their OWN search where THEY are the Maximizer.
        """
        best_score = -float('inf')
        best_move = None
        
        possible_moves = self.get_possible_moves(player_idx, self.players, self.walls)
        random.shuffle(possible_moves) # Add randomness for variety
        
        # Fallback if trapped (shouldn't happen in standard rules but good for safety)
        if not possible_moves: return None

        for move in possible_moves:
            # 1. Apply Move
            new_players = copy.deepcopy(self.players)
            new_walls = copy.deepcopy(self.walls)
            
            if move['type'] == 'move':
                new_players[player_idx]['x'] = move['x']
                new_players[player_idx]['y'] = move['y']
            else:
                new_walls.append({'x': move['x'], 'y': move['y'], 'type': move['orientation']})
                new_players[player_idx]['walls'] -= 1
            
            # Check Immediate Win
            if new_players[player_idx]['y'] == new_players[player_idx]['goal_y']:
                return move

            # 2. Call Recursive Search
            # We pass `player_idx` as the "root user" so the evaluation function knows whose perspective to use
            score = self.minimax_recursive(SEARCH_DEPTH - 1, new_players, new_walls, -float('inf'), float('inf'), False, player_idx)
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def minimax_recursive(self, depth, players, walls, alpha, beta, is_maximizing, root_player_idx):
        if depth == 0:
            if root_player_idx == 0: return self.evaluate_p1(players, walls)
            else: return self.evaluate_p2(players, walls)

        # Check Terminals
        if players[0]['y'] == players[0]['goal_y']: 
            return 1000 if root_player_idx == 0 else -1000
        if players[1]['y'] == players[1]['goal_y']:
            return 1000 if root_player_idx == 1 else -1000

        current_player = root_player_idx if is_maximizing else (1 - root_player_idx)
        moves = self.get_possible_moves(current_player, players, walls)
        
        if is_maximizing:
            max_eval = -float('inf')
            for move in moves:
                # Apply
                np = copy.deepcopy(players)
                nw = copy.deepcopy(walls)
                if move['type'] == 'move':
                    np[current_player]['x'] = move['x']
                    np[current_player]['y'] = move['y']
                else:
                    nw.append({'x': move['x'], 'y': move['y'], 'type': move['orientation']})
                    np[current_player]['walls'] -= 1
                
                eval_val = self.minimax_recursive(depth - 1, np, nw, alpha, beta, False, root_player_idx)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                # Apply
                np = copy.deepcopy(players)
                nw = copy.deepcopy(walls)
                if move['type'] == 'move':
                    np[current_player]['x'] = move['x']
                    np[current_player]['y'] = move['y']
                else:
                    nw.append({'x': move['x'], 'y': move['y'], 'type': move['orientation']})
                    np[current_player]['walls'] -= 1
                
                eval_val = self.minimax_recursive(depth - 1, np, nw, alpha, beta, True, root_player_idx)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha: break
            return min_eval

    def run_turn(self):
        move = self.run_minimax(self.turn)
        if not move:
            print(f"Game Error: Player {self.turn} has no moves!")
            self.winner = 1 - self.turn
            return

        # Execute real move
        if move['type'] == 'move':
            self.players[self.turn]['x'] = move['x']
            self.players[self.turn]['y'] = move['y']
        else:
            self.walls.append({'x': move['x'], 'y': move['y'], 'type': move['orientation']})
            self.players[self.turn]['walls'] -= 1
        
        self.moves_history.append(f"P{self.turn}:{move['type']}")

        # Check Win
        if self.players[self.turn]['y'] == self.players[self.turn]['goal_y']:
            self.winner = self.turn
        else:
            self.turn = 1 - self.turn

# --- Simulation Runner ---

def main():
    results = []
    print(f"Starting Simulation: {GAMES_TO_PLAY} Games")
    print(f"P1: Distance Only | P2: Distance + Wall Conservation")
    print("-" * 60)
    print(f"{'Game':<5} | {'Winner':<25} | {'Turns':<5} | {'P1 Walls':<8} | {'P2 Walls':<8}")
    print("-" * 60)

    data_log = "Game,Winner,Turns,P1_Walls_Left,P2_Walls_Left\n"

    start_time_total = time.time()

    for i in range(1, GAMES_TO_PLAY + 1):
        sim = QuoridorSim()
        turns = 0
        
        # Loop until game ends (limit to 200 turns to prevent infinite stalemates)
        while sim.winner is None and turns < 200:
            sim.run_turn()
            turns += 1
        
        winner_name = sim.players[sim.winner]['name'] if sim.winner is not None else "Draw"
        p1_walls = sim.players[0]['walls']
        p2_walls = sim.players[1]['walls']
        
        print(f"{i:<5} | {winner_name:<25} | {turns:<5} | {p1_walls:<8} | {p2_walls:<8}")
        
        data_log += f"{i},{winner_name},{turns},{p1_walls},{p2_walls}\n"
        results.append(sim.winner)

    end_time_total = time.time()
    duration = end_time_total - start_time_total

    # Summary
    p1_wins = results.count(0)
    p2_wins = results.count(1)
    draws = results.count(None)

    print("-" * 60)
    print("SIMULATION COMPLETE")
    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Player 1 Wins (Distance Only): {p1_wins}")
    print(f"Player 2 Wins (Wall Conserv):  {p2_wins}")
    print(f"Draws: {draws}")

    # Save to file
    with open("quoridor_results.txt", "w") as f:
        f.write(f"Simulation Summary ({GAMES_TO_PLAY} Games)\n")
        f.write(f"P1 Wins: {p1_wins}\n")
        f.write(f"P2 Wins: {p2_wins}\n")
        f.write(f"Draws: {draws}\n\n")
        f.write(data_log)
    
    print("Data saved to 'quoridor_results.txt'")

if __name__ == "__main__":
    main()