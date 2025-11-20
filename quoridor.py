import tkinter as tk
from tkinter import messagebox
from collections import deque
import copy
import math
import time

# --- Constants ---
BOARD_SIZE = 9
CELL_SIZE = 50
GAP_SIZE = 10
WALL_WIDTH = GAP_SIZE
MARGIN = 20
PLAYER_COLORS = ["#2ecc71", "#e74c3c"]  # Green (P1), Red (AI)
WALL_COLOR = "#34495e"
PREVIEW_COLOR = "#95a5a6"
P1_GOAL_ROW = 0
P2_GOAL_ROW = 8
MAX_WALLS = 10

# --- Game Logic & AI ---

class QuoridorState:
    def __init__(self):
        # Player positions (row, col)
        # Player 0 starts at (8, 4) moving UP to row 0
        # Player 1 starts at (0, 4) moving DOWN to row 8
        self.players = [(8, 4), (0, 4)]
        self.walls_left = [MAX_WALLS, MAX_WALLS]
        # Walls: set of tuples (row, col, orientation)
        # Orientation: 'h' for horizontal, 'v' for vertical
        # Coordinates refer to the top-left intersection of the wall
        self.walls = set()
        self.turn = 0 # 0 for Human, 1 for AI

    def is_valid_move(self, player_idx, r, c):
        curr_r, curr_c = self.players[player_idx]
        
        # Basic adjacency check
        if abs(curr_r - r) + abs(curr_c - c) != 1:
            # Check for jump moves
            return self.is_valid_jump(player_idx, r, c)

        # Check if wall blocks direct move
        if not self.can_step(curr_r, curr_c, r, c):
            return False

        # Destination must be empty
        if (r, c) in self.players:
            return False
            
        return True

    def is_valid_jump(self, player_idx, r, c):
        curr_r, curr_c = self.players[player_idx]
        other_idx = 1 - player_idx
        other_r, other_c = self.players[other_idx]

        # Check valid jump: neighbor is opponent
        if abs(curr_r - other_r) + abs(curr_c - other_c) == 1:
            # Wall check between me and opponent
            if not self.can_step(curr_r, curr_c, other_r, other_c):
                return False
            
            # Straight jump
            if (2 * other_r - curr_r == r) and (2 * other_c - curr_c == c):
                # Wall check between opponent and landing
                return self.can_step(other_r, other_c, r, c)
            
            # Diagonal jump (only if straight jump blocked by wall or board edge)
            # This is complex, simplified here for standard rules:
            # If straight jump is blocked, can jump diagonally to sides of opponent
            if abs(r - other_r) + abs(c - other_c) == 1:
                # Only valid if the straight path behind opponent is blocked OR opponent is at edge
                jump_r = 2*other_r - curr_r
                jump_c = 2*other_c - curr_c
                
                straight_blocked = (not (0 <= jump_r < BOARD_SIZE and 0 <= jump_c < BOARD_SIZE)) or \
                                   (not self.can_step(other_r, other_c, jump_r, jump_c)) or \
                                   ((jump_r, jump_c) in self.walls) # simplistic check
                
                if straight_blocked:
                     return self.can_step(other_r, other_c, r, c)

        return False

    def can_step(self, r1, c1, r2, c2):
        """Check if a single step is blocked by a wall."""
        if r1 == r2: # Horizontal move
            c_min = min(c1, c2)
            # Blocked by vertical wall at (r1, c_min) or (r1-1, c_min)
            if (r1, c_min, 'v') in self.walls or (r1-1, c_min, 'v') in self.walls:
                return False
        else: # Vertical move
            r_min = min(r1, r2)
            # Blocked by horizontal wall at (r_min, c1) or (r_min, c1-1)
            if (r_min, c1, 'h') in self.walls or (r_min, c1-1, 'h') in self.walls:
                return False
        return True

    def is_valid_wall(self, r, c, orient):
        if self.walls_left[self.turn] <= 0:
            return False
        
        # Bounds check (walls are placed between 0..7)
        if not (0 <= r < BOARD_SIZE-1 and 0 <= c < BOARD_SIZE-1):
            return False
            
        # Intersection/Overlap check
        if (r, c, 'h') in self.walls or (r, c, 'v') in self.walls:
            return False
        
        if orient == 'h':
            if (r, c-1, 'h') in self.walls or (r, c+1, 'h') in self.walls: # Overlap half
                return False
        else:
            if (r-1, c, 'v') in self.walls or (r+1, c, 'v') in self.walls:
                return False

        # Path Existence Check (expensive but necessary)
        # Simulate adding wall
        self.walls.add((r, c, orient))
        p0_path = self.bfs(0)
        p1_path = self.bfs(1)
        self.walls.remove((r, c, orient)) # Backtrack

        return (p0_path is not None) and (p1_path is not None)

    def bfs(self, player_idx):
        """Returns shortest path distance to goal. None if no path."""
        start_node = self.players[player_idx]
        goal_row = P1_GOAL_ROW if player_idx == 0 else P2_GOAL_ROW
        
        q = deque([(start_node, 0)])
        visited = {start_node}
        
        while q:
            (curr_r, curr_c), dist = q.popleft()
            if curr_r == goal_row:
                return dist
            
            # Neighbors
            neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.can_step(curr_r, curr_c, nr, nc):
                         neighbors.append((nr, nc))
            
            for nr, nc in neighbors:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), dist + 1))
        return None

    def get_winner(self):
        if self.players[0][0] == P1_GOAL_ROW: return 0
        if self.players[1][0] == P2_GOAL_ROW: return 1
        return None

    def apply_move(self, action):
        """Apply move and return new state. Action: ('move', (r,c)) or ('wall', (r,c,o))"""
        new_state = copy.deepcopy(self)
        idx = new_state.turn
        
        if action[0] == 'move':
            new_state.players[idx] = action[1]
        elif action[0] == 'wall':
            r, c, o = action[1]
            new_state.walls.add((r, c, o))
            new_state.walls_left[idx] -= 1
            
        new_state.turn = 1 - idx
        return new_state

# --- Minimax AI ---

class QuoridorAI:
    def __init__(self, depth=2):
        self.depth = depth

    def evaluate(self, state):
        # Heuristic: My distance to goal vs Opponent distance to goal
        # AI is player 1 (Goal Row 8)
        ai_dist = state.bfs(1)
        human_dist = state.bfs(0)
        
        if ai_dist is None: return -9999 # Should not happen due to valid check
        if human_dist is None: return 9999
        
        # Lower distance is better for that player.
        # AI wants to minimize ai_dist and maximize human_dist.
        # Score = human_dist - ai_dist
        score = (human_dist - ai_dist)
        
        # Bonus for having more walls left (minor strategy)
        score += (state.walls_left[1] - state.walls_left[0]) * 0.1
        
        return score

    def get_best_move(self, state):
        _, move = self.minimax(state, self.depth, -math.inf, math.inf, True)
        return move

    def get_possible_actions(self, state, player_idx):
        actions = []
        
        # 1. Move actions
        curr_r, curr_c = state.players[player_idx]
        # Check all adjacent cells and jump targets (simplified)
        # Scan 3x3 area around player for valid moves
        possible_moves = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0: continue
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if state.is_valid_move(player_idx, nr, nc):
                        possible_moves.append(('move', (nr, nc)))
        
        # Sort moves by forward progress (heuristic for pruning)
        goal_row = P1_GOAL_ROW if player_idx == 0 else P2_GOAL_ROW
        possible_moves.sort(key=lambda m: abs(m[1][0] - goal_row))
        actions.extend(possible_moves)

        # 2. Wall actions
        # Optimization: Only consider walls near the opponent to block them
        # or near self to defend. Checking ALL 128 walls is too slow for Python.
        if state.walls_left[player_idx] > 0:
            opp_idx = 1 - player_idx
            opp_r, opp_c = state.players[opp_idx]
            
            # Try placing walls around opponent
            wall_candidates = []
            for r in range(max(0, opp_r-1), min(BOARD_SIZE-1, opp_r+2)):
                for c in range(max(0, opp_c-1), min(BOARD_SIZE-1, opp_c+2)):
                    wall_candidates.append((r, c, 'h'))
                    wall_candidates.append((r, c, 'v'))
            
            valid_walls = []
            for r, c, o in wall_candidates:
                 if state.is_valid_wall(r, c, o):
                     valid_walls.append(('wall', (r, c, o)))
            
            actions.extend(valid_walls)
            
        return actions

    def minimax(self, state, depth, alpha, beta, maximizing):
        winner = state.get_winner()
        if winner == 1: return 1000 + depth, None
        if winner == 0: return -1000 - depth, None
        
        if depth == 0:
            return self.evaluate(state), None

        actions = self.get_possible_actions(state, state.turn)
        best_move = None

        if maximizing:
            max_eval = -math.inf
            for action in actions:
                new_state = state.apply_move(action)
                eval_val, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = action
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for action in actions:
                new_state = state.apply_move(action)
                eval_val, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = action
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, best_move

# --- GUI ---

class QuoridorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quoridor vs Minimax AI")
        
        self.state = QuoridorState()
        self.ai = QuoridorAI(depth=2) # Depth 2 is responsive for Python
        
        self.canvas_width = BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * GAP_SIZE + 2 * MARGIN
        self.canvas_height = self.canvas_width
        
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="#ecf0f1")
        self.canvas.pack()
        
        self.status_var = tk.StringVar()
        self.status_var.set("Your Turn (Green). Left Click to Move, Right Click for V-Wall, Shift+Right for H-Wall")
        self.lbl_status = tk.Label(root, textvariable=self.status_var, font=("Arial", 10, "bold"))
        self.lbl_status.pack(pady=5)
        
        # Bindings
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click) # Windows/Linux
        self.canvas.bind("<Button-2>", self.on_shift_right_click) # MacOS often uses Button-2 for Right click or middle
        self.canvas.bind("<Shift-Button-3>", self.on_shift_right_click)
        
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        
        # Draw Cells
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = MARGIN + c * (CELL_SIZE + GAP_SIZE)
                y = MARGIN + r * (CELL_SIZE + GAP_SIZE)
                
                color = "#bdc3c7"
                # Highlight Goals
                if r == 0: color = "#d5f5e3" # Light green goal
                if r == 8: color = "#fadbd8" # Light red goal
                
                self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill=color, outline="")

        # Draw Walls
        for (r, c, o) in self.state.walls:
            self.draw_wall_graphic(r, c, o, WALL_COLOR)

        # Draw Players
        for i, (pr, pc) in enumerate(self.state.players):
            px = MARGIN + pc * (CELL_SIZE + GAP_SIZE) + CELL_SIZE/2
            py = MARGIN + pr * (CELL_SIZE + GAP_SIZE) + CELL_SIZE/2
            self.canvas.create_oval(px-15, py-15, px+15, py+15, fill=PLAYER_COLORS[i])

    def draw_wall_graphic(self, r, c, orient, color):
        # Convert grid coord to pixel coord (top-left of the intersection gap)
        gap_x = MARGIN + c * (CELL_SIZE + GAP_SIZE) + CELL_SIZE
        gap_y = MARGIN + r * (CELL_SIZE + GAP_SIZE) + CELL_SIZE
        
        length = 2 * CELL_SIZE + GAP_SIZE
        
        if orient == 'v':
            # Vertical wall: In the gap between col c and c+1, spanning row r and r+1
            x0 = gap_x
            y0 = gap_y - CELL_SIZE 
            x1 = x0 + GAP_SIZE
            y1 = y0 + length
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
        else:
            # Horizontal wall: In gap between row r and r+1, spanning col c and c+1
            x0 = gap_x - CELL_SIZE
            y0 = gap_y
            x1 = x0 + length
            y1 = y0 + GAP_SIZE
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

    def get_cell_from_pixels(self, x, y):
        # Simplified detection
        col = int((x - MARGIN) / (CELL_SIZE + GAP_SIZE))
        row = int((y - MARGIN) / (CELL_SIZE + GAP_SIZE))
        if 0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE:
            return row, col
        return None, None

    def get_gap_from_pixels(self, x, y):
        # Detect nearest intersection for wall placement
        # We map the click to the top-left 'gap' index
        c = int((x - MARGIN - CELL_SIZE/2) / (CELL_SIZE + GAP_SIZE))
        r = int((y - MARGIN - CELL_SIZE/2) / (CELL_SIZE + GAP_SIZE))
        return r, c

    def on_left_click(self, event):
        if self.state.turn != 0: return
        
        r, c = self.get_cell_from_pixels(event.x, event.y)
        if r is not None:
            if self.state.is_valid_move(0, r, c):
                self.human_move(('move', (r, c)))
            else:
                self.status_var.set("Invalid Move!")

    def on_right_click(self, event):
        if self.state.turn != 0: return
        r, c = self.get_gap_from_pixels(event.x, event.y)
        action = ('wall', (r, c, 'v'))
        if self.state.is_valid_wall(r, c, 'v'):
            self.human_move(action)
        else:
            self.status_var.set("Invalid Vertical Wall!")

    def on_shift_right_click(self, event):
        if self.state.turn != 0: return
        r, c = self.get_gap_from_pixels(event.x, event.y)
        action = ('wall', (r, c, 'h'))
        if self.state.is_valid_wall(r, c, 'h'):
            self.human_move(action)
        else:
            self.status_var.set("Invalid Horizontal Wall!")

    def human_move(self, action):
        self.state = self.state.apply_move(action)
        self.draw_board()
        self.check_game_over()
        if self.state.get_winner() is None:
            self.status_var.set("AI is thinking...")
            self.root.after(100, self.ai_move)

    def ai_move(self):
        start = time.time()
        action = self.ai.get_best_move(self.state)
        print(f"AI move calculated in {time.time() - start:.2f}s: {action}")
        
        if action:
            self.state = self.state.apply_move(action)
            self.draw_board()
            self.check_game_over()
            if self.state.get_winner() is None:
                self.status_var.set(f"Your Turn. Walls Left: You={self.state.walls_left[0]}, AI={self.state.walls_left[1]}")
        else:
            self.status_var.set("AI has no moves? (Bug or Stuck)")

    def check_game_over(self):
        winner = self.state.get_winner()
        if winner is not None:
            text = "You Won!" if winner == 0 else "AI Won!"
            messagebox.showinfo("Game Over", text)
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuoridorGUI(root)
    root.mainloop()