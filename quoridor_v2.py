import pygame
import sys
import copy
import random
from collections import deque

# --- Constants ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 750  # Extra height for UI
BOARD_SIZE = 9
CELL_SIZE = 50
GAP_SIZE = 12
MARGIN = 20

# Colors
COLOR_BG = (31, 41, 55)        # Dark Grey
COLOR_BOARD = (229, 231, 235)  # Light Grey
COLOR_CELL = (139, 69, 19)     # Wood/Brown
COLOR_CELL_GOAL = (150, 80, 30)# Slightly darker for goal rows
COLOR_P1 = (59, 130, 246)      # Blue
COLOR_P2 = (239, 68, 68)       # Red
COLOR_WALL = (75, 85, 99)      # Dark Grey Wall
COLOR_WALL_HOVER = (59, 130, 246) # Blueish for hover
COLOR_WALL_INVALID = (239, 68, 68) # Red for invalid
COLOR_HINT = (16, 185, 129)    # Green
COLOR_TEXT = (255, 255, 255)
COLOR_BTN = (55, 65, 81)
COLOR_BTN_HOVER = (75, 85, 99)
COLOR_BTN_ACTIVE = (59, 130, 246)

# --- Game Logic Class ---

class QuoridorGame:
    def __init__(self):
        self.players = [
            {'x': 4, 'y': 8, 'walls': 10, 'goal_y': 0},  # P1 (Human)
            {'x': 4, 'y': 0, 'walls': 10, 'goal_y': 8}   # P2 (AI)
        ]
        self.walls = []  # List of {'x', 'y', 'type': 'h'/'v'}
        self.turn = 0    # 0 for Human, 1 for AI
        self.winner = None
        
        # UI State
        self.wall_mode = False
        self.wall_orientation = 'h'
        self.ai_stats = {'heuristic': 0.0, 'minimax': 0.0}
        self.message = "Your Turn"

    def is_valid_coord(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def is_path_blocked(self, x1, y1, x2, y2, walls):
        """Check if a wall exists between two adjacent cells"""
        if y1 == y2:  # Horizontal move
            gap_x = min(x1, x2)
            gap_y = y1
            # Blocked by vertical wall at (gap_x, gap_y) or (gap_x, gap_y-1)
            for w in walls:
                if w['type'] == 'v' and w['x'] == gap_x and (w['y'] == gap_y or w['y'] == gap_y - 1):
                    return True
        elif x1 == x2:  # Vertical move
            gap_x = x1
            gap_y = min(y1, y2)
            # Blocked by horizontal wall at (gap_x, gap_y) or (gap_x-1, gap_y)
            for w in walls:
                if w['type'] == 'h' and w['y'] == gap_y and (w['x'] == gap_x or w['x'] == gap_x - 1):
                    return True
        return False

    def get_valid_moves(self, player_idx, current_walls=None):
        if current_walls is None: current_walls = self.walls
        
        p = self.players[player_idx]
        opp = self.players[1 - player_idx]
        moves = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
            nx, ny = p['x'] + dx, p['y'] + dy
            
            if self.is_valid_coord(nx, ny):
                if not self.is_path_blocked(p['x'], p['y'], nx, ny, current_walls):
                    # Check if opponent is there (Jump logic)
                    if nx == opp['x'] and ny == opp['y']:
                        jump_x, jump_y = nx + dx, ny + dy
                        # Straight Jump
                        if (self.is_valid_coord(jump_x, jump_y) and 
                            not self.is_path_blocked(nx, ny, jump_x, jump_y, current_walls)):
                            moves.append((jump_x, jump_y))
                        else:
                            # Diagonal Jump
                            diagonals = []
                            if dx == 0: # Was vertical
                                diagonals = [(-1, 0), (1, 0)]
                            else: # Was horizontal
                                diagonals = [(0, -1), (0, 1)]
                            
                            for ddx, ddy in diagonals:
                                diag_x, diag_y = nx + ddx, ny + ddy
                                if (self.is_valid_coord(diag_x, diag_y) and 
                                    not self.is_path_blocked(nx, ny, diag_x, diag_y, current_walls)):
                                    moves.append((diag_x, diag_y))
                    else:
                        moves.append((nx, ny))
        return moves

    def bfs_distance(self, start_x, start_y, target_y, current_walls):
        queue = deque([(start_x, start_y, 0)])
        visited = set([(start_x, start_y)])
        
        while queue:
            cx, cy, dist = queue.popleft()
            if cy == target_y:
                return dist
            
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if self.is_valid_coord(nx, ny) and (nx, ny) not in visited:
                    if not self.is_path_blocked(cx, cy, nx, ny, current_walls):
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        return float('inf')

    def is_valid_wall(self, x, y, orientation, current_walls):
        # 1. Bounds
        if not (0 <= x < BOARD_SIZE - 1 and 0 <= y < BOARD_SIZE - 1):
            return False
            
        # 2. Overlap/Intersection
        for w in current_walls:
            if w['x'] == x and w['y'] == y: return False # Exact same pos
            if orientation == 'h':
                # Overlap horizontal
                if w['type'] == 'h' and w['y'] == y and abs(w['x'] - x) <= 0: return False # Same
                if w['type'] == 'h' and w['y'] == y and abs(w['x'] - x) == 1: return False # Overlap
            else: # Vertical
                if w['type'] == 'v' and w['x'] == x and abs(w['y'] - y) == 1: return False # Overlap
            
            # Cross intersection is technically allowed in some rules, but standard quoridor often forbids direct crossing center
            # For simplicity, we prevent exact center crossing if it creates a visual mess, but logic allows cross if not blocking
            if w['x'] == x and w['y'] == y: return False 

        # 3. Path Connectivity Check (Expensive)
        # Temporarily add wall
        temp_walls = current_walls + [{'x': x, 'y': y, 'type': orientation}]
        
        p1_dist = self.bfs_distance(self.players[0]['x'], self.players[0]['y'], self.players[0]['goal_y'], temp_walls)
        p2_dist = self.bfs_distance(self.players[1]['x'], self.players[1]['y'], self.players[1]['goal_y'], temp_walls)
        
        if p1_dist == float('inf') or p2_dist == float('inf'):
            return False
            
        return True

    # --- AI Logic ---

    def evaluate_state(self, players, walls):
        p1_dist = self.bfs_distance(players[0]['x'], players[0]['y'], players[0]['goal_y'], walls)
        p2_dist = self.bfs_distance(players[1]['x'], players[1]['y'], players[1]['goal_y'], walls)
        
        # AI (P2) wants to Minimize p2_dist and Maximize p1_dist
        score = p1_dist - p2_dist
        
        # Wall Conservation Heuristic
        score += (players[1]['walls'] * 0.5)
        
        return score

    def get_possible_moves(self, player_idx, players, walls):
        moves = []
        
        # 1. Pawn Moves
        # Use a temporary game instance or helper to get logic without object overhead
        # We'll just reuse the logic methods passing specific state
        # NOTE: For performance, we duplicate logic slightly or pass state
        
        # Re-implement get_valid_moves for passed state
        p = players[player_idx]
        opp = players[1 - player_idx]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for dx, dy in directions:
            nx, ny = p['x'] + dx, p['y'] + dy
            if self.is_valid_coord(nx, ny):
                if not self.is_path_blocked(p['x'], p['y'], nx, ny, walls):
                    if nx == opp['x'] and ny == opp['y']:
                        jump_x, jump_y = nx + dx, ny + dy
                        if (self.is_valid_coord(jump_x, jump_y) and 
                            not self.is_path_blocked(nx, ny, jump_x, jump_y, walls)):
                            moves.append({'type': 'move', 'x': jump_x, 'y': jump_y})
                        else:
                            # Diagonals
                            diags = [(-1, 0), (1, 0)] if dx == 0 else [(0, -1), (0, 1)]
                            for ddx, ddy in diags:
                                diag_x, diag_y = nx + ddx, ny + ddy
                                if (self.is_valid_coord(diag_x, diag_y) and 
                                    not self.is_path_blocked(nx, ny, diag_x, diag_y, walls)):
                                    moves.append({'type': 'move', 'x': diag_x, 'y': diag_y})
                    else:
                        moves.append({'type': 'move', 'x': nx, 'y': ny})

        # 2. Wall Moves (Optimized)
        if players[player_idx]['walls'] > 0:
            focus_player = players[1 - player_idx] # Try to block opponent
            r = 2 # Range
            min_x = max(0, focus_player['x'] - r)
            max_x = min(BOARD_SIZE - 2, focus_player['x'] + r)
            min_y = max(0, focus_player['y'] - r)
            max_y = min(BOARD_SIZE - 2, focus_player['y'] + r)

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if self.is_valid_wall(x, y, 'h', walls):
                        moves.append({'type': 'wall', 'x': x, 'y': y, 'orientation': 'h'})
                    if self.is_valid_wall(x, y, 'v', walls):
                        moves.append({'type': 'wall', 'x': x, 'y': y, 'orientation': 'v'})
                        
        return moves

    def minimax(self, depth, players, walls, alpha, beta, maximizing):
        if depth == 0:
            return self.evaluate_state(players, walls)
        
        # Terminals
        if players[0]['y'] == players[0]['goal_y']: return -1000 # P1 Wins
        if players[1]['y'] == players[1]['goal_y']: return 1000  # AI Wins

        player_idx = 1 if maximizing else 0
        possible_moves = self.get_possible_moves(player_idx, players, walls)
        random.shuffle(possible_moves) # Variety

        if maximizing:
            max_eval = -float('inf')
            for move in possible_moves:
                # Clone State
                new_players = copy.deepcopy(players)
                new_walls = copy.deepcopy(walls)
                
                if move['type'] == 'move':
                    new_players[1]['x'] = move['x']
                    new_players[1]['y'] = move['y']
                else:
                    new_walls.append({'x': move['x'], 'y': move['y'], 'type': move['orientation']})
                    new_players[1]['walls'] -= 1
                
                eval = self.minimax(depth - 1, new_players, new_walls, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                new_players = copy.deepcopy(players)
                new_walls = copy.deepcopy(walls)
                
                if move['type'] == 'move':
                    new_players[0]['x'] = move['x']
                    new_players[0]['y'] = move['y']
                else:
                    new_walls.append({'x': move['x'], 'y': move['y'], 'type': move['orientation']})
                    new_players[0]['walls'] -= 1
                
                eval = self.minimax(depth - 1, new_players, new_walls, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

    def make_ai_move(self):
        # Static Heuristic
        base_score = self.evaluate_state(self.players, self.walls)
        self.ai_stats['heuristic'] = base_score

        # Minimax
        best_score = -float('inf')
        best_move = None
        
        possible_moves = self.get_possible_moves(1, self.players, self.walls)
        random.shuffle(possible_moves)

        # Depth 2
        for move in possible_moves:
            new_players = copy.deepcopy(self.players)
            new_walls = copy.deepcopy(self.walls)
            
            if move['type'] == 'move':
                new_players[1]['x'] = move['x']
                new_players[1]['y'] = move['y']
            else:
                new_walls.append({'x': move['x'], 'y': move['y'], 'type': move['orientation']})
                new_players[1]['walls'] -= 1
            
            # Check immediate win
            if new_players[1]['y'] == new_players[1]['goal_y']:
                best_move = move
                best_score = 1000
                break

            score = self.minimax(1, new_players, new_walls, -float('inf'), float('inf'), False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        self.ai_stats['minimax'] = best_score
        
        if best_move:
            if best_move['type'] == 'move':
                self.players[1]['x'] = best_move['x']
                self.players[1]['y'] = best_move['y']
            else:
                self.walls.append({'x': best_move['x'], 'y': best_move['y'], 'type': best_move['orientation']})
                self.players[1]['walls'] -= 1
        
        # Check win
        if self.players[1]['y'] == self.players[1]['goal_y']:
            self.winner = 1
            self.message = "AI WINS!"
        else:
            self.turn = 0
            self.message = "Your Turn"

# --- Render & Input ---

def draw_game(screen, game, font, button_rect):
    screen.fill(COLOR_BG)
    
    # Draw Board Background
    board_rect = pygame.Rect(MARGIN, MARGIN, BOARD_SIZE * CELL_SIZE + (BOARD_SIZE-1)*GAP_SIZE, BOARD_SIZE * CELL_SIZE + (BOARD_SIZE-1)*GAP_SIZE)
    # pygame.draw.rect(screen, COLOR_BOARD, board_rect)

    # Draw Cells
    mouse_pos = pygame.mouse.get_pos()
    mouse_x, mouse_y = mouse_pos
    
    valid_moves = []
    if game.turn == 0 and not game.winner:
        valid_moves = game.get_valid_moves(0)

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            px = MARGIN + x * (CELL_SIZE + GAP_SIZE)
            py = MARGIN + y * (CELL_SIZE + GAP_SIZE)
            rect = pygame.Rect(px, py, CELL_SIZE, CELL_SIZE)
            
            col = COLOR_CELL
            if y == 0 or y == 8: col = COLOR_CELL_GOAL
            
            # Hint Logic
            is_valid = (x, y) in valid_moves
            
            # Hover Logic (Movement Mode)
            if (not game.wall_mode and game.turn == 0 and not game.winner and 
                rect.collidepoint(mouse_pos) and is_valid):
                col = COLOR_HINT
            
            pygame.draw.rect(screen, col, rect, border_radius=4)
            
            # Draw Green Hint Dot
            if game.turn == 0 and not game.wall_mode and is_valid:
                center = (px + CELL_SIZE//2, py + CELL_SIZE//2)
                pygame.draw.circle(screen, COLOR_HINT, center, 6)

    # Draw Walls (Existing)
    for w in game.walls:
        px = MARGIN + w['x'] * (CELL_SIZE + GAP_SIZE)
        py = MARGIN + w['y'] * (CELL_SIZE + GAP_SIZE)
        if w['type'] == 'h':
            w_rect = pygame.Rect(px, py + CELL_SIZE, CELL_SIZE*2 + GAP_SIZE, GAP_SIZE)
        else:
            w_rect = pygame.Rect(px + CELL_SIZE, py, GAP_SIZE, CELL_SIZE*2 + GAP_SIZE)
        pygame.draw.rect(screen, COLOR_WALL, w_rect, border_radius=2)

    # Wall Preview (Ghost)
    if game.wall_mode and game.turn == 0 and not game.winner:
        # Convert mouse to grid
        mx = mouse_x - MARGIN
        my = mouse_y - MARGIN
        stride = CELL_SIZE + GAP_SIZE
        
        gx = int(round(mx / stride - 0.5))
        gy = int(round(my / stride - 0.5))
        
        if 0 <= gx < BOARD_SIZE-1 and 0 <= gy < BOARD_SIZE-1:
            valid_wall = game.is_valid_wall(gx, gy, game.wall_orientation, game.walls)
            color = COLOR_WALL_HOVER if valid_wall else COLOR_WALL_INVALID
            
            px = MARGIN + gx * stride
            py = MARGIN + gy * stride
            
            if game.wall_orientation == 'h':
                preview_rect = pygame.Rect(px, py + CELL_SIZE, CELL_SIZE*2 + GAP_SIZE, GAP_SIZE)
            else:
                preview_rect = pygame.Rect(px + CELL_SIZE, py, GAP_SIZE, CELL_SIZE*2 + GAP_SIZE)
            
            s = pygame.Surface((preview_rect.width, preview_rect.height))
            s.set_alpha(180)
            s.fill(color)
            screen.blit(s, (preview_rect.x, preview_rect.y))

    # Draw Players
    for i, p in enumerate(game.players):
        px = MARGIN + p['x'] * (CELL_SIZE + GAP_SIZE) + CELL_SIZE // 2
        py = MARGIN + p['y'] * (CELL_SIZE + GAP_SIZE) + CELL_SIZE // 2
        color = COLOR_P1 if i == 0 else COLOR_P2
        pygame.draw.circle(screen, color, (px, py), CELL_SIZE // 2 - 8)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), CELL_SIZE // 2 - 8, 2)

    # --- UI Panel ---
    ui_y = board_rect.bottom + 20
    
    # Status Text
    status_surf = font.render(game.message, True, COLOR_TEXT)
    screen.blit(status_surf, (MARGIN, ui_y))
    
    # Wall Counts
    p1_walls_txt = font.render(f"You (Blue) Walls: {game.players[0]['walls']}", True, COLOR_P1)
    screen.blit(p1_walls_txt, (MARGIN, ui_y + 30))
    
    p2_walls_txt = font.render(f"AI (Red) Walls: {game.players[1]['walls']}", True, COLOR_P2)
    screen.blit(p2_walls_txt, (SCREEN_WIDTH - p2_walls_txt.get_width() - MARGIN, ui_y + 30))
    
    # AI Stats
    heur_txt = font.render(f"AI Heuristic: {game.ai_stats['heuristic']:.1f}", True, (156, 163, 175))
    mini_txt = font.render(f"AI Minimax: {game.ai_stats['minimax']:.1f}", True, (245, 158, 11))
    screen.blit(heur_txt, (SCREEN_WIDTH - heur_txt.get_width() - MARGIN, ui_y))
    screen.blit(mini_txt, (SCREEN_WIDTH - mini_txt.get_width() - MARGIN, ui_y - 20))
    
    # Wall Mode Button
    btn_color = COLOR_BTN_ACTIVE if game.wall_mode else COLOR_BTN
    if button_rect.collidepoint(mouse_pos):
        btn_color = COLOR_BTN_HOVER if not game.wall_mode else COLOR_BTN_ACTIVE
        
    pygame.draw.rect(screen, btn_color, button_rect, border_radius=8)
    btn_txt = font.render(f"Place Wall ({'ON' if game.wall_mode else 'OFF'})", True, COLOR_TEXT)
    txt_rect = btn_txt.get_rect(center=button_rect.center)
    screen.blit(btn_txt, txt_rect)
    
    # Instructions
    hint_font = pygame.font.SysFont(None, 20)
    hint_surf = hint_font.render("Space: Rotate Wall | Click: Move/Place", True, (156, 163, 175))
    screen.blit(hint_surf, (MARGIN, SCREEN_HEIGHT - 30))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Quoridor vs AI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 24)
    
    game = QuoridorGame()
    
    # UI Geometry
    button_width = 200
    button_height = 50
    btn_x = (SCREEN_WIDTH - button_width) // 2
    btn_y = SCREEN_HEIGHT - 100
    button_rect = pygame.Rect(btn_x, btn_y, button_width, button_height)
    
    running = True
    while running:
        draw_game(screen, game, font, button_rect)
        pygame.display.flip()
        
        # AI Turn
        if game.turn == 1 and not game.winner:
            # Force a draw update so "AI Thinking" or status shows if we implemented loading text
            pygame.event.pump() 
            game.make_ai_move()
            continue

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.winner: continue
                
                mx, my = pygame.mouse.get_pos()
                
                # Button Click
                if button_rect.collidepoint((mx, my)):
                    game.wall_mode = not game.wall_mode
                    continue
                
                # Board Interaction
                stride = CELL_SIZE + GAP_SIZE
                
                if game.wall_mode:
                    # Wall Placement
                    gx = int(round((mx - MARGIN) / stride - 0.5))
                    gy = int(round((my - MARGIN) / stride - 0.5))
                    
                    if game.players[0]['walls'] > 0 and game.is_valid_wall(gx, gy, game.wall_orientation, game.walls):
                        game.walls.append({'x': gx, 'y': gy, 'type': game.wall_orientation})
                        game.players[0]['walls'] -= 1
                        game.wall_mode = False # Toggle off after placement
                        game.turn = 1
                        game.message = "AI Thinking..."
                        
                else:
                    # Pawn Movement
                    cx = int((mx - MARGIN) / stride)
                    cy = int((my - MARGIN) / stride)
                    
                    # Precise click check inside cell
                    cell_left = MARGIN + cx * stride
                    cell_top = MARGIN + cy * stride
                    if (mx >= cell_left and mx <= cell_left + CELL_SIZE and
                        my >= cell_top and my <= cell_top + CELL_SIZE):
                        
                        valid_moves = game.get_valid_moves(0)
                        if (cx, cy) in valid_moves:
                            game.players[0]['x'] = cx
                            game.players[0]['y'] = cy
                            
                            if cy == game.players[0]['goal_y']:
                                game.winner = 0
                                game.message = "YOU WIN!"
                            else:
                                game.turn = 1
                                game.message = "AI Thinking..."

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.wall_orientation = 'v' if game.wall_orientation == 'h' else 'h'
                elif event.key == pygame.K_w:
                     game.wall_mode = not game.wall_mode
                elif event.key == pygame.K_r: # Reset
                     game = QuoridorGame()

        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()