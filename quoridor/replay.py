import pygame
import sys
import csv
import time

# --- Constants ---
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 800
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
COLOR_MOVE_HIGHLIGHT = (255, 255, 0) # Yellow highlight for last move
COLOR_TEXT = (255, 255, 255)

class ReplayViewer:
    def __init__(self, filename="quoridor_trajectories.csv"):
        self.games = {} # { game_id: [rows] }
        self.load_data(filename)
        
        self.game_ids = sorted(list(self.games.keys()))
        if not self.game_ids:
            print(f"Error: No data found in {filename}")
            sys.exit()

        self.current_game_idx = 0
        self.current_turn = 0
        self.playing = False
        self.last_step_time = 0
        self.playback_speed = 0.5 # Seconds per turn

    def load_data(self, filename):
        try:
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gid = int(row['GameID'])
                    if gid not in self.games:
                        self.games[gid] = []
                    
                    # Parse Wall String "H23;V44;" -> [{'x':2, 'y':3, 'o':'H'}, ...]
                    walls = []
                    raw_walls = row['Board_Walls'].strip().split(';')
                    for w in raw_walls:
                        if len(w) >= 3:
                            walls.append({
                                'type': w[0], # 'H' or 'V'
                                'x': int(w[1]),
                                'y': int(w[2:])
                            })
                    row['parsed_walls'] = walls
                    self.games[gid].append(row)
            print(f"Loaded {len(self.games)} games.")
        except FileNotFoundError:
            print(f"File {filename} not found! Run the C++ sim first.")
            sys.exit()

    def get_current_state(self):
        gid = self.game_ids[self.current_game_idx]
        turns = self.games[gid]
        
        # Clamp turn
        if self.current_turn >= len(turns):
            self.current_turn = len(turns) - 1
            self.playing = False # Stop at end
        
        return turns[self.current_turn]

    def draw(self, screen, font):
        state = self.get_current_state()
        screen.fill(COLOR_BG)

        # --- Draw Stats Panel ---
        gid = self.game_ids[self.current_game_idx]
        total_turns = len(self.games[gid])
        
        winner_val = int(state['Winner'])
        winner_text = "Draw"
        if winner_val == 0: winner_text = "P1 (Blue) Won"
        elif winner_val == 1: winner_text = "P2 (Red) Won"
        else: winner_text = "In Progress" if self.current_turn < total_turns - 1 else "Draw"

        # Text Lines
        lines = [
            f"Game: {gid} / {len(self.games)}",
            f"Turn: {state['Turn']} / {total_turns - 1}",
            f"Winner: {winner_text}",
            f"Active: {'P1 (Blue)' if state['ActivePlayer'] == '0' else 'P2 (Red)'}",
            f"P1 Walls: {state['P1_Walls']} | P2 Walls: {state['P2_Walls']}",
            "",
            "Controls:",
            "Left/Right: Prev/Next Turn",
            "Up/Down: Prev/Next Game",
            "Space: Auto-Play"
        ]

        # Draw Text on Right Side
        text_x = BOARD_SIZE * (CELL_SIZE + GAP_SIZE) + MARGIN * 2
        y_offset = MARGIN
        for line in lines:
            txt_surf = font.render(line, True, COLOR_TEXT)
            screen.blit(txt_surf, (text_x, y_offset))
            y_offset += 30
        
        # --- Draw Board ---
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                px = MARGIN + x * (CELL_SIZE + GAP_SIZE)
                py = MARGIN + y * (CELL_SIZE + GAP_SIZE)
                rect = pygame.Rect(px, py, CELL_SIZE, CELL_SIZE)
                
                col = COLOR_CELL
                if y == 0 or y == 8: col = COLOR_CELL_GOAL
                
                # Highlight Move Target (Ghost)
                mv_type = state['Move_Type']
                mv_x = int(state['Move_X'])
                mv_y = int(state['Move_Y'])
                
                # If this turn involves moving a pawn here
                if "PAWN" in mv_type and mv_x == x and mv_y == y:
                    pygame.draw.rect(screen, (100, 100, 0), rect, border_radius=4) # Dark Yellow underlay
                    
                pygame.draw.rect(screen, col, rect, border_radius=4)

        # --- Draw Walls ---
        # Existing Walls
        for w in state['parsed_walls']:
            self.draw_wall_rect(screen, w['x'], w['y'], w['type'], COLOR_WALL)

        # Highlight Move Wall (The one added THIS turn)
        mv_type = state['Move_Type']
        if "WALL" in mv_type:
            mv_x = int(state['Move_X'])
            mv_y = int(state['Move_Y'])
            mv_ori = state['Move_Ori']
            # Draw bright highlight wall
            self.draw_wall_rect(screen, mv_x, mv_y, mv_ori, COLOR_MOVE_HIGHLIGHT)

        # --- Draw Players ---
        self.draw_player(screen, int(state['P1_X']), int(state['P1_Y']), COLOR_P1)
        self.draw_player(screen, int(state['P2_X']), int(state['P2_Y']), COLOR_P2)

    def draw_wall_rect(self, screen, x, y, ori, color):
        px = MARGIN + x * (CELL_SIZE + GAP_SIZE)
        py = MARGIN + y * (CELL_SIZE + GAP_SIZE)
        if ori == 'H':
            w_rect = pygame.Rect(px, py + CELL_SIZE, CELL_SIZE*2 + GAP_SIZE, GAP_SIZE)
        else:
            w_rect = pygame.Rect(px + CELL_SIZE, py, GAP_SIZE, CELL_SIZE*2 + GAP_SIZE)
        pygame.draw.rect(screen, color, w_rect, border_radius=2)

    def draw_player(self, screen, x, y, color):
        px = MARGIN + x * (CELL_SIZE + GAP_SIZE) + CELL_SIZE // 2
        py = MARGIN + y * (CELL_SIZE + GAP_SIZE) + CELL_SIZE // 2
        pygame.draw.circle(screen, color, (px, py), CELL_SIZE // 2 - 8)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), CELL_SIZE // 2 - 8, 2)

    def update(self):
        if self.playing:
            now = time.time()
            if now - self.last_step_time > self.playback_speed:
                self.current_turn += 1
                gid = self.game_ids[self.current_game_idx]
                if self.current_turn >= len(self.games[gid]):
                    self.current_turn = len(self.games[gid]) - 1
                    self.playing = False
                self.last_step_time = now

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH + 200, SCREEN_HEIGHT)) # Extra width for text
    pygame.display.set_caption("Quoridor Replay Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 20)

    viewer = ReplayViewer()

    running = True
    while running:
        viewer.update()
        viewer.draw(screen, font)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Turn Navigation
                if event.key == pygame.K_RIGHT:
                    viewer.current_turn += 1
                    viewer.playing = False
                elif event.key == pygame.K_LEFT:
                    viewer.current_turn = max(0, viewer.current_turn - 1)
                    viewer.playing = False
                
                # Game Navigation
                elif event.key == pygame.K_DOWN:
                    viewer.current_game_idx = (viewer.current_game_idx + 1) % len(viewer.game_ids)
                    viewer.current_turn = 0
                    viewer.playing = False
                elif event.key == pygame.K_UP:
                    viewer.current_game_idx = (viewer.current_game_idx - 1) % len(viewer.game_ids)
                    viewer.current_turn = 0
                    viewer.playing = False
                
                # Playback
                elif event.key == pygame.K_SPACE:
                    viewer.playing = not viewer.playing

        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()