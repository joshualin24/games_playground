import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import random
import os

# --- Constants ---
BOARD_SIZE = 9
ACTION_SPACE = 81 + 128 # 81 pawn moves (0-80) + 128 wall positions (81-208)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. The Neural Networks (Separated) ---

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        # Input: 5 Channels x 9 x 9
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Policy Head
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1) # 1x1 conv
        self.policy_fc = nn.Linear(4 * 9 * 9, ACTION_SPACE)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.policy_conv(x))
        x = x.view(-1, 4 * 9 * 9)
        x = self.policy_fc(x)
        return F.log_softmax(x, dim=1) # Log probabilities

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        # Input: 5 Channels x 9 x 9
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Value Head
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * 9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.value_conv(x))
        x = x.view(-1, 2 * 9 * 9)
        x = F.relu(self.value_fc1(x))
        return torch.tanh(self.value_fc2(x)) # Output between -1 and 1

# --- 2. Data Processing & Helpers ---

def encode_board(row):
    # Returns (5, 9, 9) numpy array
    state = np.zeros((5, 9, 9), dtype=np.float32)
    
    # Coordinates
    active = int(row['ActivePlayer'])
    p1_x, p1_y = int(row['P1_X']), int(row['P1_Y'])
    p2_x, p2_y = int(row['P2_X']), int(row['P2_Y'])
    
    # Normalized Walls left (0.0 to 1.0)
    p1_walls = int(row['P1_Walls']) / 10.0
    p2_walls = int(row['P2_Walls']) / 10.0

    # Perspective Transform: We always want the network to "think" it is P1
    # If Active is P2, we swap inputs so P2 looks like P1 to the network
    if active == 0:
        my_x, my_y = p1_x, p1_y
        op_x, op_y = p2_x, p2_y
        my_walls = p1_walls
    else:
        my_x, my_y = p2_x, p2_y
        op_x, op_y = p1_x, p1_y
        my_walls = p2_walls

    state[0, my_y, my_x] = 1.0
    state[1, op_y, op_x] = 1.0
    
    # Walls
    raw_walls = str(row['Board_Walls']).strip().split(';')
    for w in raw_walls:
        if len(w) < 3: continue
        ori = w[0]
        wx = int(w[1])
        wy = int(w[2:])
        if ori == 'H':
            state[2, wy, wx] = 1.0
        else:
            state[3, wy, wx] = 1.0
            
    # Plane 4: How many walls I have left (Global plane)
    state[4, :, :] = my_walls
    
    return state

def encode_action(row):
    # Map action to index 0-208
    m_type = row['Move_Type']
    mx = int(row['Move_X'])
    my = int(row['Move_Y'])
    
    if "PAWN" in m_type:
        # 0 - 80: y*9 + x
        return my * 9 + mx
    else:
        # Wall
        ori = row['Move_Ori']
        # Walls are on 8x8 grid of gaps.
        # H: 81 + y*8 + x (Index 81 - 144)
        # V: 145 + y*8 + x (Index 145 - 208)
        
        # Safety clamp
        mx = min(mx, 7) 
        my = min(my, 7)
        
        if ori == 'H':
            return 81 + my * 8 + mx
        else:
            return 145 + my * 8 + mx

def decode_action(idx):
    # Returns dict {'type', 'x', 'y', 'ori'}
    if idx < 81:
        return {'type': 'move', 'x': idx % 9, 'y': idx // 9}
    elif idx < 145:
        w_idx = idx - 81
        return {'type': 'wall', 'x': w_idx % 8, 'y': w_idx // 8, 'ori': 'H'}
    else:
        w_idx = idx - 145
        return {'type': 'wall', 'x': w_idx % 8, 'y': w_idx // 8, 'ori': 'V'}

class QuoridorDataset(Dataset):
    def __init__(self, csv_file):
        print("Loading Dataset...")
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['Winner'] != -1]
        print(f"Loaded {len(self.data)} moves.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        state = encode_board(row)
        action = encode_action(row)
        
        active = int(row['ActivePlayer'])
        winner = int(row['Winner'])
        value = 1.0 if active == winner else -1.0
        
        return torch.tensor(state), torch.tensor(action), torch.tensor(value, dtype=torch.float32)

# --- 3. Separate Training Functions ---

def train_policy(model, dataset, epochs=10, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"--- Training POLICY Net on {device} ---")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for states, actions, _ in dataloader: # Ignore values
            states, actions = states.to(device), actions.to(device)
            
            logits = model(states)
            loss = F.nll_loss(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Policy Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "policy_net.pth")
    print("Saved policy_net.pth\n")

def train_value(model, dataset, epochs=10, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"--- Training VALUE Net on {device} ---")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for states, _, values in dataloader: # Ignore actions
            states, values = states.to(device), values.to(device)
            
            preds = model(states)
            loss = F.mse_loss(preds.squeeze(), values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Value Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "value_net.pth")
    print("Saved value_net.pth\n")

# --- 4. MCTS & Agent Logic ---

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {} # {action_idx: Node}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior 
        
    @property
    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, cpuct=1.0):
        if self.visit_count == 0:
            # Encourages exploration for unvisited nodes
            return self.prior # Simple heuristic, or handle as infinity
        u = cpuct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value + u

class SimGame:
    def __init__(self, p1_pos, p2_pos, walls, p1_w, p2_w, turn):
        self.p1 = list(p1_pos)
        self.p2 = list(p2_pos)
        self.walls = list(walls) 
        self.p1_walls = p1_w
        self.p2_walls = p2_w
        self.turn = turn
    
    def get_legal_moves(self):
        # Simplified legal move logic for inference demo
        legal = []
        curr_pos = self.p1 if self.turn == 0 else self.p2
        cx, cy = curr_pos
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = cx+dx, cy+dy
            if 0<=nx<9 and 0<=ny<9:
                legal.append(ny*9 + nx)
        w_left = self.p1_walls if self.turn == 0 else self.p2_walls
        if w_left > 0:
            legal.append(81 + 0) # Dummy H00
        return legal

class MCTS:
    def __init__(self, policy_net, value_net, num_simulations=50):
        self.policy_net = policy_net
        self.value_net = value_net
        self.num_sims = num_simulations
        self.cpuct = 1.0

    def search(self, game_state_dict):
        root = MCTSNode(parent=MCTSNode(), prior=0) 
        root.parent.visit_count = 1 # Hack to prevent div by zero in root UCB
        
        # 1. State to Tensor
        state_tensor = self._dict_to_tensor(game_state_dict).to(device)
        
        # 2. Simulations
        for _ in range(self.num_sims):
            node = root
            sim_game = self._clone_game(game_state_dict)
            
            # Selection
            # Note: In a real implementation, you traverse down the tree until you hit a leaf.
            # Here we just check one level deep for demonstration.
            if node.children:
                 action_idx, node = max(node.children.items(), key=lambda item: item[1].ucb_score(self.cpuct))
            
            # Evaluation (Dual Nets)
            with torch.no_grad():
                p_logits = self.policy_net(state_tensor.unsqueeze(0))
                v_val = self.value_net(state_tensor.unsqueeze(0))
            
            probs = torch.exp(p_logits).cpu().numpy()[0]
            value = v_val.item()
            
            # Expansion
            legal_moves = sim_game.get_legal_moves()
            for idx in legal_moves:
                if idx not in node.children:
                    node.children[idx] = MCTSNode(parent=node, prior=probs[idx])
            
            # Backprop
            curr = node
            while curr.parent is not None:
                curr.visit_count += 1
                curr.value_sum += value 
                curr = curr.parent
                value = -value 
        
        # Select best move
        if not root.children:
            return None
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return decode_action(best_action)

    def _dict_to_tensor(self, d):
        row = {
            'ActivePlayer': d['turn'],
            'P1_X': d['p1'][0], 'P1_Y': d['p1'][1],
            'P2_X': d['p2'][0], 'P2_Y': d['p2'][1],
            'P1_Walls': d['p1_walls'], 'P2_Walls': d['p2_walls'],
            'Board_Walls': ";".join(d['walls'])
        }
        return torch.tensor(encode_board(row))

    def _clone_game(self, d):
        return SimGame(d['p1'], d['p2'], d['walls'], d['p1_walls'], d['p2_walls'], d['turn'])

# --- 5. Main Entry ---

def main():
    csv_file = "quoridor_trajectories.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Init Models
    policy_model = PolicyNet().to(device)
    value_model = ValueNet().to(device)

    # Train?
    choice = input("Train Nets? (y/n): ")
    if choice.lower() == 'y':
        dataset = QuoridorDataset(csv_file)
        
        print("\n--- Phase 1: Training Policy Net ---")
        train_policy(policy_model, dataset, epochs=10)
        
        print("\n--- Phase 2: Training Value Net ---")
        train_value(value_model, dataset, epochs=10)
    else:
        # Load
        if os.path.exists("policy_net.pth"):
            policy_model.load_state_dict(torch.load("policy_net.pth", map_location=device))
            print("Loaded Policy Net.")
        if os.path.exists("value_net.pth"):
            value_model.load_state_dict(torch.load("value_net.pth", map_location=device))
            print("Loaded Value Net.")

    # Inference
    print("\n--- Testing MCTS Agent (Dual Nets) ---")
    policy_model.eval()
    value_model.eval()
    
    mcts = MCTS(policy_model, value_model, num_simulations=20)
    
    # Mock current state
    current_state = {
        'turn': 0,
        'p1': (4, 8), 'p2': (4, 0),
        'p1_walls': 10, 'p2_walls': 10,
        'walls': []
    }
    
    print("Agent thinking...")
    action = mcts.search(current_state)
    print(f"Agent chose action: {action}")

if __name__ == "__main__":
    main()