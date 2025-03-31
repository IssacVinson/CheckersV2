import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import sys
import gym
from gym import spaces

# Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BOARD_SIZE = 8
SQUARE_SIZE = 80  # Increased for better visibility
WINDOW_SIZE = (BOARD_SIZE * SQUARE_SIZE, BOARD_SIZE * SQUARE_SIZE)
FPS = 60
PLAYER1_COLOR = (255, 0, 0)
PLAYER2_COLOR = (0, 0, 255)

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Checkers")
clock = pygame.time.Clock()

class CheckersEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(CheckersEnv, self).__init__()
        self.action_space = spaces.Discrete(64)  # 8x8 board
        self.observation_space = spaces.Box(low=0, high=2, shape=(8, 8), dtype=int)

        self.selected_piece = None  # Make sure this line is properly indented

        # Define players
        self.EMPTY = 0
        self.PLAYER1 = 1
        self.PLAYER2 = 2

        self.board = self.create_board()
        self.current_player = self.PLAYER1
	

    def create_board(self):
        board = np.zeros((8, 8), dtype=int)
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    board[r, c] = self.PLAYER2
                if (7 - r + c) % 2 == 1:
                    board[7 - r, c] = self.PLAYER1
        return board

    def reset(self):
        self.board = self.create_board()
        self.current_player = self.PLAYER1
        return self.board

    def switch_player(self):
        self.current_player = self.PLAYER2 if self.current_player == self.PLAYER1 else self.PLAYER1


    def render(self, mode='console'):
        if mode == 'console':
            print("Current board state:")
            for row in self.board:
                print(' '.join(str(cell) for cell in row))
        else:
            raise NotImplementedError

    def is_valid_position(self, row, col):
        return 0 <= row < 8 and 0 <= col < 8

    def valid_moves(self):
        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r, c] == self.current_player:
                    # Check for regular moves
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        new_r, new_c = r + dr, c + dc
                        if self.is_valid_position(new_r, new_c) and self.board[new_r, new_c] == self.EMPTY:
                            moves.append((r, c, new_r, new_c))

                    # Check for captures
                    for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                        new_r, new_c = r + dr, c + dc
                        mid_r, mid_c = r + dr // 2, c + dc // 2
                        if (self.is_valid_position(new_r, new_c) and self.board[new_r, new_c] == self.EMPTY
                                and self.board[mid_r, mid_c] not in [self.EMPTY, self.current_player]):
                            moves.append((r, c, new_r, new_c))
        return moves

    def translate_click_to_action(self, row, col):
        # If no piece is selected and the clicked square has a piece belonging to the player, select it
        if self.selected_piece is None and self.board[row][col] == self.current_player:
            self.selected_piece = (row, col)
            return None  # No move action, just selection
        # If a piece is already selected, attempt to make a move
        elif self.selected_piece is not None:
            for index, move in enumerate(self.valid_moves()):
                from_row, from_col, to_row, to_col = move
                if (from_row, from_col) == self.selected_piece and (to_row, to_col) == (row, col):
                    self.selected_piece = None  # Deselect piece after move
                    return index  # Return the index of the valid move
            # If the destination is not valid, deselect the piece
            self.selected_piece = None

        return None  # If no valid move action is found


    def move_piece(self, from_row, from_col, to_row, to_col):
        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = self.EMPTY

        # Check for captures
        if abs(from_row - to_row) == 2:
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            self.board[mid_row, mid_col] = self.EMPTY

        # Check for 'kinging'
        if (self.current_player == self.PLAYER1 and to_row == 7) or (self.current_player == self.PLAYER2 and to_row == 0):
            self.board[to_row, to_col] += 2  # Using +2 to represent king state

    def step(self, action):
        # Action will be an index into valid_moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            return self.board, -10, True, {}  # No valid moves, game over

        from_row, from_col, to_row, to_col = valid_actions[action]
        self.move_piece(from_row, from_col, to_row, to_col)

        # Switch to the other player
        self.switch_player()

        reward = 0  # Base reward is 0
        if not self.valid_moves():  # The other player has no moves
            reward = 10  # Current player wins

        return self.board, reward, reward != 0, {}





class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Define your network structure here
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.memory = []
        self.batch_size = 32
        self.memory_size = 10000

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions_count):
        if np.random.rand() <= self.epsilon:
          return random.randrange(valid_actions_count)  # Ensure random choice is within valid range
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy()) % valid_actions_count  # Modulo to ensure it's within range


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(torch.from_numpy(next_state).float().unsqueeze(0))).item()
            
            target_f = self.model(torch.from_numpy(state).float().unsqueeze(0))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(torch.from_numpy(state).float().unsqueeze(0)))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())



# Pygame and Gameplay Functions
def get_square_under_mouse():
    mouse_x, mouse_y = pygame.mouse.get_pos()
    row = mouse_y // SQUARE_SIZE
    col = mouse_x // SQUARE_SIZE
    if row >= BOARD_SIZE or col >= BOARD_SIZE:
        return None
    return row, col

def human_move(env, board):
    action = None
    while action is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                row, col = get_square_under_mouse()
                if row is not None and col is not None:
                    action = env.translate_click_to_action(row, col)
                    if action is not None:
                        return action
        render_board(board)
        pygame.display.flip()

def render_board(board):
    screen.fill(BLACK)  # Fills the entire screen with black before drawing
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            # Draw pieces based on board state
            piece = board[row][col]
            if piece == 1: 
                pygame.draw.circle(screen, PLAYER1_COLOR, (int((col + 0.5) * SQUARE_SIZE), int((row + 0.5) * SQUARE_SIZE)), SQUARE_SIZE // 2 - 5)
            elif piece == 2:
                pygame.draw.circle(screen, PLAYER2_COLOR, (int((col + 0.5) * SQUARE_SIZE), int((row + 0.5) * SQUARE_SIZE)), SQUARE_SIZE // 2 - 5)



# Main Gameplay Loop
env = CheckersEnv()
state_size = 64
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Load model weights if necessary
# agent.model.load_state_dict(torch.load('checkers_model_path_here.pth'))

while True:
    render_board(env.board)
    pygame.display.flip()

    if env.current_player == env.PLAYER1:
        action = human_move(env, env.board)
    else:
        state = env.board.flatten()
        action = agent.act(state, len(env.valid_moves()))

    next_state, reward, done, _ = env.step(action)
    if done:
        print("Game Over")
        break  # Or you can reset the environment with env.reset()
    
    clock.tick(FPS)

# After the game is done, keep the window open until closed by the user
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

