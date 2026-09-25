"""Test self-exploration mechanism during transit phase."""
import numpy as np

class LocalTransitExplorer:
    def __init__(self, epsilon=0.2, lr=0.2):
        self.epsilon = epsilon
        self.lr = lr
        # Candidate transit primitives:
        # 0: hand_x_plus, 1: hand_x_minus, 2: hand_y_plus, 3: hand_y_minus,
        # 16: base_forward, 17: base_backward, 18: base_turn_left, 19: base_turn_right
        # Fetch arm horizontal primitives: 0..3
        self.candidate_actions = [0, 1, 2, 3]
        # Q-table keyed by 8 discrete heading sectors toward goal in root frame
        # Sector: 0..7 (each 45 degrees)
        self.q_table = np.zeros((8, len(self.candidate_actions)), dtype=np.float32)
        self.last_state_sector = None
        self.last_action_idx = None
        self.last_dist_xy = None
        self.rng = np.random.default_rng(42)
        self.exploration_steps = 0
        self.successful_explorations = 0

    def get_sector(self, cube, goal, root):
        delta_world = goal - cube
        delta_root = root.T @ delta_world
        angle = np.arctan2(delta_root[1], delta_root[0])
        # Map [-pi, pi] to 0..7
        sector = int(np.floor((angle + np.pi) / (2 * np.pi / 8))) % 8
        return sector

    def update_and_select(self, cube, goal, root, grasped):
        curr_dist_xy = np.linalg.norm((goal - cube)[:2])
        sector = self.get_sector(cube, goal, root)

        # Update Q-value if we took an exploration action in previous step
        if self.last_state_sector is not None and self.last_action_idx is not None and self.last_dist_xy is not None:
            # Reward: progress toward goal, penalty if dropped
            if not grasped:
                reward = -50.0
            else:
                progress = self.last_dist_xy - curr_dist_xy
                reward = progress * 1000.0  # e.g., 0.01m progress -> +10 reward
            
            old_q = self.q_table[self.last_state_sector, self.last_action_idx]
            self.q_table[self.last_state_sector, self.last_action_idx] += self.lr * (reward - old_q)
            if reward > 0:
                self.successful_explorations += 1

        self.exploration_steps += 1
        # Select action using epsilon-greedy over candidate actions
        if self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(0, len(self.candidate_actions)))
        else:
            action_idx = int(np.argmax(self.q_table[sector]))

        chosen_action = self.candidate_actions[action_idx]
        self.last_state_sector = sector
        self.last_action_idx = action_idx
        self.last_dist_xy = curr_dist_xy
        return chosen_action

def main():
    print("LocalTransitExplorer instantiated successfully.")
    explorer = LocalTransitExplorer()
    cube = np.array([0.5, 0.0, 0.8])
    goal = np.array([0.6, 0.1, 0.8])
    root = np.eye(3)
    for _ in range(5):
        act = explorer.update_and_select(cube, goal, root, grasped=True)
        print(f"Step chosen action: {act}")
        # simulate progress
        cube += np.array([0.01, 0.01, 0.0])

if __name__ == "__main__":
    main()
