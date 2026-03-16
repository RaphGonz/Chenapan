"""
gui.py — Alpha-Pan pygame GUI
Human-vs-AI interface for the Chenapan board game.
Phase 02, Plan 02: AI background thread, checkpoint load, game-over screen, win bar.
"""

import sys
import threading
import glob
import os

import pygame
import torch
import numpy as np
from enum import Enum, auto

from alpha_pan import (
    Chenapan,
    AlphaPanNet,
    MCTS,
    MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED,
    MAX_NUMBER_OF_MOVES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_SIZE = 100
BOARD_OFFSET_X = 20
BOARD_OFFSET_Y = 20
PANEL_WIDTH = 220
WINDOW_WIDTH = BOARD_OFFSET_X + 5 * CELL_SIZE + PANEL_WIDTH + 20
WINDOW_HEIGHT = BOARD_OFFSET_Y + 5 * CELL_SIZE + 20

# Colors (RGB unless noted)
BOARD_BG        = (40,  40,  40)
GRID_COLOR      = (180, 180, 180)
RED_PIECE       = (220, 50,  50)
BLACK_PIECE     = (30,  30,  30)
WHITE_DISK      = (240, 240, 240)
PANEL_BG        = (55,  55,  55)
TEXT_COLOR      = (220, 220, 220)

# RGBA highlight overlays (drawn on a per-surface basis)
HIGHLIGHT_RGBA      = (255, 255, 0,   100)   # valid destination — yellow
AI_LAST_MOVE_RGBA   = (100, 200, 255, 100)   # last AI destination — blue

PIECE_LABEL = {
    0:  "0",
    1:  "A",
    2:  "2",
    3:  "3",
    4:  "4",
    5:  "5",
    6:  "6",
    7:  "7",
    8:  "8",
    9:  "9",
    10: "V",
    11: "D",
    12: "R",
}

PIECE_RADIUS = 38


# ---------------------------------------------------------------------------
# Game state enum
# ---------------------------------------------------------------------------

class GameState(Enum):
    WAITING_FOR_HUMAN = auto()
    AI_THINKING       = auto()
    GAME_OVER         = auto()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_latest_checkpoint(model, device):
    """Load the most recently modified model*.pt file if any exist."""
    candidates = glob.glob("model*.pt")
    if not candidates:
        print("No checkpoint found — model uses random weights")
        model.eval()
        return False
    latest = max(candidates, key=os.path.getmtime)
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {latest}")
    return True


# ---------------------------------------------------------------------------
# AI background thread worker (NO pygame calls inside)
# ---------------------------------------------------------------------------

def run_ai(neutral_state_copy, mcts, gs, ai_lock, ai_done_event):
    """Run MCTS search in a background thread and store the result in gs."""
    action_probs, root_value = mcts.search(neutral_state_copy)
    action = np.unravel_index(np.argmax(action_probs), action_probs.shape)
    with ai_lock:
        gs["ai_action"] = action        # (src_idx, dest_idx) in neutral coordinates
        gs["ai_value"]  = root_value    # scalar [-1, 1], AI's perspective
    ai_done_event.set()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _cell_rect(row: int, col: int) -> pygame.Rect:
    """Return the pygame.Rect for the board cell at (row, col)."""
    x = BOARD_OFFSET_X + col * CELL_SIZE
    y = BOARD_OFFSET_Y + row * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def _draw_highlight(screen: pygame.Surface, row: int, col: int, rgba: tuple) -> None:
    """Draw a semi-transparent SRCALPHA overlay on a single cell."""
    surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    surf.fill(rgba)
    screen.blit(surf, (BOARD_OFFSET_X + col * CELL_SIZE, BOARD_OFFSET_Y + row * CELL_SIZE))


def draw_board(
    screen: pygame.Surface,
    state,
    selected_cell,
    valid_moves,
    ai_last_dest,
) -> None:
    """Render the 5x5 board: background, grid, highlights, and pieces."""

    # 1. Board background
    board_rect = pygame.Rect(
        BOARD_OFFSET_X, BOARD_OFFSET_Y,
        5 * CELL_SIZE, 5 * CELL_SIZE,
    )
    pygame.draw.rect(screen, BOARD_BG, board_rect)

    # 2. Grid lines
    for i in range(6):
        # Vertical
        x = BOARD_OFFSET_X + i * CELL_SIZE
        pygame.draw.line(
            screen, GRID_COLOR,
            (x, BOARD_OFFSET_Y),
            (x, BOARD_OFFSET_Y + 5 * CELL_SIZE),
            1,
        )
        # Horizontal
        y = BOARD_OFFSET_Y + i * CELL_SIZE
        pygame.draw.line(
            screen, GRID_COLOR,
            (BOARD_OFFSET_X, y),
            (BOARD_OFFSET_X + 5 * CELL_SIZE, y),
            1,
        )

    # 3. Valid-destination highlights (only when a piece is selected)
    if selected_cell is not None:
        src_idx = selected_cell[0] * 5 + selected_cell[1]
        if valid_moves is not None:
            for dest_idx in valid_moves[src_idx]:
                dr, dc = divmod(dest_idx, 5)
                _draw_highlight(screen, dr, dc, HIGHLIGHT_RGBA)

    # 4. AI last-move destination highlight
    if ai_last_dest is not None:
        _draw_highlight(screen, ai_last_dest[0], ai_last_dest[1], AI_LAST_MOVE_RGBA)

    # 5. Pieces
    font = pygame.font.SysFont("Arial", 26, bold=True)
    for row in range(5):
        for col in range(5):
            cell_val = int(state[row, col])
            if cell_val == 0:
                # Joker — white disk, black label
                cx = BOARD_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
                cy = BOARD_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(screen, WHITE_DISK, (cx, cy), PIECE_RADIUS)
                label_surf = font.render("0", True, (0, 0, 0))
                label_rect = label_surf.get_rect(center=(cx, cy))
                screen.blit(label_surf, label_rect)
            elif cell_val > 0:
                # Player 1 — red disk, white label
                cx = BOARD_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
                cy = BOARD_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(screen, RED_PIECE, (cx, cy), PIECE_RADIUS)
                label = PIECE_LABEL[abs(cell_val)]
                label_surf = font.render(label, True, (255, 255, 255))
                label_rect = label_surf.get_rect(center=(cx, cy))
                screen.blit(label_surf, label_rect)
            else:
                # Player 2 — black disk, white label
                cx = BOARD_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
                cy = BOARD_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(screen, BLACK_PIECE, (cx, cy), PIECE_RADIUS)
                label = PIECE_LABEL[abs(cell_val)]
                label_surf = font.render(label, True, (255, 255, 255))
                label_rect = label_surf.get_rect(center=(cx, cy))
                screen.blit(label_surf, label_rect)


def draw_panel(screen: pygame.Surface, game: Chenapan, ai_value: float) -> None:
    """Render the side panel to the right of the board."""
    panel_x = BOARD_OFFSET_X + 5 * CELL_SIZE + 10
    panel_y = BOARD_OFFSET_Y
    panel_h = 5 * CELL_SIZE

    # Background
    panel_rect = pygame.Rect(panel_x, panel_y, PANEL_WIDTH, panel_h)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)

    font = pygame.font.SysFont("Arial", 18)
    y_cursor = panel_y + 16

    # Move counter
    moves_surf = font.render(f"Moves: {game.number_of_moves}", True, TEXT_COLOR)
    screen.blit(moves_surf, (panel_x + 10, y_cursor))
    y_cursor += 32

    # Draw loop tracker
    loops_surf = font.render(
        f"Loops: {game.biggest_loop} / {MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED}",
        True,
        TEXT_COLOR,
    )
    screen.blit(loops_surf, (panel_x + 10, y_cursor))
    y_cursor += 48

    # Win probability bar label
    win_label_surf = font.render("Win prob", True, TEXT_COLOR)
    screen.blit(win_label_surf, (panel_x + 10, y_cursor))
    y_cursor += 24

    # Win probability bar
    bar_w = PANEL_WIDTH - 20
    bar_h = 20
    bar_x = panel_x + 10
    bar_y = y_cursor

    # ai_value is in [-1, 1]; convert to [0, 1] proportion
    ai_share = (ai_value + 1.0) / 2.0          # AI (black) share on the left
    ai_share = max(0.0, min(1.0, ai_share))

    ai_bar_w  = int(bar_w * ai_share)
    hu_bar_w  = bar_w - ai_bar_w

    # AI portion (black/dark)
    if ai_bar_w > 0:
        pygame.draw.rect(screen, BLACK_PIECE, pygame.Rect(bar_x, bar_y, ai_bar_w, bar_h))
    # Human portion (red)
    if hu_bar_w > 0:
        pygame.draw.rect(screen, RED_PIECE, pygame.Rect(bar_x + ai_bar_w, bar_y, hu_bar_w, bar_h))

    # Bar border
    pygame.draw.rect(screen, GRID_COLOR, pygame.Rect(bar_x, bar_y, bar_w, bar_h), 1)


# ---------------------------------------------------------------------------
# Click handler
# ---------------------------------------------------------------------------

def handle_click(pixel_pos: tuple, gs: dict, game: Chenapan) -> None:
    """
    Process a mouse click at pixel_pos and update the gs state dict.

    gs keys used:
        state               — current board numpy array
        selected_cell       — (row, col) or None
        game_state          — GameState enum value
        ai_last_dest        — (row, col) or None
        terminal_value      — int, set when game ends
        terminal_player     — int (1 = human, -1 = AI), set when game ends
        ai_thread_launched  — bool, flag for AI thread state machine
    """
    px, py = pixel_pos
    col = (px - BOARD_OFFSET_X) // CELL_SIZE
    row = (py - BOARD_OFFSET_Y) // CELL_SIZE

    if not (0 <= row < 5 and 0 <= col < 5):
        gs["selected_cell"] = None
        return

    state    = gs["state"]
    cell_val = int(state[row, col])
    src      = gs["selected_cell"]

    if src is None:
        # Select only player-1 pieces (positive values); joker and opponent do nothing
        if cell_val > 0:
            gs["selected_cell"] = (row, col)
        return

    # A piece is already selected — attempt to move
    src_idx  = src[0] * 5 + src[1]
    dest_idx = row * 5 + col
    valid_moves = game.get_valid_moves(state, 1)

    if dest_idx in valid_moves[src_idx]:
        # Valid move: apply it
        action = [src_idx, dest_idx]
        game.get_next_state(state, action, update_meta_parameters=True)
        gs["selected_cell"] = None
        gs["ai_last_dest"]  = None   # clear AI highlight on human move

        # Check terminal condition
        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            gs["terminal_value"] = value
            gs["game_state"]     = GameState.GAME_OVER
        else:
            # Trigger AI turn
            gs["game_state"]         = GameState.AI_THINKING
            gs["ai_thread_launched"] = False
    else:
        # Not a valid destination — deselect without moving
        gs["selected_cell"] = None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Alpha-Pan")
    clock = pygame.time.Clock()

    # --- Game and AI setup ---
    game  = Chenapan()
    state = game.get_initial_state()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AlphaPanNet(device)
    load_latest_checkpoint(model, device)   # sets model.eval() internally

    args = {
        "C":                  2,
        "num_searches":       60,
        "dirichlet_epsilon":  0.1,
        "dirichlet_alpha":    0.3,
    }
    mcts = MCTS(game, args, model)

    ai_lock       = threading.Lock()
    ai_done_event = threading.Event()

    gs = {
        "state":             state,
        "game_state":        GameState.WAITING_FOR_HUMAN,
        "selected_cell":     None,
        "ai_last_dest":      None,
        "terminal_value":    0,
        "ai_action":         None,
        "ai_value":          0.0,
        "ai_thread_launched": False,
    }

    font_thinking = pygame.font.SysFont("Arial", 32, bold=True)

    while True:
        # --- Event processing ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if gs["game_state"] == GameState.WAITING_FOR_HUMAN:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    handle_click(event.pos, gs, game)

            elif gs["game_state"] == GameState.GAME_OVER:
                if event.type == pygame.KEYDOWN:
                    # Restart the game
                    game.reset()
                    gs["state"]              = game.get_initial_state()
                    gs["game_state"]         = GameState.WAITING_FOR_HUMAN
                    gs["selected_cell"]      = None
                    gs["ai_last_dest"]       = None
                    gs["terminal_value"]     = 0
                    gs["ai_action"]          = None
                    gs["ai_value"]           = 0.0
                    gs["ai_thread_launched"] = False
                    ai_done_event.clear()

        # --- AI thread state machine (non-blocking) ---
        if gs["game_state"] == GameState.AI_THINKING:
            if not gs["ai_thread_launched"]:
                # Launch AI search in background
                neutral_state_copy = game.change_perspective(gs["state"]).copy()
                ai_done_event.clear()
                gs["ai_thread_launched"] = True
                t = threading.Thread(
                    target=run_ai,
                    args=(neutral_state_copy, mcts, gs, ai_lock, ai_done_event),
                    daemon=True,
                )
                t.start()
            elif ai_done_event.is_set():
                # AI computation done — read result and apply move
                with ai_lock:
                    ai_action = gs["ai_action"]   # (src_idx, dest_idx) in neutral coords
                    ai_value  = gs["ai_value"]

                gs["ai_value"] = ai_value

                # Remap neutral coords back to original-state coords
                # (rot180 maps position i -> position 24-i on a 25-cell board)
                src_orig  = 24 - ai_action[0]
                dest_orig = 24 - ai_action[1]
                action    = [src_orig, dest_orig]

                game.get_next_state(gs["state"], action, update_meta_parameters=True)
                gs["ai_last_dest"]       = (dest_orig // 5, dest_orig % 5)
                gs["ai_thread_launched"] = False

                # Check terminal condition
                value, is_terminal = game.get_value_and_terminated(gs["state"], action)
                if is_terminal:
                    gs["terminal_value"] = value
                    gs["game_state"]     = GameState.GAME_OVER
                else:
                    gs["game_state"] = GameState.WAITING_FOR_HUMAN

        # --- Render ---
        screen.fill(BOARD_BG)

        valid_moves = game.get_valid_moves(gs["state"], 1)

        draw_board(
            screen,
            gs["state"],
            gs["selected_cell"],
            valid_moves,
            gs["ai_last_dest"],
        )
        draw_panel(screen, game, gs["ai_value"])

        # "AI is thinking..." overlay text during computation
        if gs["game_state"] == GameState.AI_THINKING:
            thinking_surf = font_thinking.render("AI is thinking...", True, (200, 200, 255))
            thinking_rect = thinking_surf.get_rect(
                center=(BOARD_OFFSET_X + 5 * CELL_SIZE // 2, BOARD_OFFSET_Y + 5 * CELL_SIZE + 10)
            )
            # Clamp to visible area if near bottom edge
            if thinking_rect.bottom > WINDOW_HEIGHT:
                thinking_rect.bottom = WINDOW_HEIGHT - 4
            screen.blit(thinking_surf, thinking_rect)

        # Game-over overlay
        if gs["game_state"] == GameState.GAME_OVER:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))

            if gs["terminal_value"] == 1:
                joker_pos = np.argwhere(gs["state"] == 0)
                result_text = "You win" if (len(joker_pos) > 0 and joker_pos[0][0] == 0) else "AI wins"
            else:
                result_text = "Draw"

            font_big   = pygame.font.SysFont("Arial", 56)
            font_small = pygame.font.SysFont("Arial", 28)
            t1 = font_big.render(result_text, True, (255, 255, 255))
            t2 = font_small.render("Press any key to restart", True, (200, 200, 200))
            screen.blit(t1, t1.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 30)))
            screen.blit(t2, t2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30)))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
