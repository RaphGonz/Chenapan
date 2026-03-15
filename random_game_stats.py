# -*- coding: utf-8 -*-
"""
Play N random games and report outcome breakdown:
  - win          : a player won (value == 1)
  - draw_loop    : state visited >= MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED times
  - draw_limit   : number_of_moves >= MAX_NUMBER_OF_MOVES
  - draw_nomoves : no valid moves left for current player
"""

import numpy as np
from tqdm import trange

from alpha_pan import (
    Chenapan,
    flatten_and_sum_list_of_list,
    MAX_NUMBER_OF_MOVES,
    MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED,
)

NUM_GAMES = 1000


def pick_random_action(valid_moves):
    indices = [i for i, moves in enumerate(valid_moves) if moves]
    action_start = np.random.choice(indices)
    action_end = int(np.random.choice(valid_moves[action_start]))
    return action_start, action_end


def classify_draw(game, state, player):
    if game.biggest_loop >= MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED:
        return "draw_loop"
    if game.number_of_moves >= MAX_NUMBER_OF_MOVES:
        return "draw_limit"
    # no moves
    return "draw_nomoves"


def play_random_game(game):
    game.reset()
    state = game.get_initial_state()
    player = 1
    action = None

    while True:
        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            if value == 1:
                return "win"
            return classify_draw(game, state, player)

        valid_moves = game.get_valid_moves(state, player)

        # safety: if no moves exist, declare terminal (shouldn't reach here since
        # get_value_and_terminated already catches it, but just in case)
        if flatten_and_sum_list_of_list(valid_moves) == 0:
            return "draw_nomoves"

        action = pick_random_action(valid_moves)
        state = game.get_next_state(state, action)
        player = game.get_opponent(player)


def main():
    game = Chenapan()
    counts = {"win": 0, "draw_loop": 0, "draw_limit": 0, "draw_nomoves": 0}

    for _ in trange(NUM_GAMES, desc="Random games"):
        result = play_random_game(game)
        counts[result] += 1

    total = NUM_GAMES
    print(f"\n{'='*48}")
    print(f"  Results over {total} random games")
    print(f"  (MAX_MOVES={MAX_NUMBER_OF_MOVES}, MAX_LOOP={MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED})")
    print(f"{'='*48}")
    labels = {
        "win":          "Win (any player)",
        "draw_loop":    "Draw — loop",
        "draw_limit":   "Draw — move limit",
        "draw_nomoves": "Draw — no moves",
    }
    for key, count in counts.items():
        print(f"  {labels[key]:<22} : {count:>5}  ({count/total:6.1%})")
    print(f"{'='*48}")
    wins = counts["win"]
    draws = total - wins
    print(f"  {'Total wins':<22} : {wins:>5}  ({wins/total:6.1%})")
    print(f"  {'Total draws':<22} : {draws:>5}  ({draws/total:6.1%})")
    print(f"{'='*48}")


if __name__ == "__main__":
    main()
