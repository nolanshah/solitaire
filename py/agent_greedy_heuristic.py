from klondike import Action, Location, Klondike
import random
import argparse
import time
from typing import List, Tuple, Dict


def sample_action(valid_actions: list[Action], params: list[int] = [5, 40, 20, 10, 20, 1, 5]) -> Action:
    if len(valid_actions) == 0:
        msg = "No valid actions"
        raise ValueError(msg)

    if len(valid_actions) == 1:
        return valid_actions[0]

    weight_map: dict[Action, int] = {}
    for action in valid_actions:
        if action.from_location == Location.STOCK and action.to_location == Location.WASTE:
            weight_map[action] = params[0]
        elif action.from_location == Location.WASTE and action.to_location == Location.STOCK:
            weight_map[action] = params[1]
        elif action.from_location == Location.WASTE and action.to_location in Location.foundations():
            weight_map[action] = params[2]
        elif action.from_location == Location.WASTE and action.to_location in Location.tableaus():
            weight_map[action] = params[3]
        elif action.from_location in Location.tableaus() and action.to_location in Location.foundations():
            weight_map[action] = params[4]
        elif action.from_location in Location.foundations() and action.to_location in Location.tableaus():
            weight_map[action] = params[5]
        elif action.from_location in Location.tableaus() and action.to_location in Location.tableaus():
            weight_map[action] = params[6]
        else:
            raise ValueError(f"Invalid action: {action}")

    total = sum(weight_map.values())
    r = random.uniform(0, total)
    upto = 0
    for item, weight in weight_map.items():
        if upto + weight >= r:
            return item
        upto += weight
    raise AssertionError


def play_game(max_steps: int = 1000, verbose: bool = False, print_interval: int = 0) -> Tuple[bool, int]:
    """
    Play a single game of Klondike solitaire using the greedy heuristic agent.

    Args:
        max_steps: Maximum number of steps before giving up
        verbose: Whether to print game progress
        print_interval: Number of turns after which to print game state (0 = never print)

    Returns:
        Tuple of (win status, number of steps taken)
    """
    game = Klondike()
    game.reset()
    steps = 0

    # Keep track of seen states to avoid loops
    seen_states: Dict[str, int] = {}

    while steps < max_steps:
        # Check if game is won
        if game.is_done():
            if verbose:
                print(f"Game won in {steps} steps!")
            return True, steps

        # Get valid actions
        valid_actions = game.get_all_legal_actions()
        if not valid_actions:
            if verbose:
                print(f"No more valid actions after {steps} steps. Game lost.")
            return False, steps

        # Choose an action using the heuristic
        action = sample_action(valid_actions)

        # Apply the action
        game.step(action)
        steps += 1

        # Print game state at specified intervals
        if print_interval > 0 and steps % print_interval == 0:
            render = game.render()
            print(f"\n=== Game state at step {steps} ===")
            print(f"Foundations filled: {sum(len(f) for f in game.foundations)}/52")
            print(f"Stock: {len(game.stock)} cards remaining")
            print(f"Waste: {len(game.waste)} cards")
            print("Foundations:", [f"F{i+1}: {card}" for i, card in enumerate(render.foundations)])
            print("Tableaus:")
            for i, tableau in enumerate(render.tableaus):
                visible_cards = [card for card in tableau if card is not None]
                hidden_count = len(tableau) - len(visible_cards)
                print(f"  T{i+1}: {hidden_count} hidden, {visible_cards}")
            print("Last move:", game.history[-1] if game.history else "None")
            print("=" * 40)

        if verbose and steps % 100 == 0:
            print(f"Step {steps}, foundations filled: {sum(len(f) for f in game.foundations)}/52")

        # Simple loop detection
        game_state = str(game.render().asdict())
        if game_state in seen_states:
            seen_states[game_state] += 1
            if seen_states[game_state] > 3:  # Allow revisiting states a few times
                if verbose:
                    print(f"Loop detected after {steps} steps. Game lost.")
                return False, steps
        else:
            seen_states[game_state] = 1

    if verbose:
        print(f"Reached maximum steps ({max_steps}). Game lost.")
    return False, steps


def play_multiple_games(num_games: int = 100, max_steps: int = 1000, print_interval: int = 0) -> None:
    """
    Play multiple games and report statistics.

    Args:
        num_games: Number of games to play
        max_steps: Maximum steps per game
        print_interval: Number of turns after which to print game state (0 = never print)
    """
    wins = 0
    total_steps = 0
    step_counts: List[int] = []

    start_time = time.time()

    for i in range(num_games):
        print(f"Playing game {i+1}/{num_games}...", end="\r")
        win, steps = play_game(max_steps=max_steps, print_interval=print_interval)
        if win:
            wins += 1
        total_steps += steps
        step_counts.append(steps)

    end_time = time.time()
    duration = end_time - start_time

    win_rate = (wins / num_games) * 100
    avg_steps = total_steps / num_games

    print(f"\nResults from {num_games} games:")
    print(f"Win rate: {win_rate:.2f}% ({wins}/{num_games})")
    print(f"Average steps per game: {avg_steps:.2f}")
    print(f"Time taken: {duration:.2f} seconds ({duration/num_games:.2f} seconds per game)")


def main() -> None:
    parser = argparse.ArgumentParser(description='Play Klondike Solitaire with a greedy heuristic agent')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games to play (default: 1)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per game (default: 1000)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed game progress')
    parser.add_argument('--print-interval', type=int, default=0,
                        help='Print game state every N steps (default: 0 = never)')

    args = parser.parse_args()

    if args.games == 1:
        win, steps = play_game(max_steps=args.max_steps, verbose=args.verbose, print_interval=args.print_interval)
        print(f"Game {'won' if win else 'lost'} after {steps} steps")
    else:
        play_multiple_games(num_games=args.games, max_steps=args.max_steps, print_interval=args.print_interval)


if __name__ == "__main__":
    main()
