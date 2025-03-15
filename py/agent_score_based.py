from klondike import Action, Location, Klondike, Number
import argparse
import time
from typing import List, Dict, Tuple, Callable
import random


def score_action(game: Klondike, action: Action) -> float:
    """
    Assigns a score to an action based on heuristics.
    Higher scores are better.

    Args:
        game: The current game state
        action: The action to evaluate

    Returns:
        A score for the action
    """
    score = 0.0

    # Action priority based on source and destination
    if action.from_location == Location.STOCK and action.to_location == Location.WASTE:
        # Calculate how many times we've cycled through the stock
        # Lower score the more we've cycled to avoid endless looping
        cycles = 0
        for hist_action in game.history:
            if hist_action.from_location == Location.WASTE and hist_action.to_location == Location.STOCK:
                cycles += 1

        # Draw from stock has medium priority, but decreases with each cycle
        # We still want to explore new cards
        score += max(5.0, 40.0 - (cycles * 5.0))
    elif action.from_location == Location.WASTE and action.to_location == Location.STOCK:
        # Recycling waste to stock is a last resort
        # But we need to do it sometimes to see all cards
        # Check if we've already seen all cards
        if len(game.waste) == 0 or len(game.stock) > 0:
            score += 1.0  # Very low priority if we still have cards in stock
        else:
            score += 30.0  # Higher priority if we need to recycle to continue
    elif action.from_location == Location.WASTE and action.to_location in Location.foundations():
        # Moving to foundation is always good, highest priority
        score += 100.0
    elif action.from_location in Location.tableaus() and action.to_location in Location.foundations():
        # Moving from tableau to foundation is also very good
        # Especially if it's a top card with a hidden card beneath
        from_tableau_idx = Location.tableaus().index(action.from_location)
        from_tableau = game.tableaus[from_tableau_idx]

        if from_tableau.hidden_len() > 0 and action.change_index == from_tableau.visible_len() - 1:
            # This will uncover a hidden card - very high priority
            score += 95.0
        else:
            score += 90.0
    elif action.from_location == Location.WASTE and action.to_location in Location.tableaus():
        # Moving from waste to tableau is good
        # It's better to move cards from waste than to keep drawing
        waste_card = game.waste.inspect_top()
        if waste_card and waste_card.number == Number.KING:
            # Moving a King to an empty tableau is very good
            score += 85.0
        else:
            score += 70.0
    elif action.from_location in Location.tableaus() and action.to_location in Location.tableaus():
        # Moving between tableaus can be useful to uncover cards
        from_tableau_idx = Location.tableaus().index(action.from_location)
        from_tableau = game.tableaus[from_tableau_idx]
        to_tableau_idx = Location.tableaus().index(action.to_location)
        to_tableau = game.tableaus[to_tableau_idx]

        # Extra points for creating an empty tableau for a King
        is_creating_empty_tableau = len(from_tableau) == from_tableau.visible_len() and action.change_index == 0
        has_king_in_waste_or_tableau = False

        # Check if we have a King in waste
        waste_top = game.waste.inspect_top()
        if waste_top is not None and waste_top.number == Number.KING:
            has_king_in_waste_or_tableau = True

        # Check if we have a King in any other tableau that's not on an empty tableau
        for i, tab in enumerate(game.tableaus):
            if i != from_tableau_idx and i != to_tableau_idx and tab.visible_len() > 0:
                card_idx = 0
                while card_idx < tab.visible_len():
                    card = tab.visible.cards[card_idx]
                    if card.number == Number.KING:
                        has_king_in_waste_or_tableau = True
                        break
                    card_idx += 1

        # Reward uncovering hidden cards
        if action.change_index == 0 and from_tableau.hidden_len() > 0:
            # This will uncover a hidden card - high priority
            if is_creating_empty_tableau and has_king_in_waste_or_tableau:
                score += 88.0  # Very high if it creates an empty space for a King
            else:
                score += 80.0  # Still high priority
        else:
            # Prefer moving larger stacks that help build sequences
            from_tableau_card_idx = action.change_index
            from_tableau_card_from_top = from_tableau.visible_len() - from_tableau_card_idx - 1
            cards_being_moved = from_tableau_card_from_top + 1

            # Higher score for moving more cards (building longer sequences)
            if len(to_tableau) == 0:  # Moving to empty tableau
                # Check if we're moving a King to an empty tableau
                card_being_moved = None
                if from_tableau.visible_len() > action.change_index:
                    card_being_moved = from_tableau.visible.cards[action.change_index]

                if has_king_in_waste_or_tableau and card_being_moved and card_being_moved.number != Number.KING:
                    score += 10.0  # Low priority if we're not moving a King to empty tableau
                else:
                    score += 60.0 + cards_being_moved * 2.0
            else:
                score += 50.0 + cards_being_moved * 2.0
    elif action.from_location in Location.foundations() and action.to_location in Location.tableaus():
        # Moving from foundation to tableau is rarely optimal but sometimes necessary
        # Only do this if it helps uncover a hidden card or lets us move a large sequence
        to_tableau_idx = Location.tableaus().index(action.to_location)
        to_tableau = game.tableaus[to_tableau_idx]

        # Check if this will help build a longer sequence
        can_build_sequence = False
        for tab_idx, tableau in enumerate(game.tableaus):
            if tab_idx != to_tableau_idx and tableau.visible_len() > 0:
                # Check if there's a sequence that could be moved after this foundation card
                top_card = to_tableau.inspect_top()
                if top_card and tableau.visible_len() > 0:
                    foundation_idx = Location.foundations().index(action.from_location)
                    foundation_card = game.foundations[foundation_idx].inspect_top()
                    tableau_card = tableau.visible.cards[0]  # Check the bottom visible card

                    if (foundation_card and tableau_card and
                        foundation_card.suit.color != tableau_card.suit.color and
                        int(foundation_card.number) == int(tableau_card.number) + 1):
                        can_build_sequence = True
                        break

        if can_build_sequence:
            score += 65.0  # Higher priority if it helps build a sequence
        else:
            score += 20.0  # Otherwise low priority

    return score

def select_best_action(game: Klondike,
                       score_fn: Callable[[Klondike, Action], float],
                       exploration_factor: float = 0.1) -> Action:
    """
    Selects the action with the highest score with some exploration.

    Args:
        game: The current game state
        score_fn: Function to score actions
        exploration_factor: Probability of selecting a random action (0.0 - 1.0)

    Returns:
        The selected action
    """
    valid_actions = game.get_all_legal_actions()

    if not valid_actions:
        msg = "No valid actions"
        raise ValueError(msg)

    if len(valid_actions) == 1:
        return valid_actions[0]

    # Random exploration to avoid getting stuck in loops
    if random.random() < exploration_factor:
        return random.choice(valid_actions)

    # Score all actions
    scored_actions = [(action, score_fn(game, action)) for action in valid_actions]

    # Sort by score (highest first)
    scored_actions.sort(key=lambda x: x[1], reverse=True)

    # Get the highest score
    highest_score = scored_actions[0][1]

    # Find all actions with scores close to the highest (within 5%)
    top_actions = [action for action, score in scored_actions if score >= highest_score * 0.95]

    # If we have multiple similarly scored actions, pick one randomly
    if len(top_actions) > 1:
        return random.choice(top_actions)

    # Otherwise return the highest-scoring action
    return scored_actions[0][0]


def play_game(max_steps: int = 1000, verbose: bool = False, print_interval: int = 0,
            exploration_factor: float = 0.1) -> Tuple[bool, int]:
    """
    Play a single game of Klondike solitaire using the score-based agent.

    Args:
        max_steps: Maximum number of steps before giving up
        verbose: Whether to print game progress
        print_interval: Number of turns after which to print game state (0 = never print)
        exploration_factor: Probability of selecting a random action (0.0 - 1.0)

    Returns:
        Tuple of (win status, number of steps taken)
    """
    game = Klondike()
    game.reset()
    steps = 0

    # Keep track of seen states to avoid loops
    seen_states: Dict[str, int] = {}

    # Track the last few moves to detect simple loops
    recent_moves: List[str] = []
    stalled_counter = 0

    while steps < max_steps:
        # Check if game is won
        if game.is_done():
            if verbose:
                print(f"Game won in {steps} steps!")
            return True, steps

        # Get and select the best action
        try:
            # Increase exploration if we detect we're not making progress
            current_exploration = exploration_factor
            if stalled_counter > 5:
                # Gradually increase exploration factor if we're stalled
                current_exploration = min(0.5, exploration_factor * (1 + stalled_counter * 0.1))

            action = select_best_action(game, score_action, current_exploration)
        except ValueError:
            if verbose:
                print(f"No more valid actions after {steps} steps. Game lost.")
            return False, steps

        # Apply the action
        game.step(action)
        steps += 1

        # Track if we're making progress by checking foundation cards
        foundation_cards = sum(len(f) for f in game.foundations)
        move_description = f"{action.from_location.value}->{action.to_location.value}"

        # Keep track of the last 10 moves
        if len(recent_moves) >= 10:
            recent_moves.pop(0)
        recent_moves.append(move_description)

        # Check for simple cycling patterns (like just drawing from stock repeatedly)
        if len(recent_moves) >= 6:
            # Check for a pattern of 2-3 moves repeating
            is_cycling = False
            for pattern_len in [2, 3]:
                if len(recent_moves) >= pattern_len * 2:
                    pattern = recent_moves[-pattern_len:]
                    prev_pattern = recent_moves[-2*pattern_len:-pattern_len]
                    if pattern == prev_pattern:
                        is_cycling = True
                        break

            if is_cycling:
                stalled_counter += 1
                if verbose and stalled_counter % 5 == 0:
                    print(f"Detected cycling at step {steps}, increasing exploration...")
            else:
                stalled_counter = 0

        # Print game state at specified intervals
        if print_interval > 0 and steps % print_interval == 0:
            render = game.render()
            print(f"\n=== Game state at step {steps} ===")
            print(f"Foundations filled: {sum(len(f) for f in game.foundations)}/52")
            print(f"Stock: {game.stock} cards remaining")
            print(f"Waste: {game.waste} cards")
            print("Foundations:", [f"F{i+1}: {card}" for i, card in enumerate(render.foundations)])
            print("Tableaus:")
            for i, tableau in enumerate(render.tableaus):
                visible_cards = [card or "???" for card in tableau]
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


def play_multiple_games(num_games: int = 100, max_steps: int = 1000,
                    print_interval: int = 0, exploration_factor: float = 0.1) -> None:
    """
    Play multiple games and report statistics.

    Args:
        num_games: Number of games to play
        max_steps: Maximum steps per game
        print_interval: Number of turns after which to print game state (0 = never print)
        exploration_factor: Probability of selecting a random action (0.0 - 1.0)
    """
    wins = 0
    total_steps = 0
    step_counts: List[int] = []

    start_time = time.time()

    for i in range(num_games):
        print(f"Playing game {i+1}/{num_games}...", end="\r")
        win, steps = play_game(
            max_steps=max_steps,
            print_interval=print_interval,
            exploration_factor=exploration_factor
        )
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
    parser = argparse.ArgumentParser(description='Play Klondike Solitaire with a score-based agent')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games to play (default: 1)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per game (default: 1000)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed game progress')
    parser.add_argument('--print-interval', type=int, default=0,
                        help='Print game state every N steps (default: 0 = never)')
    parser.add_argument('--exploration', type=float, default=0.1,
                        help='Exploration factor (0.0-1.0) for random action selection (default: 0.1)')

    args = parser.parse_args()

    if args.games == 1:
        win, steps = play_game(
            max_steps=args.max_steps,
            verbose=args.verbose,
            print_interval=args.print_interval,
            exploration_factor=args.exploration
        )
        print(f"Game {'won' if win else 'lost'} after {steps} steps")
    else:
        play_multiple_games(
            num_games=args.games,
            max_steps=args.max_steps,
            print_interval=args.print_interval,
            exploration_factor=args.exploration
        )


if __name__ == "__main__":
    main()
