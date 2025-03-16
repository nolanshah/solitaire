from klondike import Action, Location, Klondike, Number
import argparse
import time
import random
import copy
import concurrent.futures
from typing import List, Dict, Tuple, Callable, Optional, Set


def score_state(game: Klondike) -> float:
    """
    Scores the current game state.
    Higher scores are better.
    
    Args:
        game: The current game state
        
    Returns:
        A score for the current state
    """
    # Base score starts at 0
    score = 0.0
    
    # Reward cards in the foundation (primary goal)
    foundation_cards = sum(len(foundation) for foundation in game.foundations)
    score += foundation_cards * 100.0  # High reward for each card in foundation
    
    # Reward uncovered cards in tableaus (cards we can see and use)
    visible_cards = sum(tableau.visible_len() for tableau in game.tableaus)
    score += visible_cards * 10.0
    
    # Penalize hidden cards (cards we can't see yet)
    hidden_cards = sum(tableau.hidden_len() for tableau in game.tableaus)
    score -= hidden_cards * 15.0
    
    # Reward empty tableaus that we can use for Kings
    empty_tableaus = sum(1 for tableau in game.tableaus if len(tableau) == 0)
    score += empty_tableaus * 20.0
    
    # Reward available kings in waste or visible tableaus that we could move to empty spaces
    kings_available = 0
    # Check waste for kings
    waste_top = game.waste.inspect_top()
    if waste_top is not None and waste_top.number == Number.KING:
        kings_available += 1
    
    # Check tableaus for kings
    for tableau in game.tableaus:
        if tableau.visible_len() > 0:
            for card_idx in range(tableau.visible_len()):
                if tableau.visible.cards[card_idx].number == Number.KING:
                    kings_available += 1
                    break  # Only count one king per tableau
    
    # Bonus if we have both empty tableaus and kings to move there
    if empty_tableaus > 0 and kings_available > 0:
        score += min(empty_tableaus, kings_available) * 30.0
    
    # Penalize cycling through the stock too many times
    waste_to_stock_cycles = 0
    for action in game.history:
        if action.from_location == Location.WASTE and action.to_location == Location.STOCK:
            waste_to_stock_cycles += 1
    
    score -= waste_to_stock_cycles * 40.0  # Significant penalty for cycling
    
    return score


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
    # Base score
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
        if waste_card is not None and waste_card.number == Number.KING:
            # Moving a King to an empty tableau is very good
            to_tableau_idx = Location.tableaus().index(action.to_location)
            to_tableau = game.tableaus[to_tableau_idx]
            if len(to_tableau) == 0:
                score += 85.0
            else:
                score += 70.0
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


class GameStateStack:
    """
    A class that manages a stack of game states to avoid deep copying.
    Allows us to push actions onto the stack, evaluate states, and undo actions.
    """
    def __init__(self, game: Klondike):
        # We need to work with a copy of the game to avoid modifying the original
        self.game = game
        self.history_stack: List[int] = []  # Tracks how many actions to undo at each step
        self.original_actions = game.get_all_legal_actions()  # Original set of legal actions
    
    def push_action(self, action: Action) -> bool:
        """
        Apply an action and push it onto the stack.
        Returns True if action was successful, False otherwise.
        """
        # First check if the action is valid (in our original set of legal actions)
        if action not in self.original_actions:
            self.history_stack.append(0)  # No history items added
            return False
            
        history_len = len(self.game.history)
        try:
            self.game.step(action)
            # Save how many history items this action added (usually 1)
            self.history_stack.append(len(self.game.history) - history_len)
            return True
        except (ValueError, AssertionError):
            # Action was invalid or failed (revert any partial changes)
            self.history_stack.append(0)
            return False
    
    def pop_action(self) -> None:
        """
        Undo the last action(s) pushed on the stack.
        """
        if not self.history_stack:
            return
            
        # Get how many history items we need to undo
        actions_to_undo = self.history_stack.pop()
        
        # Undo that many actions
        for _ in range(actions_to_undo):
            self.game.undo()
    
    def get_state_key(self) -> str:
        """
        Get a string key representing the current game state.
        """
        try:
            return str(self.game.render().asdict())
        except AssertionError:
            # If render fails, return a unique string
            return f"invalid_state_{len(self.history_stack)}"
    
    def get_score(self) -> float:
        """
        Get the score of the current game state.
        """
        try:
            return score_state(self.game)
        except AssertionError:
            # If scoring fails, return a very low score
            return float('-inf')
    
    def is_done(self) -> bool:
        """
        Check if the game is done.
        """
        try:
            return self.game.is_done()
        except AssertionError:
            # If checking fails, assume not done
            return False
    
    def get_legal_actions(self) -> List[Action]:
        """
        Get all legal actions from the current state.
        """
        try:
            return self.game.get_all_legal_actions()
        except AssertionError:
            # If getting legal actions fails, return empty list
            return []


def evaluate_action_with_lookahead(game: Klondike, action: Action, depth: int = 3) -> float:
    """
    Evaluates an action by looking ahead a few moves without deep copying the game state.
    Uses a stack-based approach to apply and undo actions.
    
    Args:
        game: The current game state
        action: The action to evaluate
        depth: How many moves to look ahead
        
    Returns:
        A score incorporating the immediate and potential future rewards
    """
    try:
        # Create a game state stack to manage applying/undoing actions
        # We need to make a copy of the game state to avoid modifying the original
        game_copy = game.copy()
        state_stack = GameStateStack(game_copy)
        
        # Apply the initial action
        action_success = state_stack.push_action(action)
        if not action_success:
            # If the action is invalid, return a very low score
            return float('-inf')
        
        # Score after the first action
        immediate_score = state_stack.get_score()
        
        # If we've reached the desired depth or the game is won, return the immediate score
        if depth <= 1 or state_stack.is_done():
            # Undo the action before returning
            state_stack.pop_action()
            return immediate_score
        
        # Track visited states to avoid loops
        visited_states: Set[str] = set()
        visited_states.add(state_stack.get_state_key())
        
        # Internal function to explore future states
        def explore_future_states(current_depth: int) -> float:
            if current_depth <= 0 or state_stack.is_done():
                return state_stack.get_score()
            
            # Get all legal actions
            future_actions = state_stack.get_legal_actions()
            if not future_actions:
                return state_stack.get_score()
            
            # Calculate scores for each possible future action
            future_scores = []
            
            for future_action in future_actions:
                # Skip actions that involve hidden cards - we can't predict these
                if (future_action.from_location in Location.tableaus() and 
                    future_action.to_location == Location.WASTE and
                    len(game_copy.stock) == 0):
                    # Skip this action as it involves a hidden card
                    continue
                    
                # Push this action onto our stack
                if state_stack.push_action(future_action):
                    # Check if we've seen this state before to avoid loops
                    state_key = state_stack.get_state_key()
                    if state_key not in visited_states:
                        visited_states.add(state_key)
                        
                        # Recursively explore future moves
                        future_score = explore_future_states(current_depth - 1)
                        future_scores.append(future_score)
                    
                    # Always undo the action after exploring
                    state_stack.pop_action()
            
            # Return the best possible future score, or current state score if no future moves
            if future_scores:
                return max(future_scores)
            else:
                return state_stack.get_score()
        
        # Get the best future score from this action
        future_score = explore_future_states(depth - 1)
        
        # Undo the initial action before returning
        state_stack.pop_action()
        
        # Combine immediate and future score with a discount factor
        # We value immediate rewards more than potential future rewards
        discount_factor = 0.7  # Adjust this value to control the balance
        combined_score = immediate_score + discount_factor * future_score
        
        return combined_score
    except Exception:
        # If any exception occurs during evaluation, return a low score
        # This is a fallback to ensure the agent can continue playing
        return float('-inf')


def evaluate_action_parallel(game: Klondike, action: Action, depth: int) -> Tuple[Action, float]:
    """Helper function for parallel evaluation of actions"""
    try:
        score = evaluate_action_with_lookahead(game, action, depth=depth)
        return action, score
    except Exception:
        return action, float('-inf')


def select_best_action(game: Klondike, 
                      lookahead_depth: int = 3,
                      exploration_factor: float = 0.1,
                      max_workers: int = 4) -> Action:
    """
    Selects the action with the highest score using parallel lookahead evaluation.
    
    Args:
        game: The current game state
        lookahead_depth: How many moves to look ahead
        exploration_factor: Probability of selecting a random action (0.0 - 1.0)
        max_workers: Maximum number of parallel workers for evaluation
        
    Returns:
        The selected action
    """
    try:
        valid_actions = game.get_all_legal_actions()
        
        if not valid_actions:
            msg = "No valid actions"
            raise ValueError(msg)
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Random exploration to avoid getting stuck in loops
        if random.random() < exploration_factor:
            return random.choice(valid_actions)
        
        # Create a separate copy of the game for each worker
        game_copies = [game.copy() for _ in range(min(max_workers, len(valid_actions)))]
        
        # Score all actions with lookahead in parallel
        scored_actions = []
        
        # Use ThreadPoolExecutor for parallelization
        # ProcessPoolExecutor would be faster but has more overhead and serialization issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all actions for evaluation
            future_to_action = {
                executor.submit(
                    evaluate_action_parallel, 
                    game_copies[i % len(game_copies)],  # Reuse game copies
                    action, 
                    lookahead_depth
                ): action for i, action in enumerate(valid_actions)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_action):
                try:
                    action, score = future.result()
                    scored_actions.append((action, score))
                except Exception:
                    # If evaluation fails, assign a very low score
                    action = future_to_action[future]
                    scored_actions.append((action, float('-inf')))
        
        # Filter out actions with -inf scores (invalid actions)
        valid_scored_actions = [(a, s) for a, s in scored_actions if s > float('-inf')]
        
        # If we have no valid scored actions, fall back to random
        if not valid_scored_actions:
            return random.choice(valid_actions)
        
        # Sort by score (highest first)
        valid_scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        highest_score = valid_scored_actions[0][1]
        
        # Find all actions with scores close to the highest (within 5%)
        top_actions = [action for action, score in valid_scored_actions 
                    if score >= highest_score * 0.95]
        
        # If we have multiple similarly scored actions, pick one randomly
        if len(top_actions) > 1:
            return random.choice(top_actions)
        
        # Otherwise return the highest-scoring action
        return valid_scored_actions[0][0]
    
    except Exception:
        # If anything goes wrong, fall back to random selection
        try:
            valid_actions = game.get_all_legal_actions()
            if valid_actions:
                return random.choice(valid_actions)
            else:
                # If no valid actions, return a fallback action
                return Action(Location.STOCK, Location.WASTE)
        except Exception:
            # If we can't even get valid actions, pick the first valid move we can find
            # Try moving from stock to waste first (common valid move)
            return Action(Location.STOCK, Location.WASTE)


def play_game(max_steps: int = 1000, verbose: bool = False, print_interval: int = 0, 
            lookahead_depth: int = 3, exploration_factor: float = 0.1, 
            max_workers: int = 4) -> Tuple[bool, int]:
    """
    Play a single game of Klondike solitaire using the parallel lookahead agent.

    Args:
        max_steps: Maximum number of steps before giving up
        verbose: Whether to print game progress
        print_interval: Number of turns after which to print game state (0 = never print)
        lookahead_depth: How many moves to look ahead
        exploration_factor: Probability of selecting a random action (0.0 - 1.0)
        max_workers: Maximum number of parallel workers for action evaluation

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
                
            action = select_best_action(
                game, 
                lookahead_depth=lookahead_depth,
                exploration_factor=current_exploration,
                max_workers=max_workers
            )
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


def play_game_wrapper(game_id: int, max_steps: int, print_interval: int, 
                    lookahead_depth: int, exploration_factor: float, 
                    max_workers: int, verbose: bool = False) -> Tuple[int, bool, int]:
    """
    Wrapper function for parallel game execution
    
    Returns:
        Tuple of (game_id, win status, number of steps taken)
    """
    if verbose:
        print(f"Starting game {game_id}...")
    
    win, steps = play_game(
        max_steps=max_steps, 
        print_interval=print_interval,
        lookahead_depth=lookahead_depth,
        exploration_factor=exploration_factor,
        max_workers=max_workers,
        verbose=verbose
    )
    
    return game_id, win, steps


def play_multiple_games(num_games: int = 100, max_steps: int = 1000, 
                    print_interval: int = 0, lookahead_depth: int = 3,
                    exploration_factor: float = 0.1, max_workers: int = 4,
                    game_parallelism: int = 1) -> None:
    """
    Play multiple games in parallel and report statistics.

    Args:
        num_games: Number of games to play
        max_steps: Maximum steps per game
        print_interval: Number of turns after which to print game state (0 = never print)
        lookahead_depth: How many moves to look ahead
        exploration_factor: Probability of selecting a random action (0.0 - 1.0)
        max_workers: Maximum number of parallel workers for action evaluation per game
        game_parallelism: Number of games to run in parallel (default: 1)
    """
    wins = 0
    total_steps = 0
    step_counts: List[int] = []
    
    game_workers = min(game_parallelism, num_games)  # Don't create more workers than games
    
    start_time = time.time()
    
    # Track progress
    games_completed = 0
    
    # Play games in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=game_workers) as executor:
        # Create and submit all game tasks
        future_to_game = {
            executor.submit(
                play_game_wrapper, 
                i, 
                max_steps, 
                0,  # No print interval for parallel games to avoid output mess
                lookahead_depth,
                exploration_factor,
                max_workers,
                False  # No verbose output for parallel games
            ): i for i in range(num_games)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_game):
            games_completed += 1
            print(f"Playing games... {games_completed}/{num_games}", end="\r")
            
            try:
                game_id, game_win, game_steps = future.result()
                if game_win:
                    wins += 1
                total_steps += game_steps
                step_counts.append(game_steps)
            except Exception as e:
                print(f"Game encountered an error: {e}")
                # Add a failed game with maximum steps
                step_counts.append(max_steps)
                total_steps += max_steps

    end_time = time.time()
    duration = end_time - start_time

    win_rate = (wins / num_games) * 100
    avg_steps = total_steps / num_games

    print(f"\nResults from {num_games} games:")
    print(f"Win rate: {win_rate:.2f}% ({wins}/{num_games})")
    print(f"Average steps per game: {avg_steps:.2f}")
    print(f"Time taken: {duration:.2f} seconds ({duration/num_games:.2f} seconds per game)")
    print(f"Configuration: lookahead={lookahead_depth}, workers={max_workers}, exploration={exploration_factor}, game_parallelism={game_workers}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Play Klondike Solitaire with a parallel lookahead agent')
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
    parser.add_argument('--lookahead', type=int, default=3,
                        help='Number of moves to look ahead (default: 3)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for action evaluation per game (default: 4)')
    parser.add_argument('--game-parallelism', type=int, default=1,
                        help='Number of games to run in parallel (default: 1)')

    args = parser.parse_args()

    if args.games == 1:
        win, steps = play_game(
            max_steps=args.max_steps, 
            verbose=args.verbose, 
            print_interval=args.print_interval,
            lookahead_depth=args.lookahead,
            exploration_factor=args.exploration,
            max_workers=args.workers
        )
        print(f"Game {'won' if win else 'lost'} after {steps} steps")
        print(f"Configuration: lookahead={args.lookahead}, workers={args.workers}, exploration={args.exploration}")
    else:
        play_multiple_games(
            num_games=args.games, 
            max_steps=args.max_steps, 
            print_interval=args.print_interval if args.game_parallelism == 1 else 0,
            lookahead_depth=args.lookahead,
            exploration_factor=args.exploration,
            max_workers=args.workers,
            game_parallelism=args.game_parallelism
        )


if __name__ == "__main__":
    main()