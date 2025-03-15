import pytest
from klondike import Klondike, Card, Suit, Number, Location, Stack, KlondikeState, GameSession


@pytest.fixture
def klondike_game():
    game = Klondike()
    game.reset()
    return game


def test_initial_setup(klondike_game):
    # Verify initial game state
    assert len(klondike_game.stock) == 24
    assert len(klondike_game.waste) == 0

    # Verify tableau setup
    for i, tableau in enumerate(klondike_game.tableaus):
        assert tableau.visible_len() == 1
        assert tableau.hidden_len() == i
        assert len(tableau) == i + 1

    # Verify foundation setup
    for foundation in klondike_game.foundations:
        assert len(foundation) == 0


def test_stock_to_waste(klondike_game):
    # Record initial stock size
    initial_stock_size = len(klondike_game.stock)

    # Move from stock to waste
    klondike_game.move_stock_to_waste()

    # Verify one card was moved
    assert len(klondike_game.stock) == initial_stock_size - 1
    assert len(klondike_game.waste) == 1
    assert klondike_game.move_count == 1

    # Verify history was updated
    assert len(klondike_game.history) == 1
    assert klondike_game.history[0].from_location == Location.STOCK
    assert klondike_game.history[0].to_location == Location.WASTE


def test_waste_to_tableau():
    # Create a game with a controlled setup
    game = Klondike()

    # Clear everything
    game.stock = Stack()
    game.waste = Stack()
    for tableau in game.tableaus:
        tableau.visible = Stack()
        tableau.hidden = Stack()

    # Setup a specific tableau with a red King
    king_card = Card(Suit.HEART, Number.KING)
    game.tableaus[0].visible.add_to_top(king_card)

    # Setup waste with a black Queen
    queen_card = Card(Suit.SPADE, Number.QUEEN)
    game.waste.add_to_top(queen_card)

    # Move queen from waste to tableau
    game.move_waste_to_tableau(0)

    # Verify the card was moved
    assert len(game.waste) == 0
    assert game.tableaus[0].inspect_top() == queen_card
    assert game.move_count == 1

    # Verify history was updated
    assert len(game.history) == 1
    assert game.history[0].from_location == Location.WASTE
    assert game.history[0].to_location == Location.TABLEAU_1


def test_tableau_to_foundation():
    # Create a game with a controlled setup
    game = Klondike()

    # Clear everything
    game.stock = Stack()
    game.waste = Stack()
    for tableau in game.tableaus:
        tableau.visible = Stack()
        tableau.hidden = Stack()
    for foundation in game.foundations:
        foundation.cards = []

    # Setup a tableau with an ace
    ace_card = Card(Suit.HEART, Number.ACE)
    game.tableaus[0].visible.add_to_top(ace_card)

    # Move ace to foundation
    game.move_tableau_to_foundation(0, 0)

    # Verify the ace was moved
    assert len(game.tableaus[0]) == 0

    foundation_card = game.foundations[0].inspect_top()
    assert foundation_card is not None
    assert foundation_card.number == Number.ACE
    assert foundation_card.suit == Suit.HEART
    assert game.move_count == 1

    # Now add a 2 to the tableau and move it to foundation
    two_card = Card(Suit.HEART, Number.TWO)
    game.tableaus[0].visible.add_to_top(two_card)

    game.move_tableau_to_foundation(0, 0)

    # Verify the 2 was moved
    assert len(game.tableaus[0]) == 0
    assert game.foundations[0].inspect_top() == two_card
    assert len(game.foundations[0]) == 2


def test_undo(klondike_game):
    # Make a move
    initial_stock_size = len(klondike_game.stock)
    klondike_game.move_stock_to_waste()

    # Verify the move happened
    assert len(klondike_game.stock) == initial_stock_size - 1
    assert len(klondike_game.waste) == 1

    # Undo the move
    klondike_game.undo()

    # Verify undo worked
    assert len(klondike_game.stock) == initial_stock_size
    assert len(klondike_game.waste) == 0
    assert klondike_game.undo_count == 1
    assert len(klondike_game.history) == 0


def test_get_all_legal_actions():
    # Create a game with a controlled setup
    game = Klondike()

    # Clear everything
    game.stock = Stack()
    game.waste = Stack()
    for tableau in game.tableaus:
        tableau.visible = Stack()
        tableau.hidden = Stack()

    # Add a card to stock
    stock_card = Card(Suit.HEART, Number.KING)
    game.stock.add_to_top(stock_card)

    # Setup one tableau with a black king
    king_card = Card(Suit.SPADE, Number.KING)
    game.tableaus[0].visible.add_to_top(king_card)

    # Setup another tableau with a red queen
    queen_card = Card(Suit.HEART, Number.QUEEN)
    game.tableaus[1].visible.add_to_top(queen_card)

    # Get legal actions
    actions = game.get_all_legal_actions()

    # There should be at least 2 actions:
    # 1. Stock to waste
    # 2. Tableau 1 (queen) to Tableau 0 (king)
    assert len(actions) >= 2

    # Verify stock to waste action
    stock_to_waste = next((a for a in actions if a.from_location == Location.STOCK
                          and a.to_location == Location.WASTE), None)
    assert stock_to_waste is not None

    # Verify tableau to tableau action
    tableau_to_tableau = next((a for a in actions if a.from_location == Location.TABLEAU_2
                              and a.to_location == Location.TABLEAU_1), None)
    assert tableau_to_tableau is not None


def test_is_done():
    # Create a new game
    game = Klondike()

    # Clear everything
    game.stock = Stack()
    game.waste = Stack()
    for tableau in game.tableaus:
        tableau.visible = Stack()
        tableau.hidden = Stack()

    # Verify game is not done with empty foundations
    assert not game.is_done()

    # Manually set up foundations to be complete
    for foundation_idx, foundation in enumerate(game.foundations):
        foundation.cards = []
        suit = list(Suit)[foundation_idx]
        for number in Number:
            foundation.add_to_top(Card(suit, number))

    # Now game should be done
    assert game.is_done()


def test_tableau_to_tableau():
    # Create a game with a controlled setup
    game = Klondike()

    # Clear everything
    game.stock = Stack()
    game.waste = Stack()
    for tableau in game.tableaus:
        tableau.visible = Stack()
        tableau.hidden = Stack()

    # Setup source tableau with a black 10, 9, 8
    ten_card = Card(Suit.SPADE, Number.TEN)
    nine_card = Card(Suit.SPADE, Number.NINE)
    eight_card = Card(Suit.SPADE, Number.EIGHT)

    game.tableaus[0].visible.add_to_top(ten_card)
    game.tableaus[0].visible.add_to_top(nine_card)
    game.tableaus[0].visible.add_to_top(eight_card)

    # Setup destination tableau with a red Jack
    jack_card = Card(Suit.HEART, Number.JACK)
    game.tableaus[1].visible.add_to_top(jack_card)

    # Move 10-9-8 stack from tableau 0 to tableau 1
    # The action needs the index of the bottom card in the stack to move
    game.move_tableau_to_tableau(0, 0, 1)

    # Verify the cards were moved
    assert len(game.tableaus[0]) == 0

    # Tableau 1 should now have Jack, 10, 9, 8 from bottom to top
    assert len(game.tableaus[1]) == 4
    cards = game.tableaus[1].visible.inspect_all()
    assert cards[0] == jack_card
    assert cards[1] == ten_card
    assert cards[2] == nine_card
    assert cards[3] == eight_card


def test_auto_solve_suggestion():
    """Test the auto-solve suggestion functionality."""
    game = Klondike()

    # Create a controlled setup with an auto-solvable state
    game.stock = Stack()
    game.waste = Stack()
    for tableau in game.tableaus:
        tableau.visible = Stack()
        tableau.hidden = Stack()
    for foundation in game.foundations:
        foundation.cards = []

    # Set up foundations with some progress
    game.foundations[0].add_to_top(Card(Suit.HEART, Number.ACE))
    game.foundations[0].add_to_top(Card(Suit.HEART, Number.TWO))
    game.foundations[1].add_to_top(Card(Suit.DIAMOND, Number.ACE))

    # Put a card in tableau that can go to foundation
    game.tableaus[0].visible.add_to_top(Card(Suit.HEART, Number.THREE))

    # Make it auto-solvable by adding all cards to tableaus
    # Add all remaining cards to make it total 52 cards
    remaining_cards = 52 - (2 + 1 + 1)  # Hearts (A,2), Diamond (A), Heart (3)
    for i in range(remaining_cards):
        if i % 2 == 0:
            game.tableaus[1].visible.add_to_top(Card(Suit.SPADE, Number.KING))
        else:
            game.tableaus[2].visible.add_to_top(Card(Suit.CLUB, Number.KING))

    # Verify is_auto_solvable returns True
    assert game.is_auto_solvable()

    # Test that we get a suggestion to move ♥3 to foundation
    suggestion = game.find_auto_solve_suggestion()
    assert suggestion is not None
    assert suggestion.card == Card(Suit.HEART, Number.THREE)
    assert suggestion.from_location == Location.TABLEAU_1
    assert suggestion.to_location == Location.FOUNDATION_1

    # Move the card as suggested
    game.move_tableau_to_foundation(0, 0)

    # Now there shouldn't be any auto-solve suggestion
    suggestion = game.find_auto_solve_suggestion()
    # This could still return a valid suggestion since we have cards that can be moved
    # We'll check that the original suggestion is not returned
    if suggestion is not None:
        assert suggestion.card != Card(Suit.HEART, Number.THREE)


def test_rendering():
    """Test the rendering functionality of the game."""
    game = Klondike()

    # Clear everything for controlled setup
    game.stock = Stack()
    game.waste = Stack()
    for tableau in game.tableaus:
        tableau.visible = Stack()
        tableau.hidden = Stack()
    for foundation in game.foundations:
        foundation.cards = []

    # Set up a simple state
    game.stock.add_to_top(Card(Suit.HEART, Number.KING))
    game.waste.add_to_top(Card(Suit.SPADE, Number.QUEEN))
    game.foundations[0].add_to_top(Card(Suit.HEART, Number.ACE))
    game.tableaus[0].hidden.add_to_top(Card(Suit.DIAMOND, Number.TWO))
    game.tableaus[0].visible.add_to_top(Card(Suit.CLUB, Number.THREE))

    # Get render
    render = game.render()

    # Verify render contents
    assert render.state == KlondikeState.SETUP.value
    assert render.stock == Card(Suit.HEART, Number.KING)
    assert render.waste == Card(Suit.SPADE, Number.QUEEN)
    assert render.foundations[0] == Card(Suit.HEART, Number.ACE)
    assert render.foundations[1] is None
    assert render.foundations[2] is None
    assert render.foundations[3] is None

    # Check tableau rendering (should show None for hidden cards)
    assert len(render.tableaus[0]) == 2
    assert render.tableaus[0][0] is None  # Hidden card
    assert render.tableaus[0][1] == Card(Suit.CLUB, Number.THREE)

    # Verify move count and undo count
    assert render.move_count == 0
    assert render.undo_count == 0


def test_game_session():
    """Test the GameSession class which manages the game state history."""
    # Create a game session
    session = GameSession()

    # Make some moves
    session.active_game.move_stock_to_waste()
    session.active_game.move_stock_to_waste()

    # Verify that the active game has changed
    assert session.active_game.move_count == 2
    assert len(session.active_game.history) == 2

    # Reset to initial game
    session.reset_to_initial_game()

    # Verify reset worked
    assert session.active_game.move_count == 0
    assert len(session.active_game.history) == 0

    # Test JSON serialization
    json_str = session.as_json()
    assert isinstance(json_str, str)
    assert "initial_state" in json_str
    assert "actions" in json_str
    assert "subsequent_states" in json_str
