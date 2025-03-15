import random
import copy
import json
from enum import Enum
from dataclasses import dataclass
from typing import override, Any


class Color(str, Enum):
    RED = "RED"
    BLACK = "BLACK"

    def __int__(self) -> int:
        return {
            Color.RED: 0,
            Color.BLACK: 1,
        }[self]

    def one_hot(self) -> list[int]:
        return [1 if i == int(self) else 0 for i in range(len(Color))]

    @staticmethod
    def one_hot_zeros() -> list[int]:
        return [0 for _ in range(len(Color))]


class Suit(str, Enum):
    HEART = "HEART"
    DIAMOND = "DIAMOND"
    CLUB = "CLUB"
    SPADE = "SPADE"

    @staticmethod
    def index_map():
        return {
            0: Suit.HEART,
            1: Suit.DIAMOND,
            2: Suit.CLUB,
            3: Suit.SPADE,
        }

    @property
    def color(self):
        if self == Suit.HEART or self == Suit.DIAMOND:
            return Color.RED
        elif self == Suit.CLUB or self == Suit.SPADE:
            return Color.BLACK
        else:
            msg = f"Suit {self} has no color"
            raise ValueError(msg)

    @override
    def __str__(self) -> str:
        return {
            Suit.HEART: "♥",
            Suit.DIAMOND: "♦",
            Suit.CLUB: "♣",
            Suit.SPADE: "♠",
        }[self]

    def __int__(self) -> int:
        for idx, sut in Suit.index_map().items():
            if sut == self:
                return idx

        msg = f"Suit {self} not found in index map"
        raise ValueError(msg)

    def one_hot(self) -> list[int]:
        return [1 if i == int(self) else 0 for i in range(len(Suit))]

    @staticmethod
    def one_hot_zeros() -> list[int]:
        return [0 for _ in range(len(Suit))]


class Number(str, Enum):
    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"

    @property
    def int_repr(self) -> int:
        return {
            Number.ACE: 1,
            Number.TWO: 2,
            Number.THREE: 3,
            Number.FOUR: 4,
            Number.FIVE: 5,
            Number.SIX: 6,
            Number.SEVEN: 7,
            Number.EIGHT: 8,
            Number.NINE: 9,
            Number.TEN: 10,
            Number.JACK: 11,
            Number.QUEEN: 12,
            Number.KING: 13,
        }[self]

    @staticmethod
    def number_map():
        return {item.int_repr: item for item in Number}

    @staticmethod
    def repr_map():
        return {item.value: item for item in Number}

    @override
    def __str__(self) -> str:
        return self.value

    @override
    def __repr__(self) -> str:
        return self.value

    def __int__(self) -> int:
        return int(self.int_repr)

    def one_hot(self) -> list[int]:
        return [1 if i == int(self) else 0 for i in range(len(Number))]

    @staticmethod
    def one_hot_zeros() -> list[int]:
        return [0 for _ in range(len(Number))]


@dataclass
class Card:
    suit: Suit
    number: Number

    def as_jsonable_dict(self) -> dict:
        return {
            "suit": self.suit.value,
            "rank": self.number.value,
            "color": self.suit.color.value,
        }

    @override
    def __str__(self) -> str:
        return f"{self.number} {self.suit}"

    @override
    def __repr__(self) -> str:
        return f"{self.number} {self.suit}"

    @override
    def __eq__(self, other: Any) -> bool:  # pyright: ignore [reportAny]
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.number == other.number

    @override
    def __hash__(self) -> int:
        return hash((self.suit, self.number))


type HidableCard = Card | None


class Stack:
    def __init__(self, initial_cards: list[Card] | None = None):
        self.initial_cards = initial_cards or []
        self.cards = self.initial_cards.copy()

    def as_jsonable_dict(self) -> dict:
        return {
            "cards": [card.as_jsonable_dict() for card in self.cards],
        }

    @override
    def __str__(self) -> str:
        return f"Stack: {self.cards}"

    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def sort(self) -> None:
        self.cards.sort(key=lambda card: repr(card))

    def add_to_top(self, card: Card) -> None:
        self.cards.append(card)

    def add_to_bottom(self, card: Card) -> None:
        self.cards.insert(0, card)

    def add_multiple_to_top(self, cards: list[Card]) -> None:
        self.cards.extend(cards)

    def add_multiple_to_bottom(self, cards: list[Card]) -> None:
        self.cards = cards + self.cards

    def inspect_top(self) -> HidableCard:
        if len(self.cards) == 0:
            return None
        return self.cards[-1]

    def inspect_bottom(self) -> HidableCard:
        if len(self.cards) == 0:
            return None
        return self.cards[0]

    def inspect_all(self) -> list[Card]:
        return self.cards.copy()

    def get_from_top(self) -> HidableCard:
        if len(self.cards) == 0:
            return None
        return self.cards.pop()

    def get_from_bottom(self) -> HidableCard:
        if len(self.cards) == 0:
            return None
        return self.cards.pop(0)

    def get_all(self) -> list[Card]:
        cards = self.cards
        self.cards = []
        return cards

    def rotate_top_to_bottom(self) -> None:
        if len(self.cards) == 0:
            return
        card = self.get_from_top()
        assert card is not None  # noqa: S101
        self.add_to_bottom(card)

    def rotate_bottom_to_top(self) -> None:
        if len(self.cards) == 0:
            return
        card = self.get_from_bottom()
        assert card is not None  # noqa: S101
        self.add_to_top(card)

    def reverse(self) -> None:
        self.cards.reverse()

    def reset(self) -> None:
        self.cards = self.initial_cards.copy()

    def __len__(self) -> int:
        return len(self.cards)


class StackWithHidden:
    class HidableCardInternal:
        def __init__(self, card: Card, *, hidden: bool = False):
            self.card = card
            self.hidden = hidden

        @override
        def __str__(self) -> str:
            return f"{self.card} {'HIDDEN' if self.hidden else ''}"

    def as_jsonable_dict(self) -> dict:
        return {
            "hidden": self.hidden.as_jsonable_dict(),
            "visible": self.visible.as_jsonable_dict(),
        }

    def __init__(self):
        self.hidden = Stack()
        self.visible = Stack()

    def reset(self) -> None:
        self.hidden.reset()
        self.visible.reset()

    @override
    def __str__(self) -> str:
        return f"StackWithHidden: {self.hidden} {self.visible}"

    def add_to_top(self, card: Card, *, hide: bool = False) -> None:
        if hide:
            self.hidden.add_to_top(card)
        else:
            self.visible.add_to_top(card)

    def add_multiple_to_top(self, cards: list[Card], *, hide: bool = False) -> None:
        if hide:
            self.hidden.add_multiple_to_top(cards)
        else:
            self.visible.add_multiple_to_top(cards)

    def inspect_top(self) -> HidableCard:
        return self.visible.inspect_top()

    def inspect_all(self) -> list[HidableCard]:
        cards: list[HidableCard] = [None for _ in range(len(self.hidden))]
        cards.extend(self.visible.inspect_all())
        return cards

    def inspect_all_with_hidden(self) -> list[HidableCardInternal]:
        cards = [self.HidableCardInternal(card, hidden=True) for card in self.hidden.inspect_all()]
        cards.extend([self.HidableCardInternal(card, hidden=False) for card in self.visible.inspect_all()])
        return cards

    def get_from_top(self) -> tuple[HidableCard, bool]:
        assert not (len(self.visible) == 0 and len(self.hidden) > 0)  # noqa: S101
        card_from_top = self.visible.get_from_top()
        did_move = False
        if len(self.visible) == 0 and len(self.hidden) > 0:
            from_top = self.hidden.get_from_top()
            assert from_top is not None  # noqa: S101
            self.visible.add_to_bottom(from_top)
            did_move = True

        return card_from_top, did_move

    def hide_bottom_visible(self) -> None:
        card = self.visible.get_from_bottom()
        if card is None:
            return
        self.hidden.add_to_top(card)

    def __len__(self) -> int:
        return self.visible_len() + self.hidden_len()

    def visible_len(self) -> int:
        return len(self.visible)

    def hidden_len(self) -> int:
        return len(self.hidden)


class Location(str, Enum):
    STOCK = "STOC"
    WASTE = "WAST"
    FOUNDATION_1 = "FOUN_1"  # SOLUTION PILE
    FOUNDATION_2 = "FOUN_2"
    FOUNDATION_3 = "FOUN_3"
    FOUNDATION_4 = "FOUN_4"
    TABLEAU_1 = "TABL_1"  # PLAYING PILE
    TABLEAU_2 = "TABL_2"
    TABLEAU_3 = "TABL_3"
    TABLEAU_4 = "TABL_4"
    TABLEAU_5 = "TABL_5"
    TABLEAU_6 = "TABL_6"
    TABLEAU_7 = "TABL_7"

    @staticmethod
    def tableaus():
        return [
            Location.TABLEAU_1,
            Location.TABLEAU_2,
            Location.TABLEAU_3,
            Location.TABLEAU_4,
            Location.TABLEAU_5,
            Location.TABLEAU_6,
            Location.TABLEAU_7,
        ]

    @staticmethod
    def foundations():
        return [
            Location.FOUNDATION_1,
            Location.FOUNDATION_2,
            Location.FOUNDATION_3,
            Location.FOUNDATION_4,
        ]

    def one_hot(self) -> list[int]:
        return [1 if self == loc else 0 for loc in Location]

    @staticmethod
    def from_index(index: int) -> "Location":
        for pos, loc in enumerate(Location):
            if index == pos:
                return loc
        msg = f"Invalid index {index}"
        raise ValueError(msg)

    @staticmethod
    def to_index(loc: "Location") -> int:
        for pos, location in enumerate(Location):
            if loc == location:
                return pos
        msg = f"Invalid location {loc}"
        raise ValueError(msg)


class KlondikeState(str, Enum):
    SETUP = "SETUP"
    PLAYING = "PLAYING"
    WON = "WON"


@dataclass(kw_only=True, unsafe_hash=True, eq=True)
class Render:
    state: str
    stock: HidableCard
    waste: HidableCard
    foundations: list[HidableCard]  # top card on each foundation
    tableaus: list[list[HidableCard]]  # all cards on each tableau with null for hidden cards
    move_count: int
    undo_count: int

    def asdict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "stock": self.stock,
            "waste": self.waste,
            "foundations": self.foundations,
            "tableaus": self.tableaus,
            "move_count": self.move_count,
            "undo_count": self.undo_count,
        }


@dataclass(unsafe_hash=True, eq=True)
class HistoryAction:
    from_location: Location
    to_location: Location
    cards: list[Card] | None
    hidden_to_visible: bool = False

    @override
    def __str__(self) -> str:
        return f"{self.from_location.value} -> {self.to_location.value}"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def as_jsonable_dict(self) -> dict:
        return {
            "from_location": self.from_location.value,
            "to_location": self.to_location.value,
            "cards": [card.as_jsonable_dict() for card in self.cards] if self.cards is not None else None,
            "hidden_to_visible": self.hidden_to_visible,
        }


@dataclass(unsafe_hash=True, eq=True)
class Action:
    from_location: Location
    to_location: Location
    change_index: int = 0

    @override
    def __str__(self) -> str:
        return f"{self.from_location.value} ({self.change_index}) -> {self.to_location.value}"

    @override
    def __repr__(self) -> str:
        return self.__str__()


@dataclass(unsafe_hash=True, eq=True)
class AutoSolveSuggestion:
    from_location: Location
    to_location: Location
    card: Card

    @override
    def __str__(self) -> str:
        return f"{self.card} : {self.from_location.value} -> {self.to_location.value}"

    @override
    def __repr__(self) -> str:
        return self.__str__()


class Klondike:
    def __init__(self) -> None:
        self.deck = [Card(suit, number) for suit in Suit for number in Number]
        self.foundations = [
            Stack(),
            Stack(),
            Stack(),
            Stack(),
        ]
        self.tableaus = [
            StackWithHidden(),
            StackWithHidden(),
            StackWithHidden(),
            StackWithHidden(),
            StackWithHidden(),
            StackWithHidden(),
            StackWithHidden(),
        ]
        self.stock = Stack()
        self.waste = Stack()

        self.history: list[HistoryAction] = []
        self.state = KlondikeState.SETUP

        self.move_count = 0
        self.undo_count = 0

    def reset(self):
        for foundation in self.foundations:
            foundation.reset()
        for tableau in self.tableaus:
            tableau.reset()
        self.stock.reset()
        self.waste.reset()
        self.history = []
        self.move_count = 0
        self.undo_count = 0
        self.state = KlondikeState.SETUP

        hand = Stack(self.deck.copy())
        hand.shuffle()
        for idx, tableau in enumerate(self.tableaus):
            for k in range(idx + 1):
                from_top = hand.get_from_top()
                assert from_top is not None  # noqa: S101
                tableau.add_to_top(from_top, hide=k != idx)
        self.stock.add_multiple_to_top(hand.get_all())

    def move_stock_to_waste(self):
        self.state = KlondikeState.PLAYING

        card = self.stock.get_from_top()
        if card is None:
            return
        self.waste.add_to_top(card)
        self.history.append(HistoryAction(Location.STOCK, Location.WASTE, [card]))
        self.move_count += 1

    def cycle_waste_to_stock(self):
        self.state = KlondikeState.PLAYING

        self.waste.reverse()
        self.stock.add_multiple_to_top(self.waste.get_all())
        self.history.append(HistoryAction(Location.WASTE, Location.STOCK, None))
        self.move_count += 1

    def is_card_placable_on_foundation(self, card: Card, foundation: Stack) -> bool:
        if len(foundation) == 0:
            return card.number == Number.ACE
        top_card = foundation.inspect_top()
        if top_card is None:
            return False
        return top_card.suit == card.suit and int(top_card.number) == int(card.number) - 1

    def is_card_placable_on_tableau(self, card: Card, tableau: StackWithHidden) -> bool:
        if len(tableau) == 0:
            return card.number == Number.KING
        top_card = tableau.inspect_top()
        if top_card is None:
            return False
        return top_card.suit.color != card.suit.color and int(top_card.number) == int(card.number) + 1

    def move_waste_to_foundation(self, foundation_idx: int):
        self.state = KlondikeState.PLAYING

        foundation = self.foundations[foundation_idx]
        card = self.waste.inspect_top()
        if card is None or not self.is_card_placable_on_foundation(card, foundation):
            return
        _ = self.waste.get_from_top()
        assert card is not None  # noqa: S101
        foundation.add_to_top(card)
        self.history.append(HistoryAction(Location.WASTE, Location.foundations()[foundation_idx], [card]))
        self.move_count += 1

    def move_tableau_to_foundation(self, tableau_idx: int, foundation_idx: int):
        self.state = KlondikeState.PLAYING

        tableau = self.tableaus[tableau_idx]
        foundation = self.foundations[foundation_idx]
        card = tableau.inspect_top()
        if card is None or not self.is_card_placable_on_foundation(card, foundation):
            return
        card, hidden_to_visible = tableau.get_from_top()
        assert card is not None  # noqa: S101
        foundation.add_to_top(card)
        self.history.append(
            HistoryAction(
                Location.tableaus()[tableau_idx],
                Location.foundations()[foundation_idx],
                [card],
                hidden_to_visible=hidden_to_visible,
            ),
        )
        self.move_count += 1

    def move_waste_to_tableau(self, tableau_idx: int):
        self.state = KlondikeState.PLAYING

        tableau = self.tableaus[tableau_idx]
        card = self.waste.inspect_top()
        if card is None or not self.is_card_placable_on_tableau(card, tableau):
            return
        _ = self.waste.get_from_top()
        tableau.add_to_top(card)
        self.history.append(HistoryAction(Location.WASTE, Location.tableaus()[tableau_idx], [card]))
        self.move_count += 1

    def move_tableau_to_tableau(self, from_tableau_idx: int, from_tableau_card_idx: int, to_tableau_idx: int):
        self.state = KlondikeState.PLAYING

        from_tableau = self.tableaus[from_tableau_idx]
        to_tableau = self.tableaus[to_tableau_idx]

        from_tableau_card_from_top = len(from_tableau) - from_tableau_card_idx - 1

        # check if the move is valid
        if from_tableau_card_from_top >= len(from_tableau):
            return
        if from_tableau_card_from_top <= len(from_tableau) - 1:
            card = from_tableau.inspect_all()[from_tableau_card_idx]
            if card is None or not self.is_card_placable_on_tableau(card, to_tableau):
                return

        hand = Stack()
        hidden_to_visible = False
        for _ in range(from_tableau_card_from_top + 1):
            ft, hidden_to_visible = from_tableau.get_from_top()
            assert ft is not None  # noqa: S101
            hand.add_to_bottom(ft)
        to_tableau.add_multiple_to_top(hand.get_all())

        self.history.append(
            HistoryAction(
                Location.tableaus()[from_tableau_idx],
                Location.tableaus()[to_tableau_idx],
                hand.inspect_all(),
                hidden_to_visible=hidden_to_visible,
            ),
        )
        self.move_count += 1

    def move_foundation_to_tableau(self, foundation_idx: int, tableau_idx: int):
        self.state = KlondikeState.PLAYING

        foundation = self.foundations[foundation_idx]
        tableau = self.tableaus[tableau_idx]
        card = foundation.inspect_top()
        if card is None or not self.is_card_placable_on_tableau(card, tableau):
            return
        card = foundation.get_from_top()
        assert card is not None  # noqa: S101
        tableau.add_to_top(card)
        self.history.append(
            HistoryAction(
                Location.foundations()[foundation_idx],
                Location.tableaus()[tableau_idx],
                [card],
            ),
        )
        self.move_count += 1

    def is_done(self) -> bool:
        return all(len(foundation) == 13 for foundation in self.foundations)  # noqa: PLR2004

    def is_auto_solvable(self) -> bool:
        return (
            sum([len(tableau) for tableau in self.tableaus]) + sum([len(foundation) for foundation in self.foundations])
        ) == 52  # noqa: PLR2004

    def find_auto_solve_suggestion(self) -> AutoSolveSuggestion | None:
        if not self.is_auto_solvable() or self.is_done():
            return None

        for tableau_idx, tableau in enumerate(self.tableaus):
            card = tableau.inspect_top()
            if card is None:
                continue
            for foundation_idx, foundation in enumerate(self.foundations):
                if self.is_card_placable_on_foundation(card, foundation):
                    return AutoSolveSuggestion(
                        Location.tableaus()[tableau_idx],
                        Location.foundations()[foundation_idx],
                        card,
                    )

        return None

    def undo(self):
        self.state = KlondikeState.PLAYING

        if len(self.history) == 0:
            return

        action = self.history.pop()

        if action.from_location == Location.STOCK and action.to_location == Location.WASTE:
            from_top = self.waste.get_from_top()
            assert from_top is not None  # noqa: S101
            self.stock.add_to_top(from_top)
        elif action.from_location == Location.WASTE and action.to_location == Location.STOCK:
            self.stock.reverse()
            self.waste.add_multiple_to_top(self.stock.get_all())
        elif action.from_location == Location.WASTE and action.to_location in Location.foundations():
            foundation = self.foundations[Location.foundations().index(action.to_location)]
            from_top = foundation.get_from_top()
            assert from_top is not None  # noqa: S101
            self.waste.add_to_top(from_top)
        elif action.from_location == Location.WASTE and action.to_location in Location.tableaus():
            to_tableau = self.tableaus[Location.tableaus().index(action.to_location)]
            from_top, _did_move = to_tableau.get_from_top()
            assert from_top is not None  # noqa: S101
            self.waste.add_to_top(from_top)
        elif action.from_location in Location.tableaus() and action.to_location in Location.foundations():
            tableau = self.tableaus[Location.tableaus().index(action.from_location)]
            foundation = self.foundations[Location.foundations().index(action.to_location)]
            from_top = foundation.get_from_top()
            assert from_top is not None  # noqa: S101
            tableau.add_to_top(from_top)
            if action.hidden_to_visible:
                tableau.hide_bottom_visible()
        elif action.from_location in Location.foundations() and action.to_location in Location.tableaus():
            from_foundation = self.foundations[Location.foundations().index(action.from_location)]
            to_tableau = self.tableaus[Location.tableaus().index(action.to_location)]
            from_top = from_foundation.get_from_top()
            assert from_top is not None  # noqa: S101
            to_tableau.add_to_top(from_top)
        elif action.from_location in Location.tableaus() and action.to_location in Location.tableaus():
            from_tableau = self.tableaus[Location.tableaus().index(action.from_location)]
            to_tableau = self.tableaus[Location.tableaus().index(action.to_location)]
            hand = Stack()
            assert action.cards is not None  # noqa: S101
            for _ in range(len(action.cards)):
                from_top, _ = to_tableau.get_from_top()
                assert from_top is not None  # noqa: S101
                hand.add_to_top(from_top)
            hand.reverse()
            from_tableau.add_multiple_to_top(hand.get_all())
            if action.hidden_to_visible:
                from_tableau.hide_bottom_visible()
        else:
            return

        self.undo_count += 1

    def get_all_legal_actions(self) -> list[Action]:  # noqa: PLR0912
        suggestions: list[Action] = []
        stock_len = len(self.stock)
        waste_len = len(self.waste)
        foundations_len = len(self.foundations)
        tableaus_len = len(self.tableaus)

        if stock_len > 0:
            suggestions.append(Action(Location.STOCK, Location.WASTE))
        if waste_len > 0 and stock_len == 0:
            suggestions.append(Action(Location.WASTE, Location.STOCK))
        if waste_len > 0:
            waste_top = self.waste.inspect_top()
            assert waste_top is not None  # noqa: S101
            for foundation_idx, foundation in enumerate(self.foundations):
                if foundations_len > 0 and self.is_card_placable_on_foundation(waste_top, foundation):
                    suggestions.append(Action(Location.WASTE, Location.foundations()[foundation_idx]))
                for tableau_idx, tableau in enumerate(self.tableaus):
                    if tableaus_len > 0 and self.is_card_placable_on_tableau(waste_top, tableau):
                        suggestions.append(Action(Location.WASTE, Location.tableaus()[tableau_idx]))

        for tableau_idx, tableau in enumerate(self.tableaus):
            if len(tableau) == 0:
                continue
            tableau_top = tableau.inspect_top()
            assert tableau_top is not None  # noqa: S101
            for foundation_idx, foundation in enumerate(self.foundations):
                if foundations_len > 0 and self.is_card_placable_on_foundation(tableau_top, foundation):
                    suggestions.append(
                        Action(
                            Location.tableaus()[tableau_idx],
                            Location.foundations()[foundation_idx],
                        ),
                    )
            for other_tableau_idx, other_tableau in enumerate(self.tableaus):
                if other_tableau_idx == tableau_idx:
                    continue
                for idx, card in enumerate(tableau.inspect_all_with_hidden()):
                    if card.hidden:
                        continue
                    if self.is_card_placable_on_tableau(card.card, other_tableau):
                        suggestions.append(
                            Action(
                                Location.tableaus()[tableau_idx],
                                Location.tableaus()[other_tableau_idx],
                                change_index=idx,
                            ),
                        )

        for foundation_idx, foundation in enumerate(self.foundations):
            if len(foundation) == 0:
                continue
            foundation_top = foundation.inspect_top()
            assert foundation_top is not None  # noqa: S101
            for tableau_idx, tableau in enumerate(self.tableaus):
                if tableaus_len > 0 and self.is_card_placable_on_tableau(foundation_top, tableau):
                    suggestions.append(
                        Action(
                            Location.foundations()[foundation_idx],
                            Location.tableaus()[tableau_idx],
                        ),
                    )

        return suggestions

    def step(self, action: Action) -> None:
        suggestion_equivalent_action = None
        for suggestion in self.get_all_legal_actions():
            if suggestion.from_location == action.from_location and suggestion.to_location == action.to_location:
                suggestion_equivalent_action = suggestion
                break

        if suggestion_equivalent_action is None:
            msg = "Invalid action"
            raise ValueError(msg)

        if action.from_location == Location.STOCK and action.to_location == Location.WASTE:
            self.move_stock_to_waste()
        elif action.from_location == Location.WASTE and action.to_location == Location.STOCK:
            self.cycle_waste_to_stock()
        elif action.from_location == Location.WASTE and action.to_location in Location.foundations():
            self.move_waste_to_foundation(
                Location.foundations().index(action.to_location),
            )
        elif action.from_location == Location.WASTE and action.to_location in Location.tableaus():
            self.move_waste_to_tableau(
                Location.tableaus().index(action.to_location),
            )
        elif action.from_location in Location.tableaus() and action.to_location in Location.foundations():
            self.move_tableau_to_foundation(
                Location.tableaus().index(action.from_location),
                Location.foundations().index(action.to_location),
            )
        elif action.from_location in Location.foundations() and action.to_location in Location.tableaus():
            self.move_foundation_to_tableau(
                Location.foundations().index(action.from_location),
                Location.tableaus().index(action.to_location),
            )
        elif action.from_location in Location.tableaus() and action.to_location in Location.tableaus():
            assert suggestion_equivalent_action is not None  # noqa: S101
            self.move_tableau_to_tableau(
                Location.tableaus().index(action.from_location),
                suggestion_equivalent_action.change_index,
                Location.tableaus().index(action.to_location),
            )

    def render(self) -> Render:
        return Render(
            state=self.state.value,
            stock=self.stock.inspect_top(),
            waste=self.waste.inspect_top(),
            foundations=[foundation.inspect_top() for foundation in self.foundations],
            tableaus=[tableau.inspect_all() for tableau in self.tableaus],
            move_count=self.move_count,
            undo_count=self.undo_count,
        )

    def copy(self) -> "Klondike":
        return copy.deepcopy(self)

    def as_jsonable_dict(self) -> dict:
        return {
            "foundations": [foundation.as_jsonable_dict() for foundation in self.foundations],
            "tableaus": [tableau.as_jsonable_dict() for tableau in self.tableaus],
            "stock": self.stock.as_jsonable_dict(),
            "waste": self.waste.as_jsonable_dict(),
        }

class GameSession:
    def __init__(self):
        self.initial_game = Klondike()
        self.initial_game.reset()
        self.active_game = copy.deepcopy(self.initial_game)

    def reset_to_initial_game(self):
        self.active_game = copy.deepcopy(self.initial_game)

    def as_jsonable_dict(self) -> dict:
        return {
            "initial_state": self.initial_game.as_jsonable_dict(),
            "actions": [
                action.as_jsonable_dict() for action in self.active_game.history
            ],
            "subsequent_states": [
                self.active_game.as_jsonable_dict() for _ in self.active_game.history
            ],
        }

    def as_json(self) -> str:
        return json.dumps(self.as_jsonable_dict())
