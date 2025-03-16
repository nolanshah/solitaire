use super::card::{Card, Deck, Number};
use super::stack::{Stack, StackH};
use std::collections::VecDeque;

#[derive(Eq, Hash, Clone, Copy, PartialEq, Debug)]
pub enum Location {
    PILE1,
    PILE2,
    PILE3,
    PILE4,
    PILE5,
    PILE6,
    PILE7,
    STOCK,
    WASTE,
    SOLVE1,
    SOLVE2,
    SOLVE3,
    SOLVE4,
}

// Define an Action enum similar to the Python implementation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Action {
    MoveStockToWaste,
    CycleWasteToStock,
    MoveWasteToFoundation(u8),
    MoveWasteToTableau(u8),
    MoveTableauToFoundation(u8, u8),
    MoveFoundationToTableau(u8, u8),
    MoveTableauToTableau(u8, u8, u8), // src_pile, src_card_index, dst_pile
}

impl Location {
    pub fn all_piles() -> [Location; 7] {
        return [
            Location::PILE1,
            Location::PILE2,
            Location::PILE3,
            Location::PILE4,
            Location::PILE5,
            Location::PILE6,
            Location::PILE7,
        ];
    }

    pub fn all_solves() -> [Location; 4] {
        return [
            Location::SOLVE1,
            Location::SOLVE2,
            Location::SOLVE3,
            Location::SOLVE4,
        ];
    }

    pub fn all() -> [Location; 13] {
        return [
            Location::PILE1,
            Location::PILE2,
            Location::PILE3,
            Location::PILE4,
            Location::PILE5,
            Location::PILE6,
            Location::PILE7,
            Location::STOCK,
            Location::WASTE,
            Location::SOLVE1,
            Location::SOLVE2,
            Location::SOLVE3,
            Location::SOLVE4,
        ];
    }
}

#[derive(Eq, Hash, Clone, Copy, PartialEq)]
pub enum GameState {
    NULL,
    INITIALIZED,
    INPROGRESS,
    DONE,
}

struct GameAction {
    from: Location,
    to: Location,
    cards: Option<Vec<Card>>,
    hidden_to_visible: Option<bool>,
}

pub struct GameSuggestion {
    from: Location,
    to: Location,
    card: Card,
}

pub(crate) struct Game {
    deck: Vec<Card>,
    piles: [StackH; 7],
    waste: Stack,
    solves: [Stack; 4],
    stock: Stack,

    history: VecDeque<GameAction>,
    state: GameState,

    move_count: u32,
    undo_count: u32,
}

impl Game {
    pub fn new() -> Game {
        return Self {
            deck: Deck::standard(),
            piles: [
                StackH::new(),
                StackH::new(),
                StackH::new(),
                StackH::new(),
                StackH::new(),
                StackH::new(),
                StackH::new(),
            ],
            waste: Stack::new(),
            solves: [Stack::new(), Stack::new(), Stack::new(), Stack::new()],
            stock: Stack::new(),
            history: VecDeque::new(),
            state: GameState::NULL,
            move_count: 0,
            undo_count: 0,
        };
    }

    pub fn reset(&mut self) {
        for solve in self.solves.iter_mut() {
            solve.reset();
        }
        for pile in self.piles.iter_mut() {
            pile.reset();
        }
        self.waste.reset();
        self.stock.reset();
        self.history = VecDeque::new();
        self.state = GameState::NULL;

        let mut hand = Stack::from(self.deck.clone().into());
        hand.shuffle();

        for (i, pile) in self.piles.iter_mut().enumerate() {
            for k in 0..i + 1 {
                let should_hide = i != k; // not the last card
                pile.add_to_top(hand.get_from_top().unwrap(), should_hide)
            }
        }
        self.stock.add_many_to_top(hand.get_all());

        self.state = GameState::INITIALIZED;
        self.move_count = 0;
        self.undo_count = 0;
    }

    pub fn undo(&mut self) {}

    pub fn move_stock_to_waste(&mut self, move_three: bool) -> Result<u8, &'static str> {
        self.state = GameState::INPROGRESS;

        let mut hand = Stack::new();
        let move_count = if move_three { 3 } else { 1 };
        for i in 0..move_count {
            let card = self.stock.get_from_top();
            if card.is_some() {
                hand.add_to_top(card.unwrap());
            } else if i == 0 {
                return Err("Illegal move: stock is empty");
            }
            // ELSE: noop when card is null and we're on the second or third card -- this is acceptable
        }

        let cards = hand.get_all();
        self.waste.add_many_to_top(cards.clone());

        self.history.push_back(GameAction {
            from: Location::STOCK,
            to: Location::WASTE,
            cards: Some(cards.into()),
            hidden_to_visible: None,
        });
        self.move_count += 1;

        return Ok(0);
    }

    pub fn cycle_waste_to_stock(&mut self) {
        self.state = GameState::INPROGRESS;
        self.waste.reverse();
        self.stock.add_many_to_top(self.waste.get_all());
        self.history.push_front(GameAction {
            from: Location::WASTE,
            to: Location::STOCK,
            cards: None,
            hidden_to_visible: None,
        });
        self.move_count += 1;
    }

    #[inline]
    fn is_card_placable_on_pile(card: &Card, pile: &mut StackH) -> bool {
        let dest = pile.inspect_top();
        if dest.is_none() {
            if card.number == Number::KING {
                return true;
            }
        } else {
            let dest = dest.unwrap();
            if card.number.number() == (dest.number.number() - 1)
                && card.suit.color() != dest.suit.color()
            {
                return true;
            }
        }
        return false;
    }

    #[inline]
    fn is_card_placable_on_solve(card: &Card, solve: &mut Stack) -> bool {
        let dest = solve.inspect_top();
        if dest.is_none() {
            if card.number == Number::ACE {
                return true;
            }
        } else {
            let dest = dest.unwrap();
            if card.number.number() == (dest.number.number() + 1) && card.suit == dest.suit {
                return true;
            }
        }
        return false;
    }

    pub fn find_card_placement_location(
        &mut self,
        card: &Card,
        location: Location,
        is_stack: bool,
    ) -> Option<Location> {
        if !Location::all_solves().contains(&location) && !is_stack {
            for (i, loc) in Location::all_solves().iter_mut().enumerate() {
                if Self::is_card_placable_on_solve(card, &mut self.solves[i]) {
                    return Some(*loc);
                }
            }
        }

        for (i, loc) in Location::all_piles().iter_mut().enumerate() {
            if Self::is_card_placable_on_pile(card, &mut self.piles[i]) {
                return Some(*loc);
            }
        }

        return None;
    }

    pub fn move_waste_to_pile(&mut self, dst_pile_index: u8) -> Result<u8, &'static str> {
        self.state = GameState::INPROGRESS;

        let pile: &mut StackH = &mut self.piles[dst_pile_index as usize];

        // check move validity
        let card = self.waste.inspect_top();
        if card.is_none() {
            return Err("illegal move: waste is empty");
        }
        let card = card.unwrap();
        if !Self::is_card_placable_on_pile(card, pile) {
            return Err("illegal move: card not placable on pile");
        }

        // execute move
        let card = self.waste.get_from_top().unwrap();
        pile.add_to_top(card, false);

        self.history.push_back(GameAction {
            from: Location::WASTE,
            to: Location::all_piles()[dst_pile_index as usize],
            cards: Some(Vec::from([card])),
            hidden_to_visible: None,
        });
        self.move_count += 1;

        return Ok(0);
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.solves.iter().map(|solve| solve.size()).sum::<u16>() == 52
    }

    #[inline]
    pub fn is_auto_solvable(&self) -> bool {
        let solves_card_count: u16 = self.solves.iter().map(|solve| solve.size()).sum::<u16>();
        let piles_visible_card_count: u16 = self.piles.iter().map(|pile| pile.size()).sum::<u16>();
        return solves_card_count + piles_visible_card_count == 52;
    }

    pub fn find_auto_solve_card_placement_location(&mut self) -> Option<GameSuggestion> {
        if self.is_auto_solvable() || self.is_done() {
            return None;
        }

        for (src_index, src_pile) in self.piles.iter_mut().enumerate() {
            let candidate_card = src_pile.inspect_top();
            if candidate_card.is_none() {
                continue;
            }
            let candidate_card = candidate_card.unwrap();
            let src_location = Location::all_piles()[src_index];

            for (dst_index, dst_solve) in self.solves.iter_mut().enumerate() {
                if Self::is_card_placable_on_solve(candidate_card, dst_solve) {
                    let dst_location = Location::all_solves()[dst_index];
                    return Some(GameSuggestion {
                        from: src_location,
                        to: dst_location,
                        card: *candidate_card,
                    });
                }
            }
        }

        panic!("auto-solvable, but could not complete")
    }

    pub fn move_waste_to_solve(&mut self, dst_solve_index: u8) -> Result<u8, &'static str> {
        self.state = GameState::INPROGRESS;

        let solve: &mut Stack = &mut self.solves[dst_solve_index as usize];

        // check move validity
        let card = self.waste.inspect_top();
        if card.is_none() {
            return Err("illegal move: waste is empty");
        }
        let card = card.unwrap();
        if !Self::is_card_placable_on_solve(card, solve) {
            return Err("illegal move: card is not placable on solve");
        }

        // execute move
        let card = self.waste.get_from_top().unwrap();
        solve.add_to_top(card);

        self.history.push_back(GameAction {
            from: Location::WASTE,
            to: Location::all_solves()[dst_solve_index as usize],
            cards: Some(Vec::from([card])),
            hidden_to_visible: None,
        });
        self.move_count += 1;

        return Ok(0);
    }

    pub fn move_solve_to_pile(
        &mut self,
        src_solve_index: u8,
        dst_pile_index: u8,
    ) -> Result<u8, &'static str> {
        self.state = GameState::INPROGRESS;

        // check move validity
        let card = {
            let src_solve: &mut Stack = &mut self.solves[src_solve_index as usize];
            let card = src_solve.inspect_top();
            if card.is_none() {
                return Err("illegal move: waste is empty");
            }
            card
        }
        .unwrap();
        {
            let dst_pile: &mut StackH = &mut self.piles[dst_pile_index as usize];
            if !Self::is_card_placable_on_pile(card, dst_pile) {
                return Err("illegal move: card is not placable on solve");
            }
        }

        // execute move
        let card = {
            let src_solve: &mut Stack = &mut self.solves[src_solve_index as usize];
            src_solve.get_from_top().unwrap()
        };
        {
            let dst_pile: &mut StackH = &mut self.piles[dst_pile_index as usize];
            dst_pile.add_to_top(card, false);
        }

        self.history.push_back(GameAction {
            from: Location::all_solves()[src_solve_index as usize],
            to: Location::all_piles()[dst_pile_index as usize],
            cards: Some(Vec::from([card])),
            hidden_to_visible: None,
        });
        self.move_count += 1;

        return Ok(0);
    }

    pub fn move_pile_to_pile(
        &mut self,
        src_pile_index: u8,
        src_pile_card_index: u8,
        dst_pile_index: u8,
    ) -> Result<u8, &'static str> {
        self.state = GameState::INPROGRESS;

        let src_pile_cards = {
            let src_pile: &mut StackH = &mut self.piles[src_pile_index as usize];
            src_pile.inspect(false)
        };
        let card = src_pile_cards.get(src_pile_card_index as usize);

        {
            // check move validity
            let dst_pile: &mut StackH = &mut self.piles[dst_pile_index as usize];
            if card.is_none() {
                return Err("illegal move: card is not visible");
            } else if !Self::is_card_placable_on_pile(card.unwrap(), dst_pile) {
                return Err("illegal move: card is not placable on pile");
            }
        }

        // check stack move validity
        for k in 1..src_pile_card_index as usize {
            let top_card = src_pile_cards.get(k - 1);
            let bot_card = src_pile_cards.get(k);

            if top_card.is_none() || bot_card.is_none() {
                panic!("logic error: top cards should be visible");
            } else if top_card.unwrap().number.number() >= bot_card.unwrap().number.number() {
                return Err("illegal move: card does not compose a movable stack");
            } else if top_card.unwrap().suit.color() == bot_card.unwrap().suit.color() {
                return Err("illegal move: card does not compose a movable stack");
            }
        }

        // execute move
        let mut hand = Stack::new();
        let mut did_make_visible = false;

        {
            let src_pile: &mut StackH = &mut self.piles[src_pile_index as usize];
            for _k in 0..(src_pile_card_index + 1) as usize {
                let (resp, did_move_this) = src_pile.get_from_top().unwrap();
                hand.add_to_bottom(resp);
                did_make_visible = did_move_this;
            }
        }

        let cards = hand.get_all();

        {
            let dst_pile: &mut StackH = &mut self.piles[dst_pile_index as usize];
            dst_pile.add_many_to_top(cards.clone(), false);
        }

        self.history.push_back(GameAction {
            from: Location::all_piles()[src_pile_index as usize],
            to: Location::all_piles()[dst_pile_index as usize],
            cards: Some(cards.clone().into()),
            hidden_to_visible: Some(did_make_visible),
        });

        self.move_count += 1;
        return Ok(0);
    }

    pub fn move_pile_to_solve(
        &mut self,
        src_pile_index: u8,
        dst_solve_index: u8,
    ) -> Result<u8, &'static str> {
        self.state = GameState::INPROGRESS;

        let src_pile: &mut StackH = &mut self.piles[src_pile_index as usize];
        let dst_solve: &mut Stack = &mut self.solves[dst_solve_index as usize];

        // check move validity
        let card = src_pile.inspect_top();
        if card.is_none() {
            return Err("illegal move: pile is empty");
        }
        let card = card.unwrap();
        if !Self::is_card_placable_on_solve(card, dst_solve) {
            return Err("illegal move: card is not placable on solve");
        }

        // execute move
        let (card, _) = src_pile.get_from_top().unwrap();
        dst_solve.add_to_top(card);

        self.history.push_back(GameAction {
            from: Location::all_piles()[src_pile_index as usize],
            to: Location::all_solves()[dst_solve_index as usize],
            cards: Some(Vec::from([card])),
            hidden_to_visible: None,
        });
        self.move_count += 1;

        return Ok(0);
    }

    /// Get all legal actions for the current game state
    pub fn get_all_legal_actions(&mut self) -> Vec<Action> {
        let mut actions = Vec::new();

        // Stock to waste
        if self.stock().size() > 0 {
            actions.push(Action::MoveStockToWaste);
        }

        // Waste to stock (cycle)
        if self.stock().size() == 0 && self.waste().size() > 0 {
            actions.push(Action::CycleWasteToStock);
        }

        // Waste to foundation
        {
            // Create mutable clones to work with
            let mut waste_clone = self.waste.clone();
            let waste_card_opt = waste_clone.inspect_top().cloned();

            if let Some(card) = waste_card_opt {
                for (i, _foundation) in Location::all_solves().iter().enumerate() {
                    let solve_size = self.solves[i].size();
                    
                    if solve_size == 0 && card.number.number() == 1 {
                        // Ace to empty foundation
                        actions.push(Action::MoveWasteToFoundation(i as u8));
                    } else if solve_size > 0 {
                        // Check if card can be placed on foundation
                        let mut solves_clone = self.solves.clone();
                        let mut foundation_clone = &mut solves_clone[i];
                        
                        if let Some(top_card) = foundation_clone.inspect_top() {
                            if card.suit == top_card.suit
                                && card.number.number() == top_card.number.number() + 1
                            {
                                actions.push(Action::MoveWasteToFoundation(i as u8));
                            }
                        }
                    }
                }
            }
        }

        // Waste to tableau
        {
            let mut waste_clone = self.waste.clone();
            let waste_card_opt = waste_clone.inspect_top().cloned();

            if let Some(card) = waste_card_opt {
                for (i, _tableau) in Location::all_piles().iter().enumerate() {
                    // Check if card can be placed on tableau
                    let mut piles_clone = self.piles.clone();
                    let mut pile_clone = &mut piles_clone[i];
                    
                    if Self::is_card_placable_on_pile(&card, pile_clone) {
                        actions.push(Action::MoveWasteToTableau(i as u8));
                    }
                }
            }
        }

        // Tableau to foundation
        {
            let mut piles_clone = self.piles.clone();
            
            for src_i in 0..piles_clone.len() {
                let mut src_pile_clone = &mut piles_clone[src_i];
                
                if let Some(card) = src_pile_clone.inspect_top().cloned() {
                    let mut solves_clone = self.solves.clone();
                    
                    for dst_i in 0..solves_clone.len() {
                        // Check if card can be placed on foundation
                        let mut foundation_clone = &mut solves_clone[dst_i];
                        
                        if Self::is_card_placable_on_solve(&card, foundation_clone) {
                            actions.push(Action::MoveTableauToFoundation(src_i as u8, dst_i as u8));
                        }
                    }
                }
            }
        }

        // Foundation to tableau
        {
            let mut solves_clone = self.solves.clone();
            
            for src_i in 0..solves_clone.len() {
                let mut src_foundation = &mut solves_clone[src_i];
                
                if let Some(card) = src_foundation.inspect_top().cloned() {
                    let mut piles_clone = self.piles.clone();
                    
                    for dst_i in 0..piles_clone.len() {
                        // Check if card can be placed on tableau
                        let mut pile_clone = &mut piles_clone[dst_i];
                        
                        if Self::is_card_placable_on_pile(&card, pile_clone) {
                            actions.push(Action::MoveFoundationToTableau(src_i as u8, dst_i as u8));
                        }
                    }
                }
            }
        }

        // Tableau to tableau
        {
            let mut piles_clone = self.piles.clone();
            
            for src_i in 0..piles_clone.len() {
                let mut src_pile = &mut piles_clone[src_i];
                
                if src_pile.size_visible() == 0 {
                    continue; // Skip empty piles
                }
                
                let visible_cards = src_pile.inspect(false);
                
                for src_card_idx in 0..visible_cards.len() {
                    if let Some(card) = visible_cards.get(src_card_idx).cloned() {
                        let mut piles_clone_inner = self.piles.clone();
                        
                        for dst_i in 0..piles_clone_inner.len() {
                            if src_i == dst_i {
                                continue; // Skip same pile
                            }
                            
                            // Check if card stack can be placed on destination
                            let mut dst_pile = &mut piles_clone_inner[dst_i];
                            
                            if Self::is_card_placable_on_pile(&card, dst_pile) {
                                // Check if stack is valid (alternating colors, descending values)
                                let mut is_valid_stack = true;
                                for k in 1..src_card_idx {
                                    let top_card = visible_cards.get(k - 1);
                                    let bot_card = visible_cards.get(k);
                                    
                                    if top_card.is_none() || bot_card.is_none() {
                                        is_valid_stack = false;
                                        break;
                                    }
                                    
                                    if top_card.unwrap().number.number()
                                        >= bot_card.unwrap().number.number()
                                        || top_card.unwrap().suit.color()
                                            == bot_card.unwrap().suit.color()
                                    {
                                        is_valid_stack = false;
                                        break;
                                    }
                                }
                                
                                if is_valid_stack {
                                    actions.push(Action::MoveTableauToTableau(
                                        src_i as u8,
                                        src_card_idx as u8,
                                        dst_i as u8,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        actions
    }

    pub(crate) fn piles(&self) -> &[StackH; 7] {
        &self.piles
    }

    pub(crate) fn waste(&self) -> &Stack {
        &self.waste
    }

    pub(crate) fn solves(&self) -> &[Stack; 4] {
        &self.solves
    }

    pub(crate) fn stock(&self) -> &Stack {
        &self.stock
    }

    pub(crate) fn state(&self) -> GameState {
        self.state.clone()
    }

    pub(crate) fn move_count(&self) -> u32 {
        self.move_count
    }

    pub(crate) fn undo_count(&self) -> u32 {
        self.undo_count
    }
}
