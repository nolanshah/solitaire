use rand::prelude::SliceRandom;
use std::collections::VecDeque;

use super::card::Card;

#[derive(Clone)]
pub struct Stack {
    _initial: VecDeque<Card>,
    _cards: VecDeque<Card>,
}

impl Stack {
    pub fn new() -> Stack {
        return Self {
            _initial: VecDeque::new(),
            _cards: VecDeque::new(),
        };
    }

    pub fn from(initial: VecDeque<Card>) -> Stack {
        return Stack {
            _initial: initial.clone(),
            _cards: initial.clone(),
        };
    }

    pub fn shuffle(&mut self) {
        let ctg_cards = self._cards.make_contiguous();
        ctg_cards.shuffle(&mut rand::rng());
    }

    pub fn sort(&mut self) {
        let ctg_cards = self._cards.make_contiguous();
        ctg_cards.sort_by_key(|x| x.number.number());
        ctg_cards.sort_by_key(|x| x.suit.index());
    }

    pub fn get_from_top(&mut self) -> Option<Card> {
        self._cards.pop_front()
    }

    pub fn add_to_top(&mut self, to_add: Card) {
        self._cards.push_front(to_add);
    }

    pub fn add_many_to_top(&mut self, mut to_add: VecDeque<Card>) {
        for _ in 0..to_add.len() {
            self._cards.push_front(to_add.pop_back().unwrap());
        }
    }

    pub fn get_all(&mut self) -> VecDeque<Card> {
        let to_ret = self._cards.clone();
        self._cards = VecDeque::new();
        return to_ret;
    }

    pub fn get_from_bottom(&mut self) -> Option<Card> {
        self._cards.pop_back()
    }

    pub fn add_to_bottom(&mut self, to_add: Card) {
        self._cards.push_back(to_add);
    }

    pub fn add_many_to_bottom(&mut self, mut to_add: VecDeque<Card>) {
        for _ in 0..to_add.len() {
            self._cards.push_back(to_add.pop_front().unwrap());
        }
    }

    pub fn rotate_top_to_bottom(&mut self) {
        if self._cards.len() == 0 {
            return;
        }
        self._cards.rotate_left(1);
    }

    pub fn rotate_bottom_to_top(&mut self) {
        if self._cards.len() == 0 {
            return;
        }
        self._cards.rotate_right(1);
    }

    pub fn inspect(&mut self) -> VecDeque<Card> {
        return self._cards.clone();
    }

    pub fn inspect_top(&mut self) -> Option<&Card> {
        return self._cards.get(0);
    }

    pub fn inspect_bottom(&mut self) -> Option<&Card> {
        return self._cards.get(self._cards.len() - 1);
    }

    pub fn reset(&mut self) {
        self._cards = self._initial.clone();
    }

    pub fn size(&self) -> u16 {
        self._cards.len() as u16
    }

    pub fn reverse(&mut self) {
        let ctg_cards = self._cards.make_contiguous();
        ctg_cards.reverse()
    }
}

#[derive(Clone)]
pub struct StackH {
    _visible: Stack,
    _hidden: Stack,
}

impl StackH {
    pub fn new() -> StackH {
        return Self {
            _visible: Stack::new(),
            _hidden: Stack::new(),
        };
    }

    pub fn reset(&mut self) {
        self._visible.reset();
        self._hidden.reset();
    }

    pub fn add_to_top(&mut self, to_add: Card, hide: bool) {
        if hide {
            self._hidden.add_to_top(to_add);
        } else {
            self._visible.add_to_top(to_add);
        }
    }

    pub fn add_many_to_top(&mut self, to_add: VecDeque<Card>, hide: bool) {
        if hide {
            self._hidden.add_many_to_top(to_add);
        } else {
            self._visible.add_many_to_top(to_add);
        }
    }

    pub fn inspect(&mut self, with_hidden: bool) -> VecDeque<Card> {
        let mut ret = VecDeque::new();
        ret.append(&mut self._visible.inspect());
        if with_hidden {
            ret.append(&mut self._hidden.inspect());
        }
        return ret;
    }

    pub fn inspect_with_hidden(&mut self) -> Vec<(Card, bool)> {
        let mut ret = Vec::new();
        for card in self._visible.inspect() {
            ret.push((card, false))
        }
        for card in self._hidden.inspect() {
            ret.push((card, true))
        }
        return ret;
    }

    pub fn inspect_top(&mut self) -> Option<&Card> {
        self._visible.inspect_top()
    }

    pub fn get_from_top(&mut self) -> Option<(Card, bool)> {
        let card = self._visible.get_from_top();

        let mut did_move = false;
        // flip card from hidden to visible when no more visible
        if self._visible.size() == 0 && self._hidden.size() != 0 {
            self._visible
                .add_to_bottom(self._hidden.get_from_top().unwrap());
            did_move = true;
        }

        return if card.is_none() {
            None
        } else {
            Some((card.unwrap(), did_move))
        };
    }

    pub fn move_hidden_top_to_visible(&mut self) {
        self._hidden
            .add_to_top(self._visible.get_from_bottom().unwrap())
    }

    pub fn size(&self) -> u16 {
        return self._hidden.size() + self._visible.size();
    }

    pub fn size_hidden(&mut self) -> u16 {
        self._hidden.size()
    }

    pub fn size_visible(&mut self) -> u16 {
        self._visible.size()
    }
}
