#[derive(Eq, Hash, Clone, Copy, PartialEq)]
pub enum Color {
    RED,
    BLACK,
}

#[derive(Eq, Hash, Clone, Copy, PartialEq, Debug)]
pub enum Suit {
    HEART,
    DIAMOND,
    CLUB,
    SPADE,
}

#[derive(Eq, Hash, Clone, Copy, PartialEq, Debug)]
pub enum Number {
    ACE,
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
    JACK,
    QUEEN,
    KING,
}

#[derive(Eq, Hash, Clone, Copy, PartialEq, Debug)]
pub struct Card {
    pub(crate) number: Number,
    pub(crate) suit: Suit,
}

pub struct Deck {}

impl Suit {
    pub fn symbol(&self) -> &'static str {
        match *self {
            Suit::HEART => "♥",
            Suit::DIAMOND => "♦",
            Suit::CLUB => "♣",
            Suit::SPADE => "♠",
        }
    }
    pub fn color(&self) -> Color {
        match *self {
            Suit::HEART => Color::RED,
            Suit::DIAMOND => Color::RED,
            Suit::CLUB => Color::BLACK,
            Suit::SPADE => Color::BLACK,
        }
    }
    pub fn index(&self) -> u8 {
        match *self {
            Suit::HEART => 0,
            Suit::DIAMOND => 1,
            Suit::CLUB => 2,
            Suit::SPADE => 3,
        }
    }

    fn all() -> [Suit; 4] {
        return [Suit::HEART, Suit::DIAMOND, Suit::CLUB, Suit::SPADE];
    }
}

impl Number {
    pub fn symbol(&self) -> &'static str {
        match *self {
            Number::ACE => "A",
            Number::TWO => "2",
            Number::THREE => "3",
            Number::FOUR => "4",
            Number::FIVE => "5",
            Number::SIX => "6",
            Number::SEVEN => "7",
            Number::EIGHT => "8",
            Number::NINE => "9",
            Number::TEN => "10",
            Number::JACK => "J",
            Number::QUEEN => "Q",
            Number::KING => "K",
        }
    }
    pub fn number(&self) -> u8 {
        match *self {
            Number::ACE => 1,
            Number::TWO => 2,
            Number::THREE => 3,
            Number::FOUR => 4,
            Number::FIVE => 5,
            Number::SIX => 6,
            Number::SEVEN => 7,
            Number::EIGHT => 8,
            Number::NINE => 9,
            Number::TEN => 10,
            Number::JACK => 11,
            Number::QUEEN => 12,
            Number::KING => 13,
        }
    }
    pub fn from(number: u8) -> Number {
        match number {
            1 => Number::ACE,
            2 => Number::TWO,
            3 => Number::THREE,
            4 => Number::FOUR,
            5 => Number::FIVE,
            6 => Number::SIX,
            7 => Number::SEVEN,
            8 => Number::EIGHT,
            9 => Number::NINE,
            10 => Number::TEN,
            11 => Number::JACK,
            12 => Number::QUEEN,
            13 => Number::KING,
            _ => {
                panic!("invalid card number")
            }
        }
    }
    pub fn all() -> [Number; 13] {
        return [
            Number::ACE,
            Number::TWO,
            Number::THREE,
            Number::FOUR,
            Number::FIVE,
            Number::SIX,
            Number::SEVEN,
            Number::EIGHT,
            Number::NINE,
            Number::TEN,
            Number::JACK,
            Number::QUEEN,
            Number::KING,
        ];
    }
}

impl Deck {
    pub fn standard() -> Vec<Card> {
        let mut vector = vec![];
        for s in Suit::all() {
            for n in 1..14 as u8 {
                vector.push(Card {
                    number: Number::from(n),
                    suit: s,
                })
            }
        }
        return vector;
    }
}
