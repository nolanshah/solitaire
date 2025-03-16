use rand::Rng;
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

mod card;
mod klondike;
mod stack;

use klondike::{Action, Game};

/// Sample an action using weighted probabilities
fn sample_action(valid_actions: &Vec<Action>, params: &[u32]) -> Action {
    if valid_actions.len() == 0 {
        panic!("No valid actions");
    }

    if valid_actions.len() == 1 {
        return valid_actions[0].clone();
    }

    let mut weight_map: HashMap<usize, u32> = HashMap::new();

    for (idx, action) in valid_actions.iter().enumerate() {
        match action {
            Action::MoveStockToWaste => {
                weight_map.insert(idx, params[0]);
            }
            Action::CycleWasteToStock => {
                weight_map.insert(idx, params[1]);
            }
            Action::MoveWasteToFoundation(_) => {
                weight_map.insert(idx, params[2]);
            }
            Action::MoveWasteToTableau(_) => {
                weight_map.insert(idx, params[3]);
            }
            Action::MoveTableauToFoundation(_, _) => {
                weight_map.insert(idx, params[4]);
            }
            Action::MoveFoundationToTableau(_, _) => {
                weight_map.insert(idx, params[5]);
            }
            Action::MoveTableauToTableau(_, _, _) => {
                weight_map.insert(idx, params[6]);
            }
        }
    }

    let total: u32 = weight_map.values().sum();
    let mut rng = rand::rng();
    let r = rng.random_range(0..total);

    let mut upto = 0;
    for (item_idx, weight) in weight_map.iter() {
        if upto + weight >= r {
            return valid_actions[*item_idx].clone();
        }
        upto += weight;
    }

    // Fallback to first action if something went wrong
    valid_actions[0].clone()
}

/// Apply an action to the game
fn apply_action(game: &mut Game, action: &Action) -> Result<(), &'static str> {
    match action {
        Action::MoveStockToWaste => {
            game.move_stock_to_waste(false)?;
        }
        Action::CycleWasteToStock => {
            game.cycle_waste_to_stock();
        }
        Action::MoveWasteToFoundation(dst_i) => {
            game.move_waste_to_solve(*dst_i)?;
        }
        Action::MoveWasteToTableau(dst_i) => {
            game.move_waste_to_pile(*dst_i)?;
        }
        Action::MoveTableauToFoundation(src_i, dst_i) => {
            game.move_pile_to_solve(*src_i, *dst_i)?;
        }
        Action::MoveFoundationToTableau(src_i, dst_i) => {
            game.move_solve_to_pile(*src_i, *dst_i)?;
        }
        Action::MoveTableauToTableau(src_i, src_card_idx, dst_i) => {
            game.move_pile_to_pile(*src_i, *src_card_idx, *dst_i)?;
        }
    }

    Ok(())
}

/// Play a single game of Klondike solitaire using the greedy heuristic agent
fn play_game(max_steps: u32, verbose: bool, print_interval: u32) -> (bool, u32) {
    let mut game = Game::new();
    game.reset();
    let mut steps = 0;

    // Default action weights
    let params = [5, 40, 20, 10, 20, 1, 5];

    while steps < max_steps {
        // Check if game is won
        if game.is_done() {
            if verbose {
                println!("Game won in {} steps!", steps);
            }
            return (true, steps);
        }

        // Get valid actions
        let valid_actions = game.get_all_legal_actions();
        if valid_actions.is_empty() {
            if verbose {
                println!("No more valid actions after {} steps. Game lost.", steps);
            }
            return (false, steps);
        }

        // Choose an action using the heuristic
        let action = sample_action(&valid_actions, &params);

        // Apply the action
        match apply_action(&mut game, &action) {
            Ok(_) => {}
            Err(e) => {
                if verbose {
                    println!("Error applying action: {}", e);
                }
                return (false, steps);
            }
        }

        steps += 1;

        // Print game state at specified intervals
        if print_interval > 0 && steps % print_interval == 0 {
            println!("\n=== Game state at step {} ===", steps);

            // Calculate foundations filled
            let foundation_cards: u16 = game.solves().iter().map(|solve| solve.size()).sum();

            println!("Foundations filled: {}/52", foundation_cards);
            println!("Stock: {} cards remaining", game.stock().size());
            println!("Waste: {} cards", game.waste().size());

            // Print foundations
            print!("Foundations: ");
            let solves_clone = game.solves().clone();
            for (i, _) in solves_clone.iter().enumerate() {
                let mut foundation = solves_clone[i].clone();
                if let Some(card) = foundation.inspect_top() {
                    print!(
                        "F{}: {}{} ",
                        i + 1,
                        card.number.symbol(),
                        card.suit.symbol()
                    );
                } else {
                    print!("F{}: empty ", i + 1);
                }
            }
            println!();

            // Print tableaus
            println!("Tableaus:");
            let piles_clone = game.piles().clone();
            for (i, _) in piles_clone.iter().enumerate() {
                let mut tableau = piles_clone[i].clone();
                let _visible_count = tableau.size_visible();
                let hidden_count = tableau.size_hidden();

                print!("  T{}: {} hidden, [", i + 1, hidden_count);

                let visible_cards = tableau.inspect(false);
                for (j, card) in visible_cards.iter().enumerate() {
                    print!("{}{}", card.number.symbol(), card.suit.symbol());
                    if j < visible_cards.len() - 1 {
                        print!(", ");
                    }
                }
                println!("]");
            }

            // Print last move
            println!("Last move: {:?}", action);
            println!("{}", "=".repeat(40));
        }

        if verbose && steps % 100 == 0 {
            let foundation_cards: u16 = game.solves().iter().map(|solve| solve.size()).sum();
            println!(
                "Step {}, foundations filled: {}/52",
                steps, foundation_cards
            );
        }
    }

    if verbose {
        println!("Reached maximum steps ({}). Game lost.", max_steps);
    }

    (false, steps)
}

/// Play multiple games and report statistics
fn play_multiple_games(num_games: u32, max_steps: u32, print_interval: u32) {
    let mut wins = 0;
    let mut total_steps = 0;
    let mut step_counts = Vec::new();

    let start_time = Instant::now();

    for i in 0..num_games {
        print!("Playing game {}/{} ...\r", i + 1, num_games);
        io::stdout().flush().unwrap();

        let (win, steps) = play_game(max_steps, false, print_interval);
        if win {
            wins += 1;
        }
        total_steps += steps;
        step_counts.push(steps);
    }

    let duration = start_time.elapsed();
    let win_rate = (wins as f64 / num_games as f64) * 100.0;
    let avg_steps = total_steps as f64 / num_games as f64;

    println!("\nResults from {} games:", num_games);
    println!("Win rate: {:.2}% ({}/{})", win_rate, wins, num_games);
    println!("Average steps per game: {:.2}", avg_steps);
    println!(
        "Time taken: {:.2} seconds ({:.2} seconds per game)",
        duration.as_secs_f64(),
        duration.as_secs_f64() / num_games as f64
    );
}

fn main() {
    // Play a single game of Klondike solitaire using the greedy heuristic agent
    // let (win, steps) = play_game(1000, true, 100);
    // println!(
    //     "Game {} after {} steps",
    //     if win { "won" } else { "lost" },
    //     steps
    // );

    // Uncomment to play multiple games and gather statistics
    play_multiple_games(100, 1000, 0);
}
