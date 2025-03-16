use clap::Parser;
use rand::Rng;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

mod card;
mod klondike;
mod stack;

use klondike::{Action, Game};

/// Klondike Solitaire CLI application
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of games to play
    #[arg(short, long, default_value_t = 1)]
    games: u32,

    /// Maximum steps per game before giving up
    #[arg(short, long, default_value_t = 1000)]
    max_steps: u32,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Print game state every N steps (0 = never)
    #[arg(long, default_value_t = 0)]
    print_interval: u32,

    /// Number of threads to use (0 = auto)
    #[arg(short, long, default_value_t = 0)]
    threads: usize,
}

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
fn play_multiple_games(num_games: u32, max_steps: u32, print_interval: u32, thread_count: usize) {
    let wins = Arc::new(Mutex::new(0));
    let total_steps = Arc::new(Mutex::new(0));
    let step_counts = Arc::new(Mutex::new(Vec::new()));
    let games_completed = Arc::new(Mutex::new(0));

    let start_time = Instant::now();
    
    // Determine optimal number of threads if auto mode
    let num_threads = if thread_count == 0 {
        thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
    } else {
        thread_count
    };
    
    // Ensure we don't use more threads than games
    let num_threads = std::cmp::min(num_threads, num_games as usize);
    let games_per_thread = (num_games as usize + num_threads - 1) / num_threads;

    println!("Running games using {} threads", num_threads);

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let start_game = thread_id * games_per_thread;
        let end_game = std::cmp::min((thread_id + 1) * games_per_thread, num_games as usize);

        if start_game >= end_game {
            continue;
        }

        let thread_games = (end_game - start_game) as u32;
        let thread_wins = Arc::clone(&wins);
        let thread_total_steps = Arc::clone(&total_steps);
        let thread_step_counts = Arc::clone(&step_counts);
        let thread_games_completed = Arc::clone(&games_completed);

        let handle = thread::spawn(move || {
            for _ in 0..thread_games {
                let (win, steps) = play_game(max_steps, false, print_interval);

                // Update shared statistics
                let mut completed = thread_games_completed.lock().unwrap();
                *completed += 1;

                if win {
                    let mut win_count = thread_wins.lock().unwrap();
                    *win_count += 1;
                }

                {
                    let mut steps_total = thread_total_steps.lock().unwrap();
                    *steps_total += steps;
                }

                {
                    let mut steps_vec = thread_step_counts.lock().unwrap();
                    steps_vec.push(steps);
                }

                // Print progress every few games
                if *completed % 10 == 0 || *completed == num_games {
                    print!("Completed {}/{} games...\r", *completed, num_games);
                    io::stdout().flush().unwrap();
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Get final statistics
    let wins = *wins.lock().unwrap();
    let total_steps = *total_steps.lock().unwrap();

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
    
    // Calculate approximate speedup based on sequential vs parallel time
    let sequential_estimate = duration.as_secs_f64() * num_threads as f64 / num_games as f64;
    let speedup = sequential_estimate / (duration.as_secs_f64() / num_games as f64);
    println!("Approximate speedup: {:.2}x (using {} threads)", speedup, num_threads);
}

fn main() {
    // Parse command line arguments
    let args = Args::parse();

    if args.games == 1 {
        // Play a single game
        let (win, steps) = play_game(args.max_steps, args.verbose, args.print_interval);
        println!(
            "Game {} after {} steps",
            if win { "won" } else { "lost" },
            steps
        );
    } else {
        // Play multiple games and gather statistics
        play_multiple_games(args.games, args.max_steps, args.print_interval, args.threads);
    }
}
