pub type Reward = f32;
pub type Action = usize;
pub type Advantage = f32;
pub type Value = f32;


/// A player of Connect4
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Player {
    Player1,
    Player2,
}

/// Get the opponent of a given player
pub fn opponent(p: Player) -> Player {
    match p {
        Player::Player1 => Player::Player2,
        Player::Player2 => Player::Player1,
    }
}

/// How we observe the game from the perspective of a given player
#[derive(Debug, Clone)]
pub struct Observation<const WIDTH: usize, const HEIGHT: usize> {
    pub board: [[[bool; WIDTH]; HEIGHT]; 2],
}

impl<const WIDTH: usize, const HEIGHT: usize> Observation<WIDTH, HEIGHT> {
    /// return a vector of the board
    pub fn reshape_board(&self) -> Vec<bool> {
        let mut board = vec![false; WIDTH * HEIGHT * 2];
        for c in 0..2 {
            for y in 0..HEIGHT {
                board.extend_from_slice(self.board[c][y].as_slice());
            }
        }
        board
    }
}

fn has_connect4<const WIDTH: usize, const HEIGHT: usize>(
    boardslice: [[bool; WIDTH]; HEIGHT],
) -> bool {
    // Check for horizontal connect4
    for row in boardslice.iter() {
        let mut count = 0;
        for cell in row.iter() {
            if *cell {
                count += 1;
                if count == 4 {
                    return true;
                }
            } else {
                count = 0;
            }
        }
    }

    // Check for vertical connect4
    for x in 0..WIDTH {
        let mut count = 0;
        for y in 0..HEIGHT {
            if boardslice[y][x] {
                count += 1;
                if count == 4 {
                    return true;
                }
            } else {
                count = 0;
            }
        }
    }

    // Check for diagonal connect4
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let mut count = 0;
            let mut x = x;
            let mut y = y;
            while x < WIDTH && y < HEIGHT {
                if boardslice[y][x] {
                    count += 1;
                    if count == 4 {
                        return true;
                    }
                } else {
                    count = 0;
                }
                x += 1;
                y += 1;
            }
        }
    }

    // Check for diagonal connect4
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let mut count = 0;
            let mut x = x;
            let mut y = y;
            while x < WIDTH && y < HEIGHT {
                if boardslice[y][x] {
                    count += 1;
                    if count == 4 {
                        return true;
                    }
                } else {
                    count = 0;
                }
                x += 1;
                y -= 1;
            }
        }
    }

    false
}

/// State of the game
#[derive(Debug, Clone)]
pub struct State<const WIDTH: usize, const HEIGHT: usize> {
    board: [[Option<Player>; WIDTH]; HEIGHT],
    game_over: bool,
    moves: Vec<(usize, usize)>,
}

impl<const WIDTH: usize, const HEIGHT: usize> State<WIDTH, HEIGHT> {
    pub fn new() -> Self {
        Self {
            board: [[None; WIDTH]; HEIGHT],
            game_over: false,
            moves: Vec::new(),
        }
    }

    pub fn observe(&self, player: Player) -> Observation<WIDTH, HEIGHT> {
        let mut player_arr = [[false; WIDTH]; HEIGHT];
        let mut opponent_arr = [[false; WIDTH]; HEIGHT];
        for (y, row) in self.board.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                if let Some(cell_player) = cell {
                    if cell_player == &player {
                        player_arr[y][x] = true;
                    } else {
                        opponent_arr[y][x] = true;
                    }
                }
            }
        }
        Observation {
            board: [player_arr, opponent_arr],
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.game_over
    }

    pub fn dims(&self) -> (usize, usize) {
        (WIDTH, HEIGHT)
    }

    /// output in (WIDTH,)
    /// has type bool
    pub fn legal_mask(&self) -> tch::Tensor {
        let mut mask = vec![false; WIDTH];
        for x in 0..WIDTH {
            if self.board[HEIGHT - 1][x].is_none() {
                mask[x] = true;
            }
        }
        tch::Tensor::of_slice(&mask)
    }

    /// list of legal moves
    pub fn legal_moves(&self) -> Vec<Action> {
        let mut moves = Vec::new();
        for x in 0..WIDTH {
            if self.board[HEIGHT - 1][x].is_none() {
                moves.push(x);
            }
        }
        moves
    }

    /// returns tuple of (Observation, Reward)
    pub fn step(&mut self, actor: Player, action: usize) -> (Observation<WIDTH, HEIGHT>, Reward) {
        if self.game_over {
            panic!("Game is already over");
        }

        if action >= WIDTH {
            panic!("Action out of bounds");
        }

        if self.board[HEIGHT - 1][action].is_some() {
            panic!("Illegal move");
        }

        for y in 0..HEIGHT {
            if self.board[y][action].is_none() {
                self.board[y][action] = Some(actor);
                self.moves.push((y, action));
                break;
            }
        }

        let observation = self.observe(actor);
        let reward = if has_connect4(observation.board[0]) {
            1.0
        } else if has_connect4(observation.board[1]) {
            -1.0
        } else {
            0.0
        };

        if reward != 0.0 || self.moves.len() == WIDTH * HEIGHT {
            self.game_over = true;
        }

        (observation, reward)
    }

    // undo a move
    pub fn undo(&mut self) {
        if let Some((y, x)) = self.moves.pop() {
            self.board[y][x] = None;
            self.game_over = false;
        }
    }
}
