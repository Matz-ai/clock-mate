import numpy as np
import pandas as pd


piece_to_channel = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,  # White
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11  # Black
}

def fens_to_arrays(fens):
    """
    Convert a batch of FEN strings into an array.
    """
    batch_size = len(fens)
    out = np.zeros((batch_size, 8, 8, 12), dtype=np.float32)

    for i, fen in enumerate(fens):
        board_str = fen.split()[0]  # only piece placement
        for row, rank in enumerate(board_str.split("/")):
            col = 0
            for ch in rank:
                if ch.isdigit():
                    col += int(ch)
                else:
                    out[i, row, col, piece_to_channel[ch]] = 1
                    col += 1
    return out


def delta_win(df, game_id='game_id', eval_col='eval', color_col='color'):
    """
    Calculate the change in chess position evaluation between consecutive moves.
    Can be used for eval or win probability.

    This function computes how much the position evaluation changed from one move
    to the next, with the perspective adjusted based on whose turn it is.

    Parameters:
    -----------
    df : pandas.DataFrame
    game_id : str, default 'game_id'
        Column name containing unique game identifiers
    eval_col : str, default 'eval'
        Column name containing position evaluation scores
    color_col : str, default 'color'
        Column name indicating piece color ('w' for white, 'b' for black)

    Returns:
    --------
    pandas.Series
        Series containing evaluation deltas, where:
        - Positive values indicate improvement for the current player
        - Negative values indicate deterioration for the current player

    """

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Calculate the difference between consecutive evaluations within each game
    df_copy['delta_eval'] = df_copy.groupby(game_id)[eval_col].diff()

    first_moves = df_copy.groupby(game_id).head(1).index

    if eval_col == 'eval':
        df_copy.loc[first_moves, 'delta_eval'] = df_copy.loc[first_moves, eval_col] - 18
    else:
        df_copy.loc[first_moves, 'delta_eval'] = df_copy.loc[first_moves, eval_col] - 50

    # For black moves, flip the sign to show evaluation change from Black's perspective
    # This makes positive deltas always mean "good for the current player"
    black_moves = df_copy[color_col] == 'b'
    df_copy.loc[black_moves, 'delta_eval'] = -df_copy.loc[black_moves, 'delta_eval']

    return df_copy['delta_eval']


def eval_to_win_probability(eval_series):
    """
    Convert chess engine evaluation (in centipawns) to win probability percentage.

    Uses a logistic function calibrated to convert centipawn evaluations into
    estimated win probabilities, based on statistical analysis of chess games.

    Parameters:
    -----------
    eval_series : array-like
        Chess position evaluations in centipawns (100 centipawns = 1 pawn advantage)
        Can be a pandas Series, numpy array, or single numeric value

    Returns:
    --------
    array with win probabilities as percentages (0-100)
    """
    # Ensure we can handle single values or series
    eval_array = np.asarray(eval_series)

    # Logistic function calibrated for chess evaluations
    # The constant 0.00368208 converts centipawns to logistic scale
    conversion_factor = 0.00368208

    # Calculate win probability using logistic transformation
    win_prob = 50 + 50 * (2 / (1 + np.exp(-conversion_factor * eval_array)) - 1)

    return win_prob
