import numpy as np
import pandas as pd
import chess


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


def delta_eval(df):
    df_copy = df.copy()
    factor = 0.00368208

    # Compute win probabilities
    df_copy['win_prob'] = 50 + 50 * (2 / (1 + np.exp(-factor * df_copy['eval'])) - 1)

    # If last move of a game is NaN, assign 100 for white, 0 for black
    last_moves = df_copy.groupby('game_id').tail(1)
    mask = last_moves['eval'].isna()
    df_copy.loc[last_moves[mask].index, 'win_prob'] = last_moves[mask]['color'].map({'w': 100, 'b': 0})

    # Compute deltas
    df_copy['delta_eval'] = df_copy.groupby('game_id')['win_prob'].diff()

    # First moves: baseline ~52 for white
    first_idxs = df_copy.groupby('game_id').head(1).index
    df_copy.loc[first_idxs, 'delta_eval'] = df_copy.loc[first_idxs, 'win_prob'] - 52

    # Flip sign for black moves
    df_copy.loc[df_copy['color'] == 'b', 'delta_eval'] *= -1

    # Return only the relevant columns
    return df_copy[['win_prob', 'delta_eval']]



def eval_to_win_prob(eval_series):

    # Ensure we can handle single values or series
    eval_array = np.asarray(eval_series)

    # Logistic function calibrated for chess evaluations
    # The constant 0.00368208 converts centipawns to logistic scale
    conversion_factor = 0.00368208

    # Calculate win probability using logistic transformation
    win_prob = 50 + 50 * (2 / (1 + np.exp(-conversion_factor * eval_array)) - 1)



    return win_prob

def chess_features(df):
    def count_material(board):
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        white_material = sum(piece_values[piece.piece_type]
                            for piece in board.piece_map().values()
                            if piece.color == chess.WHITE)
        black_material = sum(piece_values[piece.piece_type]
                            for piece in board.piece_map().values()
                            if piece.color == chess.BLACK)
        return white_material, black_material

    num_legal_moves_list = []
    num_captures_list = []
    material_white_list = []
    material_black_list = []
    is_check_list = []

    for fen in df['fen_before']:
        board = chess.Board(fen)
        num_legal_moves = len(list(board.legal_moves))
        num_captures = len(list(board.generate_legal_captures()))
        material_white, material_black = count_material(board)
        is_check = board.is_check()

        num_legal_moves_list.append(num_legal_moves)
        num_captures_list.append(num_captures)
        material_white_list.append(material_white)
        material_black_list.append(material_black)
        is_check_list.append(int(is_check))

    # Add new columns to the DataFrame
    df['num_legal_moves'] = num_legal_moves_list
    df['num_captures'] = num_captures_list
    df['material_white'] = material_white_list
    df['material_black'] = material_black_list
    df['is_check'] = is_check_list

    return df
