import numpy as np


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
