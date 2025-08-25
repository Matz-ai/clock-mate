#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import sys
import argparse
from typing import Optional, Tuple, List, Dict

import pandas as pd
from tqdm import tqdm

import chess
import chess.pgn

EVAL_RE = re.compile(r"\[%eval\s+([^\]]+)\]")
CLK_RE  = re.compile(r"\[%clk\s+([0-9:]+)\]")    # e.g. 0:03:21
EMT_RE  = re.compile(r"\[%emt\s+([0-9.]+)\]")    # e.g. 12 or 12.5 seconds

# -------------------------
# Utilities
# -------------------------

def open_pgn_stream(path: str) -> io.TextIOBase:
    """
    Open a PGN or PGN.ZST file as a text stream.
    Requires `zstandard` for .zst files.
    """
    if path.endswith(".zst"):
        try:
            import zstandard as zstd
        except ModuleNotFoundError:
            print("ERROR: zstandard not installed. `pip install zstandard`", file=sys.stderr)
            sys.exit(2)
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="ignore")
        return text
    else:
        return open(path, "r", encoding="utf-8", errors="ignore")

def hms_to_seconds(hms: str) -> Optional[float]:
    try:
        parts = [int(p) for p in hms.split(":")]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0; m, s = parts
        elif len(parts) == 1:
            return float(parts[0])
        else:
            return None
        return float(h*3600 + m*60 + s)
    except Exception:
        return None

def parse_eval_token(token: str) -> Optional[float]:
    """Convert Lichess/PGN eval token to centipawns (mate -> +/-32000)."""
    token = token.strip()
    if token.startswith("#"):
        try:
            n = int(token[1:])  # can be negative
            return 32000.0 if n > 0 else -32000.0
        except Exception:
            return None
    try:
        pawns = float(token)
        return pawns * 100.0
    except Exception:
        return None

def parse_time_control(tc: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse TimeControl tag like '600+5' or '300' or '-'."""
    if not tc or tc == "-" or tc.strip() == "":
        return None, None
    if "+" in tc:
        base, inc = tc.split("+", 1)
        try:
            return float(base), float(inc)
        except Exception:
            return None, None
    try:
        return float(tc), 0.0
    except Exception:
        return None, None

def cadence_from_base(base_s: Optional[float]) -> Optional[str]:
    if base_s is None:
        return None
    if base_s <= 120:
        return "Bullet"
    if base_s <= 480:
        return "Blitz"
    if base_s <= 1500:
        return "Rapid"
    return "Classical"

def extract_game_id_from_site(site_url: str) -> Optional[str]:
    if not site_url:
        return None
    return site_url.rstrip("/").split("/")[-1]

# -------------------------
# Core parsing
# -------------------------

def game_has_evals(game: chess.pgn.Game) -> bool:
    """Quick check: walk mainline comments for any [%eval ...]."""
    node = game
    while node.variations:
        node = node.variation(0)
        if node.comment and EVAL_RE.search(node.comment):
            return True
    return False

def build_rows_from_game(game: chess.pgn.Game) -> Tuple[List[dict], dict]:
    """Return (moves_rows, game_info_row). Assumes evals exist somewhere."""
    tags = game.headers
    site = tags.get("Site", "")
    game_id = extract_game_id_from_site(site) or tags.get("LichessURL") or tags.get("GameId") or ""

    tc = tags.get("TimeControl", "")
    base_s, inc_s = parse_time_control(tc)
    cadence = cadence_from_base(base_s)

    game_info = {
        "game_id": game_id,
        "Event": tags.get("Event", ""),
        "Site": site,
        "Date": tags.get("Date", ""),
        "Round": tags.get("Round", ""),
        "White": tags.get("White", ""),
        "Black": tags.get("Black", ""),
        "Result": tags.get("Result", ""),
        "UTCDate": tags.get("UTCDate", ""),
        "UTCTime": tags.get("UTCTime", ""),
        "WhiteElo": tags.get("WhiteElo", ""),
        "BlackElo": tags.get("BlackElo", ""),
        "WhiteRatingDiff": tags.get("WhiteRatingDiff", ""),
        "BlackRatingDiff": tags.get("BlackRatingDiff", ""),
        "ECO": tags.get("ECO", ""),
        "Opening": tags.get("Opening", ""),
        "TimeControl": tc,
        "Termination": tags.get("Termination", ""),
        "BaseTime_s": base_s,
        "Increment_s": inc_s,
        "Cadence": cadence,
        "BlackTitle": tags.get("BlackTitle", ""),
        "WhiteTitle": tags.get("WhiteTitle", ""),
    }

    moves_rows = []
    node = game
    board = game.board()
    ply = 0
    last_clock_by_color = {"w": None, "b": None}

    any_eval = False

    while node.variations:
        next_node = node.variation(0)
        move = next_node.move

        color = "w" if board.turn else "b"
        fen_before = board.fen()
        move_san = board.san(move)
        move_uci = move.uci()
        move_number = board.fullmove_number

        comment = next_node.comment or ""
        nags = " ".join(sorted([str(n) for n in next_node.nags])) if next_node.nags else ""

        eval_cp = None
        m_eval = EVAL_RE.search(comment)
        if m_eval:
            eval_cp = parse_eval_token(m_eval.group(1))
            if eval_cp is not None:
                any_eval = True

        clk_s = None
        m_clk = CLK_RE.search(comment)
        if m_clk:
            clk_s = hms_to_seconds(m_clk.group(1))

        emt_s = None
        m_emt = EMT_RE.search(comment)
        if m_emt:
            try:
                emt_s = float(m_emt.group(1))
            except Exception:
                emt_s = None

        time_spent_s = None
        if emt_s is not None:
            time_spent_s = emt_s
        elif clk_s is not None:
            prev = last_clock_by_color[color]
            if prev is not None and clk_s is not None:
                time_spent_s = max(0.0, prev - clk_s)
            last_clock_by_color[color] = clk_s
        if clk_s is not None:
            last_clock_by_color[color] = clk_s

        moves_rows.append({
            "game_id": game_id,
            "ply": ply + 1,
            "move_number": move_number,
            "color": color,
            "move_san": move_san,
            "move_uci": move_uci,
            "comment": comment,
            "nags": nags,
            "clock_s": clk_s,
            "emt_s": emt_s,
            "time_spent_s": time_spent_s,
            "eval": eval_cp,
            "fen_before": fen_before,
        })

        board.push(move)
        node = next_node
        ply += 1

    # If no eval actually attached to any move row, return empty to signal skip
    if not any_eval:
        return [], game_info

    return moves_rows, game_info

def collect_from_pgn(path: str, max_games: int, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    df_moves_cols = [
        "game_id","ply","move_number","color","move_san","move_uci","comment","nags",
        "clock_s","emt_s","time_spent_s","eval","fen_before"
    ]
    df_info_cols = [
        "game_id","Event","Site","Date","Round","White","Black","Result",
        "UTCDate","UTCTime","WhiteElo","BlackElo","WhiteRatingDiff","BlackRatingDiff",
        "ECO","Opening","TimeControl","Termination","BaseTime_s","Increment_s","Cadence",
        "BlackTitle","WhiteTitle"
    ]

    moves_records: List[dict] = []
    info_records: List[dict] = []

    kept = 0

    with open_pgn_stream(path) as stream:
        pbar = tqdm(desc="Parsing games", unit="game")
        while kept < max_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            # quick filter: only keep if at least one [%eval ...] in comments
            if not game_has_evals(game):
                pbar.update(1)
                continue
            mrows, ginfo = build_rows_from_game(game)
            if not mrows:
                # game_has_evals True but no eval cp extracted (weird), skip
                pbar.update(1)
                continue
            moves_records.extend(mrows)
            info_records.append(ginfo)
            kept += 1
            pbar.set_postfix({"kept": kept})
            pbar.update(1)
        pbar.close()

    df_moves = pd.DataFrame(moves_records, columns=df_moves_cols)
    df_info  = pd.DataFrame(info_records, columns=df_info_cols)

    moves_path = os.path.join(out_dir, "df_moves.parquet")
    info_path  = os.path.join(out_dir, "df_game_info.parquet")
    df_moves.to_parquet(moves_path, index=False)
    df_info.to_parquet(info_path, index=False)

    print(f"Kept games with evals: {kept}")
    print(f"Saved: {moves_path}")
    print(f"Saved: {info_path}")
    return df_moves, df_info

def main():
    ap = argparse.ArgumentParser(description="Parse a Lichess PGN(.zst) and export only games WITH evals to Parquet.")
    ap.add_argument("--pgn", required=True, help="Path to lichess_db_standard_rated_YYYY-MM.pgn or .pgn.zst")
    ap.add_argument("--max-games", type=int, default=10000, help="Number of games with evals to keep")
    ap.add_argument("--out-dir", type=str, default="out_lichess_pgn", help="Output directory")
    args = ap.parse_args()

    collect_from_pgn(args.pgn, max_games=args.max_games, out_dir=args.out_dir)

if __name__ == "__main__":
    main()


#Commande pour l'utiliser : python lichess_pgn_to_parquet_with_evals.py \                                              [🐍 lewagon]
#   --pgn lichess_db_standard_rated_2025-07.pgn.zst \
#   --max-games 100000 --out-dir data_bis
