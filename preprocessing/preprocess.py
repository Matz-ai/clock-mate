import pandas as pd
from preprocessing.fen_transformation import delta_eval, chess_features

def preprocess_data(df_game= "df_game_info", df_moves= "df_moves"):
    """
    Performs data preprocessing steps on chess game dataframes.

    Args:
        df_game_info (pd.DataFrame): DataFrame with game information.
        df_moves (pd.DataFrame): DataFrame with game moves.

    Returns:
        pd.DataFrame: The preprocessed and merged DataFrame.
    """

    # Création de copie des dataframe

    df_moves_copy = df_moves.copy()
    df_game_info_copy = df_game.copy()

    df_moves_copy["win_probability"] = eval_to_win_probability(df_moves_copy["eval"])
    df_moves_copy["delta_eval"] = delta_eval(df_moves_copy)

    # Step 1: Drop unwanted game cadence from df_game_info
    df_game_info_copy = df_game_info_copy[
        (df_game_info_copy["Cadence"] != "Bullet") &
        (df_game_info_copy['BaseTime_s'] != 0.0) &
        (~df_game_info_copy['BaseTime_s'].isna())
    ].copy()

    df_moves_copy = df_moves_copy[~df_moves_copy['clock_s'].isna()].copy()

    # Step 2: Drop specified columns
    deleted_games_features = [
        "Event", "Site", "Date", "Round", "White", "Black", "UTCDate",
        "UTCTime", "ECO", 'Opening', 'WhiteRatingDiff', 'BlackRatingDiff',
        'TimeControl', 'Cadence', 'BlackTitle', 'WhiteTitle'
    ]
    df_game_info_copy = df_game_info_copy.drop(deleted_games_features, axis=1)

    deleted_moves_features = [
        'move_number', 'move_san', 'move_uci', 'comment', 'nags', "emt_s"
    ]
    # No inplace=True, assign the result back to the variable
    df_moves_copy = df_moves_copy.drop(deleted_moves_features, axis=1)

    # Step 3: Merge the two dataframes
    df_fusion = pd.merge(df_game_info_copy, df_moves_copy, on="game_id")

    # Step 4: Encode categorical columns
    mapping_color = {"w": 0, "b": 1}
    df_fusion["color"] = df_fusion["color"].map(mapping_color)

    mapping_result = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}
    df_fusion["Result"] = df_fusion["Result"].map(mapping_result)

    mapping_termination = {"Normal": 0, "Time forfeit": 1}
    df_fusion["Termination"] = df_fusion["Termination"].map(mapping_termination)

    df_fusion["eval"] = df_fusion["eval"].fillna("40000")

    # Step 5: Change string columns to integer
    df_fusion["WhiteElo"] = df_fusion["WhiteElo"].astype(int)
    df_fusion["BlackElo"] = df_fusion["BlackElo"].astype(int)
    df_fusion["eval"]= df_fusion["eval"].astype(int)

    # Step 6: Drop additional features for the dummy mode
    deleted_fusion_features = ['game_id', 'Increment_s']
    data = df_fusion.drop(deleted_fusion_features, axis=1)

    # Step 7: Fill NaN values with 0
    data["time_spent_s"].fillna(0, inplace=True)

    # Step 8: Keep games with minimum 2000 elo
    filtre_elo = (df_fusion['WhiteElo'] >= 2000) | (df_fusion['BlackElo'] >= 2000)
    data = data[filtre_elo]

    return data


#New Version
def preproc_full(df_game_info, df_moves):
    #Inital delete
    deleted_games_features = [
        "Event", "Site", "Date", "Round", "White", "Black", "UTCDate",
        "UTCTime", "ECO", 'Opening', 'WhiteRatingDiff', 'BlackRatingDiff',
        'TimeControl', 'BlackTitle', 'WhiteTitle', 'Cadence'
    ]

    deleted_moves_features = [
        'move_number', 'move_san', 'move_uci', 'comment', 'nags', "emt_s"
    ]

    df_game_info = df_game_info.drop(columns=deleted_games_features)
    df_moves = df_moves.drop(columns=deleted_moves_features)

    #Merge datasets
    df_full = pd.merge(df_game_info, df_moves, on="game_id")

    #Initial filter
    df_full = df_full[
        (df_full['Increment_s'] == 0.0) &
        (df_full['BaseTime_s'] >= 180) &
        (df_full['BaseTime_s'] <= 900)
    ]

    df_full.loc[df_full['BaseTime_s'] == df_full['clock_s'], 'time_spent_s'] = df_full.loc[df_full['BaseTime_s'] == df_full['clock_s'], 'time_spent_s'].fillna(0)

    df_full['check'] = df_full['clock_s'].shift(2) != df_full['clock_s'] + df_full['time_spent_s']
    check_summary = df_full.groupby('game_id')['check'].sum().reset_index()
    valid_games = check_summary[check_summary['check'] == 2]['game_id']
    df_full = df_full[df_full['game_id'].isin(valid_games)]
    df_full.drop(columns='check', inplace=True)

    df_full['check'] = df_full['clock_s'] == df_full['BaseTime_s']
    check_summary = df_full[(df_full['ply'] == 1) | (df_full['ply'] == 2)].groupby('game_id')['check'].sum().reset_index()
    valid_games = check_summary[check_summary['check'] == 2]['game_id']
    df_full = df_full[df_full['game_id'].isin(valid_games)]
    df_full.drop(columns='check', inplace=True)

    df_full = delta_eval(df_full)
    df_full = chess_features(df_full)

    # Step 4: Encode categorical columns
    mapping_color = {"w": 0, "b": 1}
    df_full["color"] = df_full["color"].map(mapping_color)
    df_full["WhiteWin"] = (df_full["Result"] == "1-0").astype(int)
    df_full["BlackWin"] = (df_full["Result"] == "0-1").astype(int)
    mapping_termination = {"Normal": 0, "Time forfeit": 1}
    df_full["Termination"] = df_full["Termination"].map(mapping_termination)

    # Step 5: Change string columns to integer
    df_full["WhiteElo"] = df_full["WhiteElo"].astype(int)
    df_full["BlackElo"] = df_full["BlackElo"].astype(int)

    df_full['clock_s'] = (df_full['clock_s'] + df_full['time_spent_s'])
    df_full['rel_time'] = df_full['time_spent_s']/df_full['clock_s']
    df_full = df_full[df_full['rel_time'].notna()]

    return df_full

def create_X_y(df_full):
    X = df_full[['color', 'ply', 'WhiteWin', 'BlackWin', 'delta_eval', 'WhiteElo',
             'BlackElo', 'num_legal_moves', 'num_captures',
             'material_white', 'material_black', 'is_check', 'clock_s']]
    y = df_full[['rel_time']]
    return X, y
