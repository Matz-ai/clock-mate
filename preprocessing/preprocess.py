import pandas as pd

def preprocess_data(df_game_info, df_moves):
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
    df_game_info_copy = df_game_info.copy()

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
        'move_number', 'move_san', 'move_uci', 'comment', 'nags', "emt_s", "T"
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
