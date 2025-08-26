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
    # Step 1: Drop 'Bullet' games from df_game_info
    df_game_info = df_game_info[(df_game_info["Cadence"] != "Bullet")]

    # Step 2: Drop specified columns from both dataframes
    deleted_games_features = [
        "Event", "Site", "Date", "Round", "White", "Black", "UTCDate",
        "UTCTime", "ECO", 'Opening', 'WhiteRatingDiff', 'BlackRatingDiff',
        'TimeControl', 'Cadence', 'BlackTitle', 'WhiteTitle'
    ]
    df_game_info.drop(deleted_games_features, axis=1, inplace=True)

    deleted_moves_features = [
        "ply", 'move_number', 'move_san', 'move_uci', 'comment', 'nags', "emt_s"
    ]
    df_moves.drop(deleted_moves_features, axis=1, inplace=True)

    # Step 3: Merge the two dataframes
    df_fusion = pd.merge(df_game_info, df_moves, on="game_id")

    # Step 4: Change string columns to integer
    df_fusion["WhiteElo"] = df_fusion["WhiteElo"].astype(int)
    df_fusion["BlackElo"] = df_fusion["BlackElo"].astype(int)

    # Step 5: Encode categorical columns
    mapping_color = {"w": 0, "b": 1}
    df_fusion["color"] = df_fusion["color"].map(mapping_color)

    mapping_result = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}
    df_fusion["Result"] = df_fusion["Result"].map(mapping_result)

    mapping_termination = {"Normal": 0, "Time forfeit": 1}
    df_fusion["Termination"] = df_fusion["Termination"].map(mapping_termination)

    # Step 6: Drop additional features for the dummy model
    deleted_fusion_features = ['game_id', 'BaseTime_s', 'Increment_s']
    data_dummy = df_fusion.drop(deleted_fusion_features, axis=1)

    # Step 7: Fill NaN values with 0
    data_dummy.fillna(0, inplace=True)

    # Step 8: Keep games with minnimum 2000 elo
    filtre_elo = (df_fusion['WhiteElo'] >= 2000) | (df_fusion['BlackElo'] >= 2000)
    data = data_dummy[filtre_elo]

    return data
