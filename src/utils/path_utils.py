from pathlib import Path
import pandas as pd


def convert_to_relative_paths(
    dataframe: pd.DataFrame,
    root_dir: str,
    image_path_col: str = "image_path"
) -> pd.DataFrame:
    """
    Convert absolute image paths into paths relative to the dataset root.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing an image path column.

    root_dir : str
        Root directory of the dataset.

    image_path_col : str, default="image_path"
        Name of the column containing image paths.

    Returns
    -------
    pd.DataFrame
        Copy of the DataFrame with updated relative image paths.
    """

    df = dataframe.copy()

    root_path = Path(root_dir)

    df[image_path_col] = (
        df[image_path_col]
        .apply(lambda x: str(Path(x).relative_to(root_path)))
    )

    return df