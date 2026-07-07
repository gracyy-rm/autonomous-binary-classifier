from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cvcore.image_operations.io import (
    load_image,
    image_info,
    image_statistics,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _iter_image_files(
    split_dir: str | Path,
    class_map: dict[str, str],
):
    """
    Iterate through all images in the dataset split.

    Parameters
    ----------
    split_dir : str | Path
        Path to the dataset split (e.g. train or val).

    class_map : dict[str, str]
        Mapping between folder names and labels.

    Yields
    ------
    tuple[Path, str]
        Image path and corresponding label.
    """

    split_dir = Path(split_dir).resolve()

    for folder_name, label in class_map.items():

        class_dir = split_dir / folder_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Folder not found: {class_dir}"
            )

        # FAST SCANNING: os.scandir is significantly faster than sorted(iterdir()) on Kaggle
        with os.scandir(class_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        yield Path(entry.path), label


def _process_image(
    image_path: Path,
    label: str,
    root_dir: Path,
) -> dict:
    """
    Generate metadata for a single image.

    Parameters
    ----------
    image_path : Path
        Absolute image path.

    label : str
        Class label.

    root_dir : Path
        Root directory used for generating relative paths.

    Returns
    -------
    dict
        Metadata dictionary.
    """

    image = load_image(str(image_path))

    relative_path = image_path.resolve().relative_to(root_dir.resolve())

    record = {
        "image_path": str(relative_path),
        "label": label,
        **image_info(image),
        **image_statistics(image),
    }

    return record


def create_metadata_csv(
    split_dir: str | Path,
    output_csv: str | Path,
    class_map: dict[str, str],
) -> pd.DataFrame:
    """
    Generate metadata CSV for a dataset split.
    """
    split_dir = Path(split_dir).resolve()
    root_dir = split_dir.parent

    tasks = list(
        _iter_image_files(
            split_dir=split_dir,
            class_map=class_map,
        )
    )

    records = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(
                _process_image,
                image_path=task[0],
                label=task[1],
                root_dir=root_dir,
            ): task[0]
            for task in tasks
        }

        # Create an iterator for the completed tasks
        futures_iterator = as_completed(futures)

        with tqdm(total=len(tasks), desc="Generating metadata") as pbar:
            while len(futures) > 0:
                try:
                    # If a single image takes more than 5 seconds, it will raise a TimeoutError
                    future = next(futures_iterator)
                    img_path = futures[future]
                    
                    try:
                        records.append(future.result())
                    except Exception as e:
                        print(f"\n[ERROR] Failed processing {img_path.name}: {e}")
                    
                    # Remove the completed future from our tracking dict
                    del futures[future]
                    pbar.update(1)

                except StopIteration:
                    break
                except TimeoutError:
                    # Find which outstanding future timed out to log it
                    print("\n[TIMEOUT] An image took too long to process. Skipping a corrupted file...")
                    pbar.update(1)
                    # Break out or refresh iterator if needed, but a standard timeout handles it smoothly

    metadata_df = pd.DataFrame(records)
    output_csv = Path(output_csv)
    metadata_df.to_csv(output_csv, index=False)

    return metadata_df