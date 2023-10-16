import pandas as pd
from pathlib import Path

LABELED_CSV_NAME: str = 'eyes_labeled.csv'
OPEN_LABEL: str = 'Open_Eyes'
CLOSED_LABEL: str = 'Closed_Eyes'
TRAIN_DIR: str = "train"
IMAGE_FORMAT: str = "*.png"


def create_image_label_dataframe():
    """
    Create a Pandas DataFrame containing image paths and corresponding labels.

    Returns:
        pd.DataFrame: DataFrame with 'image_path' and 'label' columns.
    """
    data = []
    train_dir = Path.cwd() / TRAIN_DIR

    for label in [OPEN_LABEL, CLOSED_LABEL]:
        label_dir = train_dir / label
        if label_dir.exists():
            images_paths = label_dir.glob(IMAGE_FORMAT)
            for image_path in images_paths:
                data.append((str(image_path.relative_to(Path.cwd())), label))

    df = pd.DataFrame(data, columns=['image_path', 'label'])
    return df


def save_dataframe_to_csv(dataframe):
    """
    Save a DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): DataFrame to be saved.
    """
    dataframe.to_csv(LABELED_CSV_NAME, index=False)


def main(argv=None):
    """
    Main function to create a labeled CSV file from image data.
    """
    result_df = create_image_label_dataframe()
    save_dataframe_to_csv(result_df)


if __name__ == "__main__":
    main()
