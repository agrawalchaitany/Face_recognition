import os
import cv2
import pandas as pd
import logging

def process_lfw_dataset(raw_data_path, processed_data_path, lfw_allnames_file):
    """
    Process the LFW dataset by resizing images to 128x128 and saving them to the processed data path.

    Args:
        raw_data_path (str): Path to the raw LFW dataset.
        processed_data_path (str): Path to save processed images.
        lfw_allnames_file (str): CSV file containing names and image counts.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        people_data = pd.read_csv(lfw_allnames_file)
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        return

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    logging.info("Starting image processing...")
    for index, row in people_data.iterrows():
        person = row.iloc[0]
        images_count_raw = row.iloc[1]

        if pd.isna(images_count_raw):
            logging.warning(f"Invalid or missing value for images count in row {index + 1}. Skipping.")
            continue

        try:
            images_count = int(images_count_raw)
        except ValueError:
            logging.warning(f"Invalid images count for {person}. Skipping.")
            continue

        person_path = os.path.join(raw_data_path, person)
        if not os.path.exists(person_path):
            logging.warning(f"Directory for {person} not found. Skipping.")
            continue

        jpg_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
        if not jpg_files:
            logging.error(f"No images found for {person}. Skipping.")
            continue

        logging.info(f"Processing images for {person}...")
        for i in range(1, images_count + 1):
            img_path = os.path.join(person_path, f"{person}_{str(i).zfill(4)}.jpg")
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Image not found for {person}_{str(i).zfill(4)}.jpg. Skipping.")
                continue

            processed_image = cv2.resize(image, (128, 128))

            save_path = os.path.join(processed_data_path, person)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, f"{person}_{str(i).zfill(4)}.jpg"), processed_image)

        logging.info(f"Finished processing for {person}")

    logging.info("Processing complete.")

if __name__ == "__main__":
    raw_data_path = "data/raw/lfw"
    processed_data_path = 'data/processed'
    lfw_allnames_file = 'datasets/lfw_allnames.csv'

    process_lfw_dataset(raw_data_path, processed_data_path, lfw_allnames_file)
