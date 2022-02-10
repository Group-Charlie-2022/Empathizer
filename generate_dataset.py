from styleformer import Styleformer
from dotenv import load_dotenv
import warnings
import zipfile
import time
import sys
import os
warnings.filterwarnings("ignore")

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi

labelled_data_path = "labelled.txt"
unlabelled_data_path = "unlabelled.txt"

# https://www.kaggle.com/mikeortman/wikipedia-sentences
unlabelled_dataset_kaggle = "mikeortman/wikipedia-sentences"
unlabelled_dataset_kaggle_file = "wikisent2.txt"

# Check whether has been downloaded.
# Since the dataset is quite large, it has been gitignored
if os.path.exists(labelled_data_path):
    print(f"'{labelled_data_path}' already exists. It will not be overwritten.")
    sys.exit()

# If the unlabelled data doesn't exist, fetch it from Kaggle
if not os.path.exists(unlabelled_data_path):
    print(f"'{unlabelled_data_path}' not found. Fetching from Kaggle: '{unlabelled_dataset_kaggle}/{unlabelled_dataset_kaggle_file}'...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(unlabelled_dataset_kaggle, unlabelled_dataset_kaggle_file, quiet=False)
    print("Extracting...")
    with zipfile.ZipFile(f"{unlabelled_dataset_kaggle_file}.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    os.rename(unlabelled_dataset_kaggle_file, unlabelled_data_path)
    os.unlink(f"{unlabelled_dataset_kaggle_file}.zip")
    print("Downloaded unlabelled dataset.")


# Load unlabelled example
with open(unlabelled_data_path, "r") as ul:
    source_sentences = list(x.strip() for x in ul.readlines()[:10] if x.strip() != "")

print(f"Labelling {len(source_sentences)} examples...")

def format_time(seconds):
    return f"{seconds:.2f}s"

# style = [0=Casual to Formal, 1=Formal to Casual, 2=Active to Passive, 3=Passive to Active etc..]
sf = Styleformer(style = 1) 
with open(labelled_data_path, "w") as l:
    start_time = time.time()
    for i, source_sentence in enumerate(source_sentences):
        target_sentence = sf.transfer(source_sentence)
        elapsed = time.time()-start_time
        print(f"Labelled [{i+1}/{len(source_sentences)}], {(100*(i+1)/len(source_sentences)):.5f}%, Elapsed: {format_time(elapsed)}, Avg: {format_time(elapsed/(i+1))}/sample, ETA: {format_time((len(source_sentences)-i-1)*elapsed/(i+1))}")
        if target_sentence is not None:
            clean = target_sentence.replace("\n", "").replace("\r", "")
            l.write(f"{clean}\n{source_sentence}\n")
        else:
            print("Failed to find good label.")
