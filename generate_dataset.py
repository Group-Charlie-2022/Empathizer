from styleformer import Styleformer
from dotenv import load_dotenv
import warnings
import zipfile
import random
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
if os.path.exists(unlabelled_data_path):
    print(f"Using unlabelled data from '{unlabelled_data_path}'")
else:
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
    source_sentences = [x.strip() for x in ul.readlines() if x.strip() != ""]
    random.shuffle(source_sentences)
print("Done filtering unlabelled dataset.")

desired_count = min(len(source_sentences), int(sys.argv[1])) if len(sys.argv) >= 2 else len(source_sentences)
print(f"Attempting to label {desired_count} examples...")

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 60 * 60:
        return f"{(seconds/60.):.2f}min"
    return f"{(seconds/3600.):.2f}h"


# style = [0=Casual to Formal, 1=Formal to Casual, 2=Active to Passive, 3=Passive to Active etc..]
succeeded = 0
sf = Styleformer(style = 1)
with open(labelled_data_path, "w") as l:
    l.write("")
start_time = time.time()
for i, source_sentence in enumerate(source_sentences):
    target_sentence = sf.transfer(source_sentence)
    if target_sentence is not None:
        succeeded += 1
        clean = target_sentence.replace("\n", "").replace("\r", "")
        with open(labelled_data_path, "a") as l:
            l.write(f"{source_sentence}\n{clean}\n")
    elapsed = time.time()-start_time
    print(f"[{succeeded}/{desired_count}], {(100*(i+1)/desired_count):.5f}%, Elapsed: {format_time(elapsed)}, Avg: {format_time(elapsed/(i+1))}/sample, ETA: {format_time((desired_count-succeeded)*elapsed/succeeded)}, Attempted: {i+1}, Success rate: {(100*succeeded/(i+1)):.2f}%")
    if succeeded == desired_count:
        break
