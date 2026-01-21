import os
import glob
import random
import tqdm
from sklearn.model_selection import train_test_split
from FACodec_AC.utils import process_wav_facodec, compute_stats
from FACodec_AC.config import Config

# Device configuration
device = Config.device

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))

# Replace manual FACodec setup with the new initializer.
from FACodec_AC.utils import init_facodec_models
fa_encoder, fa_decoder = init_facodec_models(device)


wav_dir = Config.wav_dir

all_wavs = glob.glob(os.path.join(wav_dir, '*.wav'))
print(f"Found {len(all_wavs)} wav files.")

random.seed(42)
train_files, test_files = train_test_split(all_wavs, test_size=0.1, random_state=42)
print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

output_dir = Config.facodec_dataset_dir
train_out = os.path.join(output_dir, 'train')
test_out = os.path.join(output_dir, 'test')
os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

if __name__ == "__main__":
    # Process train set with KeyboardInterrupt handling: if interrupted, jump directly to stats computation
    print("Processing train set...")
    try:
        for f in tqdm.tqdm(train_files, desc="Processing train files"):
            try:
                fp, status = process_wav_facodec(f, fa_encoder, fa_decoder, train_out, device)
                print(f"{fp}: {status}")
            except Exception as e:
                print(f"Error processing {f}: {e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught during train files processing; jumping directly to stats computation.")

    # Process test set.
    print("Processing test set...")
    try:
        for f in tqdm.tqdm(test_files, desc="Processing test files"):
            try:
                fp, status = process_wav_facodec(f, fa_encoder, fa_decoder, test_out, device)
                print(f"{fp}: {status}")
            except Exception as e:
                print(f"Error processing {f}: {e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught during test files processing; jumping directly to stats computation.")

    # Stats computation block
    stats_dir = os.path.join(Config.facodec_dataset_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    # Move to CPU for stats computation because it is needed only for indx lookup.
    fa_decoder = fa_decoder.to('cpu') 
    means, stds = compute_stats(train_out, stats_dir, fa_decoder)

