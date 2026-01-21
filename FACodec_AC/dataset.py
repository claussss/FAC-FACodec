import os
import glob
import torch
from torch.utils.data import Sampler, BatchSampler, Dataset
from FACodec_AC.config import Config
from FACodec_AC.utils import pad_token_sequence, standardize
import random
from typing import List




class ZContentDataset(Dataset):
    """
    Loads .pt files containing {'zc1_indx': ..., 'zc2_indx': ...}.
    Loads corresponding phones condition files from phones_cond_dir.
    Converts indices to continuous representations using separate codebooks and projection layers for zc1 and zc2.
    Normalizes them using pre-computed stats.
    """
    def __init__(self, data_dir, phones_cond_dir, codebook_zc1, out_proj_zc1, codebook_zc2, out_proj_zc2, facodec_dim):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
        # Enforce phones_cond_dir is provided and exists
        if not os.path.isdir(phones_cond_dir):
            raise FileNotFoundError(f"Phones condition directory {phones_cond_dir} does not exist.")
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.phones_cond_dir = phones_cond_dir
        cond_files = glob.glob(os.path.join(phones_cond_dir, "*.pt"))
        cond_names = {os.path.basename(f) for f in cond_files}
        self.files = [f for f in self.files if os.path.basename(f) in cond_names]
        if codebook_zc1 is None or out_proj_zc1 is None or codebook_zc2 is None or out_proj_zc2 is None:
            raise ValueError("Both codebooks and projection layers for zc1 and zc2 must be provided.")
        self.codebook_zc1 = codebook_zc1
        self.out_proj_zc1 = out_proj_zc1
        self.codebook_zc2 = codebook_zc2
        self.out_proj_zc2 = out_proj_zc2
        # Load normalization stats from parent's stats folder.
        stats_dir = os.path.join(os.path.dirname(data_dir), "stats")
        self.mean_zc1 = torch.load(os.path.join(stats_dir, "mean_zc1_indx.pt"))
        self.std_zc1 = torch.load(os.path.join(stats_dir, "std_zc1_indx.pt"))
        self.mean_zc2 = torch.load(os.path.join(stats_dir, "mean_zc2_indx.pt"))
        self.std_zc2 = torch.load(os.path.join(stats_dir, "std_zc2_indx.pt"))
        # Store facodec_dim (do not refer to Config here)
        self.facodec_dim = facodec_dim

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        if 'zc1_indx' not in data or 'zc2_indx' not in data:
            raise KeyError("Required keys 'zc1_indx' or 'zc2_indx' not found in data.")
        zc1 = data['zc1_indx']  # shape: [1, T]
        zc2 = data['zc2_indx']
        # Index lookup
        with torch.no_grad():
            emb1 = torch.nn.functional.embedding(zc1, self.codebook_zc1)
            emb2 = torch.nn.functional.embedding(zc2, self.codebook_zc2)
            # Conditionally apply projection if facodec_dim==256.
            if self.facodec_dim == 256:
                rep1 = self.out_proj_zc1(emb1)
                rep2 = self.out_proj_zc2(emb2)
            else:
                rep1 = emb1
                rep2 = emb2
        rep1 = rep1.transpose(1, 2)  # now shape: [1, d, T]
        rep2 = rep2.transpose(1, 2)
        zc1_norm = standardize(rep1, self.mean_zc1, self.std_zc1)
        zc2_norm = standardize(rep2, self.mean_zc2, self.std_zc2)
        cond_path = os.path.join(self.phones_cond_dir, os.path.basename(self.files[idx]))
        phone_cond = torch.load(cond_path)
        return zc1_norm, zc2_norm, phone_cond

    def get_seq_length(self, idx):
        # Load the file and return time-length of zc1_indx.
        data = torch.load(self.files[idx], map_location='cpu')
        if 'zc1_indx' not in data:
            raise KeyError("Key 'zc1_indx' not found in data.")
        zc1 = data['zc1_indx']  # shape: [1, T]
        return zc1.shape[1]
    
class FixedSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

class LengthSortedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        batches_per_bucket: int = 10,   # you can tune this
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batches_per_bucket = batches_per_bucket

    def __iter__(self):
        # 1) sort all indices by descending sequence-length
        indices = list(range(len(self.dataset)))
        indices.sort(
            key=lambda idx: self.dataset.get_seq_length(idx),
            reverse=True
        )

        # 2) if requested, do your bucket-level shuffling
        if self.shuffle:
            bucket_size = self.batch_size * self.batches_per_bucket
            buckets = [
                indices[i : i + bucket_size]
                for i in range(0, len(indices), bucket_size)
            ]
            random.shuffle(buckets)
            indices = [idx for bucket in buckets for idx in bucket]

        # 3) emit fixed-size batches
        batch: List[int] = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if (not self.drop_last) and batch:
            yield batch

    def __len__(self):
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            n += 1
        return n

def collate_fn_zcontent(batch):
    # Determine maximum time-length in batch (using zc1 shape: [1, d, T])
    max_len = max(item[0].shape[-1] for item in batch)
    padded_zc1, padded_zc2, padded_phone, masks = [], [], [], []
    for zc1, zc2, phone in batch:
        p_zc1, _ = pad_token_sequence(zc1, max_len, 0)
        p_zc2, _ = pad_token_sequence(zc2, max_len, 0)  # Pad multidim vectors with zeros
        p_phone, mask = pad_token_sequence(phone, max_len, Config.PHONE_PAD_ID)
        padded_zc1.append(p_zc1)
        padded_zc2.append(p_zc2)
        padded_phone.append(p_phone)
        masks.append(mask)
    return torch.stack(padded_zc1, dim=0), torch.stack(padded_zc2, dim=0), torch.stack(padded_phone, dim=0), torch.stack(masks, dim=0)