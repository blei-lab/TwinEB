# """
#     Data utilities for the project, including preprocessing, binarization
#     @Author: De-identified Author
# """

import torch

from training_utils import sparse_batch_collate, sparse_batch_collate_batch

def binarize_df(xx, cutoff=3):
    def bb(uu, cutoff=3):
        if cutoff == 0:
            uu[uu > 0] = 1
            return uu
        uu[uu <= cutoff] = 0
        uu[uu > cutoff] = 1
        return uu

    def quick_bin(yy, cutoff):
        yy.counts = bb(yy.counts, cutoff)
        yy.vad = bb(yy.vad, cutoff)
        yy.train = bb(yy.train, cutoff)
        return yy

    xx = quick_bin(xx, cutoff)
    xx.heldout_data = quick_bin(xx.heldout_data, cutoff)
    return xx


def count_normalize(xx):
    """Compute log1p and normalize total"""
    import scipy.sparse as sp

    def ppp(uu, libsize=10000):
        """where uu is a sparse matrix"""
        # cast uu between zero and one, then multiply by 10K
        # compute the rows sums, then divide by rowsums
        row_sums = uu.sum(axis=1)
        # divide each row by its sum
        # uu = uu / row_sums
        row_sums_diag = sp.diags(1 / row_sums.A.ravel())
        # Multiply each row by the reciprocal of its sum
        uu = row_sums_diag.dot(uu)
        # multiply by libsize
        uu = uu * libsize
        # then compute log1p
        # uu = np.log1p(uu)
        # convert back to nearest integer
        uu = uu.astype(int)
        return uu

    def pp_1(yy):
        yy.counts = ppp(yy.counts)
        yy.vad = ppp(yy.vad)
        yy.train = ppp(yy.train)
        return yy

    xx = pp_1(xx)
    xx.heldout_data = pp_1(xx.heldout_data)
    return xx



def apply_preprocessing(dataset, factor_model, binarize_data=True):
    """
    For the Movielens and goodreads datasets, and PMF models, we need to binarize the counts
    """
    bin_datasets = ["ml-100k", "ml-1m", "goodreads"]
    #bin_datasets = ["ml-100k"]
    if any([x in dataset.original_data_path for x in bin_datasets]):
        if "PMF" in factor_model:
            if binarize_data:
                print("Binarizing the data!")
                cutoff = 0 if "goodreads" in dataset.original_data_path else 1
                dataset = binarize_df(dataset, cutoff=cutoff)

    # for the RNAseq data, totoal and log1p normalize
    if "Ru1322b_" in dataset.original_data_path:
        if "PMF" in factor_model:
            print("Count normalizing the data!")
            dataset = count_normalize(dataset)
    if 'user_artists' in dataset.original_data_path:
        #breakpoint()
        pass
    return dataset



def setup_data_loader(dataset, use_batch_sampler, batch_size, dataset_kwargs):
    """
    # see here: https://gist.github.com/KyleOng/e70c80c49991613eaa8acc7c238576f5
    """
    if use_batch_sampler:
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(dataset),
            batch_size=batch_size,
            drop_last=False,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=sparse_batch_collate_batch,
            sampler=sampler,
            **dataset_kwargs,
        )
    else:
        print("Not using Batch Sampler")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=sparse_batch_collate,
            **dataset_kwargs,
        )
    return  data_loader