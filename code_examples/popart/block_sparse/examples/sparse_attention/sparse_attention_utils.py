# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt


class Convert:
    """
    There are two converters available:
    -  convert from a 2D Numpy array or a Scipy sparse
       matrix to a popsparse representation
    -  convert from a popsparse matrix
       to an Ndim Numpy array of specified shape
    """
    def to_popsparse(A, blocksize2D):
        """
        Convert an arbitrary 2D numpy array or scipy sparse matrix into
        blocks of a particular size and sparsity mask.
        These blocks and sparsity mask are in the format that popsparse expects

        Parameters
        ----------
        A : a 2D Numpy array or Scipy sparse matrix
        blocksize2D : the blocksize to use when converting the matrix to data blocks

        Returns
        -------
        blocks: the blocks of data with shape [n_blocks, blocksize2D[0]*blocksize2D[1]]
        sparsity_mask : a list of 1s and 0s representing which blocks are active/inactive
        bsr : the Scipy sparse representation of the input matrix
        """
        assert len(A.shape) == 2, "Can only convert 2D structures to popsparse"
        original_dtype = A.dtype
        if original_dtype == np.float16:
            print("Note: FP16 is not supported in Scipy sparse. Casting to FP32 and then back")
            A = np.float32(A)

        bsr = sparse.bsr_matrix(A, blocksize=blocksize2D)
        bsr.eliminate_zeros()

        # The dense blocks
        blocks = np.copy(bsr.data)
        blocks = np.reshape(blocks, [blocks.shape[0], -1])
        if original_dtype == np.float16:
            blocks = np.float16(blocks)

        # Dense mask of active blocks
        mask_data = np.array([[[1]]]*len(bsr.indices))  # mask will contain ones
        sparsity_mask = sparse.bsr_matrix((mask_data, bsr.indices, bsr.indptr))

        # Make sure the mask is the correct shape
        # (necessary if there are trailing blocks of zeros)
        sparsity_mask_shape = [bsr.shape[0]//blocksize2D[-2], bsr.shape[0]//blocksize2D[-1]]
        sparsity_mask.resize(sparsity_mask_shape)
        sparsity_mask = sparsity_mask.toarray().flatten().tolist()

        return blocks, sparsity_mask, bsr

    def to_np_array(numpyShape, dataBlocks, sparsity_mask, groupSizes, blockSize2D):
        """
        Convert from a popsparse blockspare matrix
        to an Ndim Numpy array of specified shape

        Parameters
        ----------
        numpyShape: shape of the desired output array
        dataBlocks: the blocks from popsparse
        sparsity_mask: the sparsity mask from popsparse
        groupSizes: how many blocks per group (a.k.a. bsr_rhs_lengths_per_2d_plane)
        blockSize2D: the 2D blocksize as used by the data

        Returns
        -------
        A dense numpy array of the desired shape and with the data filled in appropriately
        """
        if dataBlocks.dtype == np.float16:
            print("Note: FP16 is not supported in Scipy sparse. Casting to FP32")
            dataBlocks = np.float32(dataBlocks)

        numpyGroups = []
        sliceStart = 0
        sliceEnd = 0
        for g, groupSize in enumerate(groupSizes):
            # Mask
            mask_shape = [numpyShape[-2]//blockSize2D[-2], numpyShape[-1]//blockSize2D[-1]]
            mask_size = mask_shape[0]*mask_shape[1]
            mask = np.reshape(sparsity_mask[g*mask_size:(g+1)*mask_size], mask_shape)
            mask = sparse.bsr_matrix(mask, blocksize = (1, 1))
            mask.eliminate_zeros()

            # Data
            sliceEnd += groupSize
            data = dataBlocks[sliceStart:sliceEnd].reshape([-1, *blockSize2D])
            data = np.array(list(data))
            data = sparse.bsr_matrix((data, mask.indices, mask.indptr), blocksize=blockSize2D)
            data.resize(numpyShape[-2:])  # ensure correct 2D shape
            numpyGroups.append(data.toarray())
            sliceStart = sliceEnd

        out = np.stack(numpyGroups, 0)
        out = np.reshape(out, numpyShape)
        return out


class Patterns:
    """
    A library of helper functions used to create different
    attention patterns
    """
    def random_pattern(sequence_length, density, blocksize2D):
        """
        Create a pattern with random blocks of 0s and 1s i.e. each block is
        either filled entirely with 0s or 1s.
        Each block will be set to 1 with probability equal to density and to 0
        with probability equal to 1-density
        """
        xblocks = sequence_length//blocksize2D[1]
        yblocks = sequence_length//blocksize2D[0]
        size = (yblocks, xblocks)
        pattern = np.random.choice([0, 1], p = [1-density, density], size = size)
        pattern = np.repeat(pattern, blocksize2D[1], axis = 0)
        pattern = np.repeat(pattern, blocksize2D[0], axis = 1)
        return pattern

    def autoregressive_window(window_size):
        """
        Autoregressive mask across windows_size tokens

        Parameters
        ----------
        window_size : the length of the window in terms of tokens
        """
        local_mask = sparse.tril(np.ones([window_size, window_size]), k=0)
        return local_mask

    def summary_window(window_size, n_summary_blocks, blocksize_x):
        """
        A summary mask has all but n_summary_block*blocksize_x
        entries set to 0. The nonzero columns are at the end of the
        window.

        Parameters
        ----------
        window_size : the length of the window in terms of tokens
        n_summary_blocks : how many blocks of non-zero columns to use
        blocksize_x : the number of tokens per block in column direction
        """
        n_summary_tokens = n_summary_blocks*blocksize_x
        local_mask = sparse.lil_matrix((window_size, window_size))
        local_mask[:, window_size-n_summary_tokens:] = 1
        return local_mask

    def tile_pattern_on_diagonal(pattern, n_windows):
        """
        Uses a sparse kronecker product to tile a pattern.
        Typically used with autoregressive patterns.

        Parameters
        ----------
        pattern : a Numpy array representing an attention mask over a single window
        n_windows : how many windows in the tiles sequence
        """
        active_windows = sparse.eye(n_windows)
        global_mask = sparse.kron(active_windows, pattern)
        return global_mask

    def tile_pattern_below_diagonal(pattern, n_windows):
        """
        Uses a sparse kronecker product to tile a pattern
        below the diagonal.
        Typically used for summarization patterns.

        Parameters
        ----------
        pattern : a Numpy array representing an attention mask over a single window
        n_windows : how many windows in the tiles sequence
        """
        # exclude diagonal k=-1
        active_windows = sparse.tril(np.ones([n_windows, n_windows]), k=-1)
        global_mask = sparse.kron(active_windows, pattern)
        return global_mask

    def add_identity_pattern(pattern):
        """
        Ensure that each token can attend to itself by
        adding and identity pattern (main diagonal filled with ones)

        Parameters
        ----------
        pattern : a Numpy array representing an attention mask
        """
        return pattern + sparse.identity(pattern.shape[0])

    def set_merge_two_patterns(A, B):
        """
        Merge/combine two patterns. The resulting pattern will contain ones for entries
        which are in pattern A or pattern B and zeroes everywhere else.
        Parameters
        ----------
        A : first pattern (a Numpy array)
        B : second pattern (a Numpy array)
        """
        # Values are forced to 0 or 1
        # e.g. 1 + 1 -> 1
        return (A + B).sign()

    def set_subtract_pattern(A, B):
        """
        Zero out entries of A, which also occur in B

        Parameters
        ----------
        A : first pattern (a Numpy array)
        B : second pattern (a Numpy array)
        """
        return (A - B).sign()

    def plot_pattern(A, blocksize2D, filename, show_token_grid=False, show_block_grid=True):
        """
        Plots a pattern A using matplotlib and save the figure to filename
        Plotting will be slow for long sequences.
        Use show_token_grid variable to display the token-level grid (usually not
        useful for large sequence lengths)
        Use show_block_grid variable to display the block-level grid

        Parameters
        ----------
        A : a 2D Numpy array or Scipy sparse matrix
        blocksize2D : the gridsize to use for the block-level grid
        filename : a string describing where to store the image
        show_token_grid: whether to show a grey token-level grid
        show_block_grid: whether to show a sea-green block-level grid
        """
        # Handle both 2D Numpy arrays and Scipy sparse matrices
        array_to_plot = A if not sparse.issparse(A) else A.toarray()

        plt.figure()
        plt.axis('off')

        edge_colors = 'grey' if show_token_grid else 'face'
        # pcolor plots things wrong side up, so flip Up Down

        plt.pcolor(np.flipud(array_to_plot), edgecolors = edge_colors, cmap = 'Blues', vmin = 0, vmax = 1)

        if show_block_grid:
            y = np.arange(0, array_to_plot.shape[0]+1, blocksize2D[0])
            x = np.arange(0, array_to_plot.shape[1]+1, blocksize2D[1])
            plt.vlines(x, 0, array_to_plot.shape[1], colors="lightseagreen", lw=1)
            plt.hlines(y, 0, array_to_plot.shape[0], colors="lightseagreen", lw=1)
        plt.axis('equal')
        plt.savefig(filename, bbox_inches='tight', dpi=200)
        plt.close()


class Heads:
    """
    A class giving quick access to a few starter types of attention heads
    and a utility to concatenate them (for multi-head attention)
    Each heads is characterized by a tuple of (blocks, sparsity, bsr)
    """
    def from_custom_pattern(pattern, blocksize2D):
        """
        Given a custom user-defined pattern create the tuple of
        (blocks, sparsity, bsr) that define an attention head
        Note that blocks should be either fully dense or full sparse. Any sparsity inside
        the block won't be handled by the block-sparse sparse softmax.
        """
        blocks, sparsity, scipy_bsr = Convert.to_popsparse(pattern, blocksize2D)
        return blocks, sparsity, scipy_bsr

    def dense_self_attention(sequence_length, blocksize2D):
        """
        A complete self attention mask (all tokens attend to all other tokens),
        but in block-sparse representation (useful for regression testing)

        """
        a = np.ones([sequence_length, sequence_length])
        blocks, sparsity, scipy_bsr = Convert.to_popsparse(a, blocksize2D)
        return blocks, sparsity, scipy_bsr

    def causal_autoregressive(sequence_length, blocksize2D):
        """
        A autoregressive mask across the entire sequence_length (lower triangular)
        but in block-sparse representation (useful for regression testing)
        """
        # Autoregressive pattern across the entire sequence
        a = Patterns.autoregressive_window(sequence_length)
        # To popsparse
        blocks, sparsity, scipy_bsr = Convert.to_popsparse(a, blocksize2D)
        return blocks, sparsity, scipy_bsr

    def causal_windows_with_summaries(window_size, n_windows, n_summary_blocks, blocksize2D):
        """
        A autoregressive mask on each window and summarization blocks to pass information
        along to future windows
        """
        # Autoregressive pattern
        a = Patterns.autoregressive_window(window_size)
        a = Patterns.tile_pattern_on_diagonal(a, n_windows)

        # Summarization pattern
        s = Patterns.summary_window(window_size, n_summary_blocks, blocksize2D[1])
        s = Patterns.tile_pattern_below_diagonal(s, n_windows)

        # Merged pattern
        merged_pattern = Patterns.set_merge_two_patterns(a, s)

        # To popsparse
        blocks, sparsity, merged_pattern = Convert.to_popsparse(merged_pattern, blocksize2D)
        return blocks, sparsity, merged_pattern

    def block_gram(sequence_length, n_blocks, blocksize2D):
        """
        Similar to an n-gram model the block_gram mask attends to
        n_blocks * blocksize_x tokens forward and backward.
        e.g. if n_blocks is 2 then look at current block, one forward and one backward
        """
        shapeInBlocks = [sequence_length//blocksize2D[0], sequence_length//blocksize2D[1]]
        merged_pattern = None
        for i in range(-n_blocks+1, n_blocks):
            C = sparse.dia_matrix((np.ones((1, sequence_length)), [i]), shape=shapeInBlocks)
            b = sparse.kron(C, np.ones(blocksize2D))
            if merged_pattern is not None:
                merged_pattern += b
            else:
                merged_pattern = b

        blocks, sparsity, scipy_bsr = Convert.to_popsparse(merged_pattern, blocksize2D)
        return blocks, sparsity, scipy_bsr

    def causal_block_gram(sequence_length, n_blocks, blocksize2D):
        """
        Attend to n_blocks*blocksize2D[0] of preceding tokens.
        """
        assert n_blocks > 0, "n_blocks must be larger than 0"

        # The autoregressive blocks
        a = Patterns.autoregressive_window(blocksize2D[1])
        shapeInBlocks = [sequence_length//blocksize2D[0], sequence_length//blocksize2D[1]]
        a = Patterns.tile_pattern_on_diagonal(a, shapeInBlocks[1])

        # All remaining blocks
        for i in range(-n_blocks+1, 0):
            C = sparse.dia_matrix((np.ones((1, sequence_length)), [i]), shape=shapeInBlocks)
            a += sparse.kron(C, np.ones(blocksize2D))

        blocks, sparsity, scipy_bsr = Convert.to_popsparse(a, blocksize2D)
        return blocks, sparsity, scipy_bsr

    def concatenate_heads(list_of_heads, repeats_per_head):
        """
        Given a list od different heads (blocks, sparsity) tuple for each,
        and a list of how many times to repeat each head, returns the concatenated blocks
        and sparsity masks as well as the bsr_rhs_lengths_per_2d_plane variable needed
        by grouped BsMatmul

        Parameters
        ----------
        list_of_heads : a list of tuples of (blocks, sparsity_mask, bsr)
        repeats_per_head : a list of integers denoting how many times to repeat a head
        """
        concatBlocks = []
        concatSparsity = []
        bsr_rhs_lengths_per_2d_plane = []
        for (blocks, sparsity, *_), repeats in zip(list_of_heads, repeats_per_head):
            concatBlocks.append(np.tile(blocks, [repeats, 1]))
            concatSparsity.append(np.tile(sparsity, [repeats, 1]))
            assert blocks.shape[0] == sum(sparsity), "Number of data blocks does not match sparsity"
            bsr_rhs_lengths_per_2d_plane.extend([sum(sparsity)]*repeats)

        concatBlocks = np.concatenate(concatBlocks, axis=0)
        concatSparsity = np.concatenate(concatSparsity, axis=0).flatten().tolist()
        return concatBlocks, concatSparsity, bsr_rhs_lengths_per_2d_plane


if __name__ == "__main__":
    # Examples of all types of attention heads

    # Causal windows with summaries
    window_size = 128
    n_windows = 5
    n_summary_blocks = 1
    blocksize2D = [16, 16]
    blocks, sparsity_mask, M = Heads.causal_windows_with_summaries(window_size, n_windows, n_summary_blocks, blocksize2D)

    # Custom user-defined pattern
    # (random block-level pattern here), note that general sub-block sparsity
    # is not supported in sparse softmax
    density = 0.3
    np.random.seed(0)
    sequence_length = 256
    pattern = Patterns.random_pattern(sequence_length, density, blocksize2D)
    blocks, sparsity_mask, M = Heads.from_custom_pattern(pattern, blocksize2D)

    # Full causal
    sequence_length = 512
    blocks, sparsity_mask, M = Heads.causal_autoregressive(sequence_length, blocksize2D)

    # Full dense self attention
    sequence_length = 128
    blocks, sparsity_mask, M = Heads.dense_self_attention(sequence_length, blocksize2D)

    # Block gram
    sequence_length = 512
    n_blocks_for_block_gram = 6
    blocks, sparsity_mask, M = Heads.block_gram(sequence_length, n_blocks_for_block_gram, blocksize2D)

    # Causal block gram
    blocks, sparsity_mask, M = Heads.causal_block_gram(sequence_length, n_blocks_for_block_gram, blocksize2D)

    # Show off some plots
    print('Making example plot')
    blocks, sparsity_mask, M = Heads.causal_windows_with_summaries(64, 10, 1, [16, 16])
    Patterns.plot_pattern(M, blocksize2D, 'test_plot_global_pattern.png', False, True)
