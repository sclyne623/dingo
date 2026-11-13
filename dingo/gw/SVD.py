import numpy as np
import pandas as pd
import scipy
import scipy.linalg
from dingo.core.dataset import DingoDataset
import cupy as cp
import math
from scipy.sparse.linalg import svds


def randomized_svd_complex(X, k, n_oversamples=15, n_iter=7, seed=0):
    """Top-k singular values and right singular vectors of X (real or complex) by
    randomized subspace iteration (Halko, Martinsson & Tropp 2011, Alg. 4.4), with
    conjugate transposes throughout and QR re-orthonormalization at every step.

    Returns s (k,) in descending order and Vh (k, X.shape[1]). X is never copied:
    X^H Q is formed as (Q^H X)^H.
    """
    rng = np.random.default_rng(seed)
    m, N = X.shape
    size = min(k + n_oversamples, m, N)
    omega = rng.standard_normal((N, size))
    if np.iscomplexobj(X):
        omega = omega + 1j * rng.standard_normal((N, size))
    omega = omega.astype(X.dtype, copy=False)
    Q, _ = scipy.linalg.qr(X @ omega, mode="economic", check_finite=False)
    for _ in range(n_iter):
        Z, _ = scipy.linalg.qr(
            (Q.conj().T @ X).conj().T, mode="economic", check_finite=False
        )
        Q, _ = scipy.linalg.qr(X @ Z, mode="economic", check_finite=False)
    B = Q.conj().T @ X  # (size, N); X ~ Q B
    _, s, Vh = scipy.linalg.svd(B, full_matrices=False, check_finite=False)
    return s[:k], Vh[:k]


def projection_mismatches(data, V, Vh, truncations, chunk_size=1000):
    """Mismatch between each row of data and its projection onto the first n basis
    vectors, for each n in truncations.

    Same formula as the original per-waveform loop (compress with V[:, :n],
    reconstruct with Vh[:n], 1 - Re<d, r> / (|d| |r|)), but evaluated in row chunks
    with matrix-matrix products, so it runs on the multithreaded BLAS instead of one
    memory-bound matrix-vector product per waveform and truncation.

    Returns an array of shape (len(truncations), len(data)).
    """
    data = np.asarray(data)
    truncations = [int(n) for n in truncations]
    n_max = max(truncations)
    out = np.empty((len(truncations), len(data)))
    for start in range(0, len(data), chunk_size):
        d = data[start : start + chunk_size]
        norm1 = np.sqrt(np.sum(np.abs(d) ** 2, axis=1))
        d_conj = d.conj()
        # Columns of a matrix product are independent, so (d @ V)[:, :n] is d @ V[:, :n].
        compressed = d @ V[:, :n_max]
        for j, n in enumerate(truncations):
            reconstructed = compressed[:, :n] @ Vh[:n]
            norm2 = np.sqrt(np.sum(np.abs(reconstructed) ** 2, axis=1))
            inner = np.sum(d_conj * reconstructed, axis=1).real
            out[j, start : start + len(d)] = 1 - inner / (norm1 * norm2)
    return out


class SVDBasis(DingoDataset):

    dataset_type = "svd_basis"

    def __init__(
        self,
        file_name=None,
        dictionary=None,
    ):
        self.V = None
        self.Vh = None
        self.s = None
        self.n = None
        self.mismatches = None
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["V", "s", "mismatches"],
        )

    def generate_basis(
        self, training_data: np.ndarray, n: int, method: str = "random", n_iter: int = 30
    ):
        """Generate the SVD basis from training data and store it.

        The SVD decomposition takes

        training_data = U @ diag(s) @ Vh

        where U and Vh are unitary.

        Parameters
        ----------
        training_data: np.ndarray
            Array of waveform data on the physical domain
        n: int
            Number of basis elements to keep.
            n=0 keeps all basis elements.
        method: str
            Select SVD method, e.g. 'random', 'svds', or 'scipy'.
        n_iter: int
            Power iterations for method='random' (ignored otherwise). Each one costs
            two passes over training_data; 30 is the historical default.
        """
        if method == "random":
            if n == 0:
                n = min(training_data.shape)

            # Not sklearn's randomized_svd: it projects with Q.T instead of Q^H, which
            # is wrong for complex data (wrong singular values, partly wrong basis).
            s, Vh = randomized_svd_complex(
                training_data, n, n_oversamples=15, n_iter=n_iter, seed=0
            )

            self.Vh = Vh.astype(np.complex128)  # TODO: fix types
            self.V = self.Vh.T.conj()
            self.n = n
            self.s = s
        elif method == "scipy":
            # Code below uses scipy's svd tool. Likely slower.
            try:
                U, s, Vh = scipy.linalg.svd(training_data, full_matrices=False)
            except:
                U, s, Vh = svds(training_data, k=n)
                # svds returns s in ascending order; reverse to match full SVD behavior
                idx = s.argsort()[::-1]
                s = s[idx]
                U = U[:, idx]
                Vh = Vh[idx, :]
            V = Vh.T.conj()

            if (n == 0) or (n > len(V)):
                self.V = V
                self.Vh = Vh
            else:
                self.V = V[:, :n]
                self.Vh = Vh[:n, :]

            self.n = len(self.Vh)
            self.s = s
        elif method == "svds":
            max_rank = min(training_data.shape) - 1
            if max_rank < 1:
                raise ValueError(
                    "Training data shape is too small for svds. "
                    f"Got shape={training_data.shape}."
                )
            if n == 0:
                k = max_rank
            else:
                k = min(n, max_rank)

            _, s, Vh = svds(training_data, k=k)
            # svds returns singular values in ascending order.
            idx = s.argsort()[::-1]
            s = s[idx]
            Vh = Vh[idx, :]
            V = Vh.T.conj()

            self.V = V
            self.Vh = Vh
            self.n = len(self.Vh)
            self.s = s
        elif method == "cupy":
            print("Using CuPy for SVD on GPU.")
            
            # 1. Move data from CPU (NumPy) to GPU (CuPy)
            training_data_gpu = cupy.asarray(training_data)

            # 2. Perform SVD on the GPU
            U_gpu, s_gpu, Vh_gpu = cupy.linalg.svd(training_data_gpu, full_matrices=False)

            # 3. Move results back to CPU (NumPy)

            s = cupy.asnumpy(s_gpu)
            Vh = cupy.asnumpy(Vh_gpu)
            V = Vh.T.conj()


            # Truncate the basis to the desired size `n` (same logic as scipy)
            if (n == 0) or (n > len(V)):
                self.V = V
                self.Vh = Vh
            else:
                self.V = V[:, :n]
                self.Vh = Vh[:n, :]
            
            self.n = len(self.Vh)
            self.s = s
        elif method == "cupy-tsqr":
            batch_size = 5000
            print(f"Using manual Tall-Skinny QR on GPU with batch size {batch_size}.")
            
            # ... (The QR decomposition loop remains the same) ...
            num_batches = math.ceil(training_data.shape[0] / batch_size)
            batches = np.array_split(training_data, num_batches)
            
            first_batch_gpu = cp.asarray(batches[0])
            r_matrix = cp.linalg.qr(first_batch_gpu, mode='r')

            for i, batch in enumerate(batches[1:]):
                print(f"  Processing update batch {i+2}/{num_batches}...")
                batch_gpu = cp.asarray(batch)
                combined_matrix = cp.vstack((r_matrix, batch_gpu))
                r_matrix = cp.linalg.qr(combined_matrix, mode='r')

            print("  Moving final R matrix back to CPU...")
            r_matrix_cpu = cp.asnumpy(r_matrix)
            del r_matrix, first_batch_gpu, batch_gpu, combined_matrix
            cp.get_default_memory_pool().free_all_blocks()
            
            # === THE FIX: Use memory-efficient `svds` for the final step ===
            print(f"  Performing final SVD for top {n} components on CPU using svds...")
            # We ask for n components, which is the final truncated size of the basis.
            # k must be less than min(shape)-1, which n=600 will be.
            _, s_unordered, Vh_unordered = svds(r_matrix_cpu, k=n)

            # CRITICAL: svds does not sort the results. We must do it manually.
            sort_indices = np.argsort(s_unordered)[::-1]
            s = s_unordered[sort_indices]
            Vh = Vh_unordered[sort_indices, :]
            
            # The results are already NumPy arrays.
            print("  SVD on R matrix complete.")
            self.s = s
            self.Vh = Vh
            self.V = self.Vh.T.conj()
            self.n = len(self.Vh) # The size is now n

        else:
            raise ValueError(f"Unsupported SVD method: {method}.")

        

    def compute_test_mismatches(
        self,
        data: np.ndarray,
        parameters: pd.DataFrame = None,
        increment: int = 50,
        verbose: bool = False,
    ):
        """
        Test SVD basis by computing mismatches of compressed / decompressed data
        against original data. Results are saved as a DataFrame.

        Parameters
        ----------
        data : np.ndarray
            Array of data sets to validate against.
        parameters : pd.DataFrame
            Optional labels for the data sets. This is useful for checking performance on
            particular regions of the parameter space.
        increment : int
            Specifies SVD truncations for computing mismatches. E.g., increment = 50
            means that the SVD will be truncated at size [50, 100, 150, ..., len(data)].
        verbose : bool
            Whether to print summary statistics.
        """
        if len(data) != len(parameters):
            raise ValueError(
                f"Incompatible data: len(data) == {len(data)} and len("
                f"parameters) == {len(parameters)} do not match."
            )
        if parameters is not None:
            self.mismatches = parameters.copy()
        else:
            self.mismatches = pd.DataFrame()

        truncations = np.append(np.arange(increment, self.n, increment), self.n)
        mismatches = projection_mismatches(data, self.V, self.Vh, truncations)
        for n, m in zip(truncations, mismatches):
            self.mismatches[f"mismatch n={n}"] = m

        if verbose:
            self.print_validation_summary()

    def print_validation_summary(self):
        """
        Print a summary of the validation mismatches.
        """
        if self.mismatches is not None:
            for col in self.mismatches:
                if "mismatch" in col:
                    n = int(col.split(sep="=")[-1])
                    mismatches = self.mismatches[col]
                    print(f"n = {n}")
                    print("  Mean mismatch = {}".format(np.mean(mismatches)))
                    print("  Standard deviation = {}".format(np.std(mismatches)))
                    print("  Max mismatch = {}".format(np.max(mismatches)))
                    print("  Median mismatch = {}".format(np.median(mismatches)))
                    print("  Percentiles:")
                    print("    99    -> {}".format(np.percentile(mismatches, 99)))
                    print("    99.9  -> {}".format(np.percentile(mismatches, 99.9)))
                    print("    99.99 -> {}".format(np.percentile(mismatches, 99.99)))

    def decompress(self, coefficients: np.ndarray):
        """
        Convert from basis coefficients back to raw data representation.

        Parameters
        ----------
        coefficients : np.ndarray
            Array of basis coefficients

        Returns
        -------
        array of decompressed data
        """
        return coefficients @ self.Vh

    def compress(self, data: np.ndarray):
        """
        Convert from data (e.g., frequency series) to compressed representation in
        terms of basis coefficients.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        array of basis coefficients
        """
        return data @ self.V

    def from_file(self, filename):
        """
        Load the SVD basis from a HDF5 file.

        Parameters
        ----------
        filename : str
        """
        super().from_file(filename)
        if self.V is None:
            raise KeyError("File does not contain SVD V matrix. No SVD basis to load.")
        self.Vh = self.V.T.conj()
        self.n = self.V.shape[1]

    def from_dictionary(self, dictionary: dict):
        """
        Load the SVD basis from a dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary should contain at least a 'V' key, and optionally an 's' key.
        """
        super().from_dictionary(dictionary)
        if self.V is None:
            raise KeyError("dict does not contain SVD V matrix. No SVD basis to load.")
        self.Vh = self.V.T.conj()
        self.n = self.V.shape[1]

    # def truncate(self, n: int):
    #     """
    #     Truncate size of SVD.
    #
    #     Parameters
    #     ----------
    #     n : int
    #         New SVD size. Should be less than current size.
    #     """
    #     if n > self.n or n < 0:
    #         print(f"Cannot truncate SVD from size n={self.n} to n={n}.")
    #     else:
    #         self.V = self.V[:, :n]
    #         self.Vh = self.Vh[:n, :]
    #         self.s = self.s[:n]
    #         self.n = n


class ApplySVD(object):
    """Transform operator for applying an SVD compression / decompression."""

    def __init__(self, svd_basis: SVDBasis, inverse: bool = False):
        """
        Parameters
        ----------
        svd_basis : SVDBasis
        inverse : bool
            Whether to apply for the forward (compression) or inverse (decompression)
            transform. Default: False.
        """
        self.svd_basis = svd_basis
        self.inverse = inverse

    def __call__(self, waveform: dict):
        """
        Parameters
        ----------
        waveform : dict
            Values should be arrays containing waveforms to be transformed.

        Returns
        -------
        dict of the same form as the input, but with transformed waveforms.
        """
        if not self.inverse:
            func = self.svd_basis.compress
        else:
            func = self.svd_basis.decompress
        return {k: func(v) for k, v in waveform.items()}
