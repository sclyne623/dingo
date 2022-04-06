import numpy as np
import torch
import lal
from bilby.core.prior import PriorDict
from abc import ABC, abstractmethod


class GNPEBase(ABC):
    """
    A base class for Group Equivariant Neural Posterior Estimation [1].

    This implements GNPE for *approximate* equivariances. For exact equivariances,
    additional processing should be implemented within a subclass.

    [1]: https://arxiv.org/abs/2111.13139
    """

    def __init__(self, kernel_dict, operators):
        self.kernel = PriorDict(kernel_dict)
        self.operators = operators

    @abstractmethod
    def __call__(self, input_sample):
        pass

    def apply_gnpe(self, input_parameters):
        """
        Applies a GNPE transformation. Given input parameters, perturbs based on the
        kernel to produce "proxy" ("hatted") parameters, i.e., samples

            \hat g ~ p(\hat g | g).

        Typically the GNPE NDE will be conditioned on \hat g.

        In addition, the data simplification is achieved by transforming according to
        (\hat g)^{-1}, so this is provided as well.

        Parameters:
        -----------
        input_parameters : dict
            Initial parameter values to be perturbed. dict values can be either floats
            (for training) or torch Tensors (for inference).

        Returns
        -------
        A dict of parameters that includes (a) the proxy parameters, and (b) their
        inverses (under the original parameter keys), which will subsequently used for
        transforming the data.
        """
        result = {}
        for k in self.kernel:
            if k not in input_parameters:
                raise KeyError(
                    f"Input parameters are missing key {k} required for GNPE."
                )
            g = input_parameters[k]
            g_hat = self.perturb(g, k)
            result[k + "_proxy"] = g_hat
            result[k] = self.inverse(g_hat, k)
        return result

    def perturb(self, g, k):
        """
        Generate proxy variables based on initial parameter values.

        Parameters
        ----------
        g : Union[float, torch.Tensor]
            Initial parameter values
        k : str
            Parameter name. This is used to identify the group binary operator.

        Returns
        -------
        Proxy variables in the same format as g.
        """
        # First we sample from the kernel, ensuring the correct data type,
        # and accounting for possible batching.
        #
        # Batching is implemented only for torch Tensors (expected at inference time),
        # whereas un-batched data in float form is expected during training.
        if type(g) == torch.Tensor:
            epsilon = self.kernel[k].sample(len(g))
            epsilon = torch.tensor(epsilon, dtype=g.dtype, device=g.device)
        elif type(g) == float:
            epsilon = self.kernel[k].sample()
        else:
            raise NotImplementedError(f"Unsupported data type {type(g)}.")

        return self.multiply(g, epsilon, k)

    def multiply(self, a, b, k):
        op = self.operators[k]
        if op == "+":
            return a + b
        else:
            raise NotImplementedError(
                f"Unsupported group multiplication operator: {op}"
            )

    def inverse(self, a, k):
        op = self.operators[k]
        if op == "+":
            return -a
        else:
            raise NotImplementedError(
                f"Unsupported group multiplication operator: {op}"
            )


class GNPECoalescenceTimes(GNPEBase):
    """
    GNPE [1] Transformation for detector coalescence times.

    For each of the detector coalescence times, a proxy is generated by adding a
    perturbation epsilon from the GNPE kernel to the true detector time. This proxy is
    subtracted from the detector time, such that the overall time shift only amounts to
    -epsilon in training. This standardizes the input data to the inference network,
    since the applied time shifts are always restricted to the range of the kernel.

    To preserve information at inference time, conditioning of the inference network on
    the proxies is required. To that end, the proxies are stored in sample[
    'gnpe_proxies'].

    We can enforce an exact equivariance under global time translations, by subtracting
    one proxy (by convention: the first one, usually for H1 ifo) from all other
    proxies, and from the geocent time, see [1]. This is enabled with the flag
    exact_global_equivariance.

    [1]: arxiv.org/abs/2111.13139
    """
    def __init__(
        self, ifo_list, kernel, exact_global_equivariance=True, inference=False
    ):
        """
        Parameters
        ----------
        ifo_list : bilby.gw.detector.InterferometerList
            List of interferometers.
        kernel : str
            Defines a Bilby prior, to be used for all interferometers.
        exact_global_equivariance : bool = True
            Whether to impose the exact global time translation symmetry.
        inference : bool = False
            Whether to use inference or training mode.
        """
        self.ifo_time_labels = [ifo.name + "_time" for ifo in ifo_list]
        kernel_dict = {k: kernel for k in self.ifo_time_labels}
        operators = {k: "+" for k in self.ifo_time_labels}
        super().__init__(kernel_dict, operators)

        self.inference = inference
        self.exact_global_equivariance = exact_global_equivariance

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        new_parameters = self.apply_gnpe(extrinsic_parameters)

        # If we are in training mode, we assume that the time shifting due to different
        # arrival times of the signal in individual detectors has not yet been applied
        # to the data; instead the arrival times are stored in extrinsic_parameters.
        # Hence we subtract off the proxy times from these arrival times, so that time
        # shifting of the data only has to be done once.
        if not self.inference:
            for k in self.ifo_time_labels:
                new_parameters[k] = new_parameters[k] + extrinsic_parameters[k]

        # If we are imposing the global time shift symmetry, then we treat the first
        # proxy as "preferred", in the sense that it defines the global time shift.
        # This symmetry is enforced as follows:
        #
        #    1) Do not explicitly condition the model on the preferred proxy
        #    2) Subtract the preferred proxy from geocent_time (assumed to be a regression
        #    parameter). Note that this must be undone at inference time.
        #    3) Subtract the preferred proxy from the remaining proxies. These remaining
        #    proxies then define time shifts relative to the global time shift.
        #
        # Imposing the global time shift does not impact the transformation of the
        # data: we do not change the values of the true detector coalescence times
        # stored in extrinsic_parameters, only the proxies.
        if self.exact_global_equivariance:
            dt = new_parameters.pop(self.ifo_time_labels[0] + '_proxy')
            if not self.inference:
                if "geocent_time" not in extrinsic_parameters:
                    raise KeyError(
                        "geocent_time should be in extrinsic_parameters at "
                        "this point during training."
                    )
                new_parameters["geocent_time"] = (
                    extrinsic_parameters["geocent_time"] - dt
                )
            else:
                new_parameters["geocent_time"] = -dt
            for k in self.ifo_time_labels[1:]:
                new_parameters[k + "_proxy"] -= dt

        extrinsic_parameters.update(new_parameters)
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class GNPEChirpMass(object):
    """
    GNPE [1] Transformation for chirp mass.

    Todo

    [1]: arxiv.org/abs/2111.13139
    """

    def __init__(self, frequencies, kernel_kwargs):
        """
        :param frequencies: np.array
            sample frequencies of strain data
        :param kernel_kwargs: dict
            kwargs for gnpe kernel
        :param mean: float = 0
            mean for standardization of proxy
        :param std: float = 1
            standard deviation for standardization of proxy
        """
        self.f = frequencies
        self.kernel = get_gnpe_kernel(kernel_kwargs)

    def __call__(self, input_sample):
        sample = input_sample.copy()
        # Copy extrinsic parameters to not overwrite input_sample. Does this really
        # matter?
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        # get proxy by adding perturbation from kernel to Mc
        Mc_hat = sample["parameters"]["chirp_mass"] + self.kernel()
        # convert to SI units
        Mc_SI_hat = Mc_hat * lal.GMSUN_SI

        rescaling = np.exp(
            1j
            * (3 / 4)
            * (8 * np.pi * self.f * (Mc_SI_hat / lal.C_SI ** 3)) ** (-5 / 3)
        )
        hc = sample["waveform"]["h_cross"] * rescaling
        hp = sample["waveform"]["h_plus"] * rescaling
        sample["waveform"] = {"h_cross": hc, "h_plus": hp}

        extrinsic_parameters.update({"chirp_mass_proxy": Mc_hat})
        sample["extrinsic_parameters"] = extrinsic_parameters

        # proxies_array = (np.array([Mc_hat]) - self.mean) / self.std
        # if "gnpe_proxies" in sample:
        #     sample["gnpe_proxies"] = np.concatenate(
        #         (sample["gnpe_proxies"], proxies_array)
        #     )
        # else:
        #     sample["gnpe_proxies"] = proxies_array
        return sample


def get_gnpe_kernel(kernel_kwargs):
    """
    Returns kernel from kernel_kwargs.

    :param kernel_kwargs: dict
        kernel_kwargs['type'] contains the type of the kernel (choices:
        'uniform' and 'random'). The remaining kwargs are passed to the
        corresponding numpy function.
    :return: kernel
    """
    # kernel_type = kernel_kwargs.pop('type')
    kernel_type = kernel_kwargs["type"]
    kernel_kwargs = {k: v for k, v in kernel_kwargs.items() if k != "type"}
    if kernel_type == "uniform":

        def kernel():
            return np.random.uniform(**kernel_kwargs)

        return kernel
    elif kernel_type == "normal":

        def kernel():
            return np.random.normal(**kernel_kwargs)

        return kernel
    else:
        raise NotImplementedError(f"Unknown kernel type {kernel_type}.")
