from ..acquisitions import *
from ..likelihood import Likelihood


def check_acquisition(acquisition, model, inputs, surrogate_model):

    if isinstance(acquisition, str):
        acq_name = acquisition.lower()

        if acq_name == "pi":
            return PI(model, inputs)

        elif acq_name == "ei":
            return EI(model, inputs)

        elif acq_name == "us":
            return US(model, inputs, surrogate_model)

        elif acq_name == "us_bo":
            return US_BO(model, inputs)

        elif acq_name == "us_lw":
            return US_LW(model, inputs, surrogate_model)

        elif acq_name == "us_iw":
            likelihood = Likelihood(model, inputs, weight_type="nominal")
            a = US_LW(model, inputs, likelihood=likelihood)
            return a

        elif acq_name == "us_lwbo":
            return US_LWBO(model, inputs)

        elif acq_name == "lcb":
            return LCB(model, inputs)

        elif acq_name == "lcb_lw":
            return LCB_LW(model, inputs)

        elif acq_name == "ivr":
            return IVR(model, inputs)

        elif acq_name == "ivr_bo":
            return IVR_BO(model, inputs)

        elif acq_name == "ivr_iw":
            likelihood = Likelihood(model, inputs, weight_type="nominal")
            a = IVR_LW(model, inputs, likelihood=likelihood)
            return a

        elif acq_name == "ivr_lw":
            return IVR_LW(model, inputs)

        elif acq_name == "ivr_lwbo":
            return IVR_LWBO(model, inputs)

        elif acq_name == "us_lwraw":
            likelihood = Likelihood(model, inputs, fit_gmm=False)
            a = US_LW(model, inputs, likelihood=likelihood)
            return a

        elif acq_name == "us_lwboraw":
            likelihood = Likelihood(model, inputs, fit_gmm=False)
            a = US_LWBO(model, inputs, likelihood=likelihood)
            return a

        elif acq_name == "lcb_lwraw":
            likelihood = Likelihood(model, inputs, fit_gmm=False)
            a = LCB_LW(model, inputs, likelihood=likelihood)
            return a

        else:
            raise NotImplementedError

    elif isinstance(acquisition, Acquisition):
        return acquisition

    elif issubclass(acquisition, Acquisition):
        return acquisition(model, inputs)

    else:
        raise ValueError


