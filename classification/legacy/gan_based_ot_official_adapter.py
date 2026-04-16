# -*- coding: utf-8 -*-
"""
Official GAN-based OT adapter.

You should replace this stub with the official implementation you choose.
Required API:
    run_gan_based_ot_prob_map(adata_sub, target_classes, source_time_num, target_time_num)
Return:
    dict[cell_id(str)] -> np.ndarray shape (n_classes,)
"""


def run_gan_based_ot_prob_map(adata_sub, target_classes, source_time_num, target_time_num):
    try:
        import ganot  # noqa: F401
    except Exception as e:
        raise NotImplementedError(
            "Official GAN-based OT backend is not configured yet. "
            "No supported package detected (e.g., `ganot`). "
            "Please install the official implementation and wire this adapter."
        ) from e
    raise NotImplementedError(
        "Detected `ganot`, but adapter mapping is not implemented yet. "
        "Please implement run_gan_based_ot_prob_map with the official API."
    )
