# C-MAE-MAG-Eye

This package contains the ET-HAE Conditional Masked Autoencoder extension for EyeBench.

The implementation follows the Masked Autoencoder training idea from Meta's
`facebookresearch/mae` repository, but is rewritten for EyeBench's token-aligned
gaze tensors rather than copied from the image-patch implementation.

Reference: https://github.com/facebookresearch/mae

