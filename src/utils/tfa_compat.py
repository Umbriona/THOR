"""
Optional `tensorflow_addons`.

THOR does not use tensorflow-addons at runtime. The only code that constructs
tfa layers is `CycleGan.load_models()`, whose single call site is commented out
(`models_cyclegan_self_loss.py` line 49), so `tfa.layers.InstanceNormalization`
is never actually built. Spectral normalisation comes from THOR's own vendored
`utils.layers_new.SpectralNormalization`, not from tfa.

The package was nevertheless imported at module level by several model modules,
which made them unimportable wherever tfa has no wheels. That matters in two
ways:

  * tensorflow-addons publishes **no aarch64 wheels**, so the package cannot be
    installed on ARM machines at all - including Grace/ARM GPU nodes.
  * tensorflow-addons reached end of life in May 2024 and will not be updated.

Importing `tfa` from this module yields the real package when it is installed,
and otherwise a placeholder that raises a clear, actionable error if anything
ever touches it.
"""

try:  # pragma: no cover - depends on the platform
    import tensorflow_addons as tfa  # noqa: F401

except ImportError:  # pragma: no cover

    class _TensorFlowAddonsUnavailable:
        """Stand-in that fails loudly only if tfa is genuinely used."""

        __slots__ = ()

        def __getattr__(self, name):
            raise ImportError(
                f"tensorflow_addons.{name} was requested, but tensorflow-addons "
                "is not installed.\n"
                "\n"
                "THOR does not need it: training and generation never construct "
                "tfa layers. If you reach this error you have re-enabled "
                "CycleGan.load_models() or added new code that uses tfa.\n"
                "\n"
                "On x86_64 you can install it with:\n"
                "    pip install tensorflow-addons==0.23.0\n"
                "On aarch64 no wheels exist and the project is end-of-life; use "
                "tf.keras.layers.GroupNormalization(groups=-1) in place of "
                "tfa.layers.InstanceNormalization, and "
                "utils.layers_new.SpectralNormalization in place of "
                "tfa.layers.SpectralNormalization."
            )

        def __repr__(self):
            return "<tensorflow_addons unavailable (not installed)>"

    tfa = _TensorFlowAddonsUnavailable()
