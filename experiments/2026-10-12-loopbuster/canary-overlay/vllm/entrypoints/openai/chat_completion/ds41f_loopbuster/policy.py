"""One narrow sustained-confirmation policy, selected on disclosed replays.

No phrase blacklist or forced/injected trigger. It requires five qualifying
600-word windows, evaluated 256 words apart: 1,024 further words must pass after
the first suspicious window. This is not held-out false-positive calibration.
"""
try:
    from .collapse import CollapseConfig, CollapseDetector
except ImportError:
    from collapse import CollapseConfig, CollapseDetector


class SustainedCollapse(CollapseDetector):
    def __init__(self):
        super().__init__(CollapseConfig(every=256, consecutive=5))
