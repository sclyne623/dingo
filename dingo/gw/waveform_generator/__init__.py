from .waveform_generator import (
    WaveformGenerator,
    LISAWaveformGenerator,
    NewInterfaceWaveformGenerator,
    generate_waveforms_parallel,
    sum_contributions_m,
)

# BBHxWaveformGenerator is optional (requires BBHx to be installed)
try:
    from .waveform_generator import BBHxWaveformGenerator
except (ImportError, NameError):
    BBHxWaveformGenerator = None

from dingo.gw.domains import UniformFrequencyDomain
