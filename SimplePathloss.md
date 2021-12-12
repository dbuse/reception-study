---
jupyter:
  jupytext:
    comment_magics: true
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# An Overview of Simple Path Loss and 11p Range

```python
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy

sns.set_theme(style='whitegrid')
```

```python
def simple_path_loss(wavelength, distance, alpha):
    return (wavelength  / (4 * np.pi * distance)) ** alpha
```

```python
def to_wavelength(frequency):
    """Return wavelength in meters for frequency in Hz"""
    return 299792458.0 / frequency
```

```python
def mW2dBm(mW):
    """Milli-Watt to dBm"""
    return 10 * np.log10(mW)

def dBm2mW(dBm):
    """dBm to Milli-Watt"""
    return 10 ** (dBm / 10)

assert all(math.isclose(mW, dBm2mW(mW2dBm(mW))) for mW in [20, 200, 2000, 20000])
```

## Pure Free Space Path Loss

```python
center_frequency = 5.89e9  # 5.89 GHz, center frequency of the IEEE 801.11p control channel
alpha = 2.0
max_distance = 5000.0
```

```python
distances = pd.Series(np.linspace(1, max_distance, 1000), name="distance (m)")
path_loss_mW = pd.Series(simple_path_loss(to_wavelength(center_frequency), distances, alpha), name="path loss (mW)")
path_loss_dBm = pd.Series(mW2dBm(path_loss_mW), name="path loss (dBm)")

fig, (left, right) = plt.subplots(1, 2, tight_layout=True, figsize=(12, 6))
sns.lineplot(x=distances, y=path_loss_mW, ax=left)
sns.lineplot(x=distances, y=path_loss_dBm, ax=right)
left.set(yscale="log")
```

### Free Space Path Loss with Higher *alpha* Values

```python
multi_alpha_path_loss = (
    pd.DataFrame(
        {
            f"{cur_alpha:.1f}": mW2dBm(simple_path_loss(to_wavelength(center_frequency), distances, cur_alpha))
            for cur_alpha in [2, 2.2, 2.5, 3.0, 3.5]
        }
    )
    .assign(distance=distances)
    .melt(id_vars=["distance"], var_name="alpha", value_name="path loss (dBm)")
)
fig, ax = plt.subplots(tight_layout=True, figsize=(12, 6))
sns.lineplot(data=multi_alpha_path_loss, x="distance", y="path loss (dBm)", hue="alpha", ax=ax)
ax.set(xscale="log")
```

## With transceiver config

```python
minPowerLevel_dBm = -98
noiseLevel_dBm = -98

transmit_powers_mW = [20, 100, 200, 1000, 2000]

def snr(signal_mW):
    return signal_mW / dBm2mW(noiseLevel_dBm)
```

```python
mW2dBm(pd.Series(transmit_powers_mW, index=transmit_powers_mW))
```

```python
rss = pd.DataFrame({
    f"{transmit_power_mW} mW": mW2dBm(transmit_power_mW * path_loss_mW)
    for transmit_power_mW in transmit_powers_mW
}).set_index(distances)
```

```python
fig, (left, mid, right) = plt.subplots(1, 3, figsize=(16, 6), tight_layout=True)
sns.lineplot(
    data=rss.reset_index().melt(id_vars="distance (m)", var_name="transmit power", value_name="received power (dBm)"),
    y="received power (dBm)",
    x="distance (m)",
    hue="transmit power",
    ax=left,
)
sns.lineplot(
    data=rss.pipe(dBm2mW).pipe(snr).reset_index().melt(id_vars="distance (m)", var_name="transmit power", value_name="Signal to Noise Ratio"),
    y="Signal to Noise Ratio",
    x="distance (m)",
    hue="transmit power",
    ax=mid,
)
sns.lineplot(
    data=rss.pipe(dBm2mW).pipe(snr).pipe(mW2dBm).reset_index().melt(id_vars="distance (m)", var_name="transmit power", value_name="Signal to Noise Ratio (in dB)"),
    y="Signal to Noise Ratio (in dB)",
    x="distance (m)",
    hue="transmit power",
    ax=right,
)
left.hlines(y=[minPowerLevel_dBm], xmin=distances.iloc[0], xmax=distances.iloc[-1], colors=["grey"], linestyles=["dashed"])
mid.set(yscale="log")
```
