# sc_filter_interp
WakenAI switched-cap filter model based on numpy and using interpolation.

---

## Example

```python
import matplotlib.pyplot as plt
import sc_filter_interp as sfi

# test signal
fs = 16000.
fsig = 310.
t_vec = np.arange(0, 1, 1/fs)
signal = np.cos(2*np.pi*(fc + 10.)*t_vec)

scfilter = sfi.SCFilter(fs, len(t_vec), fc=sfok.CHIP_DFLT_FC, fbw=sfok.CHIP_DFLT_FBW)
output, t_vec_out = scfilter(signal)

plt.figure()
[plt.plot(t, out) for t, out in zip(t_vec_out, output)]
plt.show()
```
