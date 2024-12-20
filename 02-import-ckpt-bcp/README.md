# Convert HF checkpoint to Nemo-2.0 checkpoint

Atm very simple, just view the example in `scratch.sh`. Note that this `.sh` script is meant for
viewing only, and not for execution. I just used `.sh` to get syntax highlighting.

Please note to review `convert-ckpt-short.py` and choose which model to import. Atm, this file
hardcodes the model name. From smallest to largest models, the order is gemma2_2b, llama3_8b,
llama3_70b.

## Appendix

`*.log.gz` are intentionally versioned sample log (but compressed, to save space). VScode users may
use the [hyunkyunmoon.gzipdecompressor](https://github.com/hyeongyun0916/GZIP_Decompressor)
extension to quickly view it with 1-2 clicks.
