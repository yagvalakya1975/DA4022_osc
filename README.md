# Setup

From the repository root, create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install --upgrade "numpy<2" "pandas<3" "torch==2.2.2" "transformers==4.57.6" "datasets==3.6.0" "pyarrow==19.0.1" pytest
```

These versions are pinned to work together on Intel Macs: PyTorch 2.2.2 is the available wheel, Transformers 4.57.6 supports PyTorch 2.2, and the NumPy/Pandas/PyArrow pins avoid incompatible binary builds.

## Run calibration

From the repository root, run:

```bash
python run_calibration.py
```

This loads the pinned Qwen2.5-0.5B checkpoint, streams four seeded 2,048-token windows from the pinned English C4 train split, and saves the activation-column energy dictionary to `qwen25_05b_c4_activation_energy.pt`. The default seed is `42`. The script uses CUDA when available and CPU otherwise; it keeps the model in FP32 for calibration.

You can change the settings or output path, for example:

```bash
python run_calibration.py --num-sequences 8 --seed 7 --device cpu --output artifacts/c4_energy.pt
```

The first run downloads the model and reads calibration documents from Hugging Face, so it needs internet access. The model needs several GB of memory; CPU execution can be slow. The saved file maps each eligible layer name to its FP32 CPU energy vector. The script also checks that the energies match all eligible layers and prints the target count (168 for Qwen2.5-0.5B).

If the model download stays at `0%` or reports a read timeout, stop it with `Ctrl+C` and retry with Xet disabled and a longer read timeout:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 python run_calibration.py
```

This produces the calibration handoff for a quantizer. It does not fit AAAC quantization parameters or save a quantized model checkpoint; that step is not implemented in this repository yet.

## Run in Google Colab

Open [`calibration_colab.ipynb`](calibration_colab.ipynb) in Google Colab, select a GPU runtime, and run the cells in order. The notebook is self-contained, uses the same pinned model and C4 revisions as the script, and downloads `qwen25_05b_c4_activation_energy.pt` at the end.

## Run tests

```bash
python -m pytest
```
