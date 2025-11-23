# NSFW Prompt Detection for SD

This repository is a fork of [thefcraft/nsfw-prompt-detection-sd](https://github.com/thefcraft/nsfw-prompt-detection-sd).

The original project provides a Keras model to detect NSFW content in Stable Diffusion prompts. However, due to significant changes in the TensorFlow/Keras ecosystem (specifically the transition from Keras 2 to Keras 3), the tokenizer (`.pickle`) and the original model files (`.pickle`/`.h5`) do not work out-of-the-box in modern environments.

**This fork fixes compatibility issues and provides converted files in modern, standard formats (`.json` and `.keras`).**

## The Problem

If you try to load the original files in a modern environment (TensorFlow 2.16+ / Keras 3+), you will encounter several fatal errors:

**1. Tokenizer Pickle Issues:**
The `nsfw_classifier_tokenizer.pickle` fails to load because it references `keras.preprocessing.text`, which has been moved/refactored in newer TensorFlow versions.

```text
ModuleNotFoundError: No module named 'keras.preprocessing.text'
```

**2. Model Pickle Issues:**
The `nsfw_classifier.pickle` relies on internal Keras paths (`keras.saving.pickle_utils`) that have been removed.

**3. Deprecated Model Arguments (`.h5` file):**
The original `.h5` model used an LSTM layer with the `time_major` argument. This argument has been removed in Keras 3.

```text
ValueError: Unrecognized keyword arguments passed to LSTM: {'time_major': False}
```

## The Solution

I have processed the original weights and tokenizer, converting them to version-agnostic standard formats.

**Key Changes:**

1.  **Tokenizer Conversion:** Converted the Pickle-based tokenizer to a standard **`tokenizer.json`** format.
    - JSON removes dependencies on specific Python class paths, making it safe to use across different TensorFlow/Keras versions.
2.  **Model Conversion:** Converted the model to the standard **`nsfw_classifier.keras`** format.
    - Implemented a `CompatibleLSTM` class to intercept and strip the deprecated `time_major` argument during conversion.

## Migration Scripts

If you are interested in how the fix was applied, here are the scripts used for conversion:

- [fix_tokenizer.py](scripts/fix_tokenizer.py)
- [fix_model.py](scripts/fix_model.py)

## Quick Start

You do not need to install legacy TensorFlow versions or apply any patches. Simply use the new files provided in this repo.

### Requirements

- Python 3.10+
- TensorFlow 2.16+ (or any version running Keras 3)

### Usage Example

- [inference_demo.py](./inference_demo.py)
