import json
import re
from typing import List, Sequence, Tuple, Union

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

# Configuration
MAX_SEQUENCE_LENGTH: int = 50


def load_resources(tokenizer_path: str, model_path: str) -> Tuple[Tokenizer, Model]:
    """Load tokenizer and Keras model for inference.

    Args:
        tokenizer_path: Path to the saved tokenizer JSON.
        model_path: Path to the serialized Keras model.

    Returns:
        A tuple containing the reconstructed tokenizer and compiled model.
    """
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        # Depending on how it was saved, we might need json.dumps or pass the dict directly
        # tokenizer_from_json expects a JSON string
        tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

    # Load the clean Keras model
    model = load_model(model_path)
    return tokenizer, model


def preprocess(
    text: Union[str, List[str]], is_first: bool = True
) -> Union[str, List[str]]:
    """Normalize prompt text using the project-specific rules.

    Args:
        text: Single prompt or list of prompts to normalize.
        is_first: Internal flag to signal recursive processing for nested segments.

    Returns:
        Cleaned text string or list of cleaned strings matching the input structure.
    """
    if is_first:
        if isinstance(text, str):
            pass
        elif isinstance(text, list):
            output = []
            for i in text:
                output.append(preprocess(i))
            return output

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Normalize parentheses
    text = re.sub(r"\(+", "(", text)
    text = re.sub(r"\)+", ")", text)

    # Handle weights in prompts, e.g., (word:1.2)
    matches = re.findall(r"\(.*?\)", text)
    for match in matches:
        text = text.replace(match, preprocess(match[1:-1], is_first=False))

    # Normalize separators
    text = text.replace("\n", ",").replace("|", ",")

    # Clean up whitespace and empty strings
    if is_first:
        output = text.split(",")
        output = list(map(lambda x: x.strip(), output))
        output = [x for x in output if x != ""]
        return ", ".join(output)

    return text


def format_output(
    prompts: Sequence[str],
    negative_prompts: Sequence[str],
    predictions: Sequence[Tuple[str, float]],
) -> None:
    """Pretty-print prompts alongside their predictions.

    Args:
        prompts: Original prompts.
        negative_prompts: Paired negative prompts.
        predictions: Sequence of (label, confidence percentage) tuples.
    """
    for idx, prompt in enumerate(prompts):
        label, confidence = predictions[idx]
        print("*" * 65)
        print(f"Prompt: {prompt}")
        print(f"Negative Prompt: {negative_prompts[idx]}")
        print(f"Prediction: {label} -- {confidence}%")


def main() -> None:
    """Run an end-to-end inference demo using saved tokenizer and model."""
    # Paths to your new files
    tokenizer_path = "tokenizer.json"
    model_path = "nsfw_classifier.keras"

    print("Loading model and tokenizer...")
    tokenizer, model = load_resources(tokenizer_path, model_path)

    # Test data
    prompts = [
        "a landscape with trees and mountains in the background",
        "nude, sexy, 1girl, nsfw",
    ]
    negative_prompts = ["nsfw", "worst quality"]

    print("Preprocessing inputs...")
    # Preprocess text using the specific regex rules
    processed_prompts = preprocess(prompts)
    processed_neg_prompts = preprocess(negative_prompts)

    # Convert text to sequences
    x_seq = tokenizer.texts_to_sequences(processed_prompts)
    z_seq = tokenizer.texts_to_sequences(processed_neg_prompts)

    # Pad sequences
    x_pad = pad_sequences(x_seq, maxlen=MAX_SEQUENCE_LENGTH)
    z_pad = pad_sequences(z_seq, maxlen=MAX_SEQUENCE_LENGTH)

    print("Running prediction...")
    # Model expects a list of [prompt, negative_prompt]
    raw_preds = model.predict([x_pad, z_pad])

    # Process results
    # Threshold is 0.5: > 0.5 is NSFW, <= 0.5 is SFW
    final_results = []
    for score in raw_preds:
        val = score[0]
        if val > 0.5:
            final_results.append(("NSFW", float("{:.2f}".format(val * 100))))
        else:
            final_results.append(("SFW", float("{:.2f}".format(100 - val * 100))))

    print("Results:")
    format_output(prompts, negative_prompts, final_results)


if __name__ == "__main__":
    main()
