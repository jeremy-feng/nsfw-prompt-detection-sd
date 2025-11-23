import pickle
import sys

import tensorflow as tf

# Patch the missing module to allow pickle loading
sys.modules["keras.preprocessing.text"] = tf.keras.preprocessing.text

# Load legacy pickle
with open("nsfw_classifier_tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Save as JSON
with open("../tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())
