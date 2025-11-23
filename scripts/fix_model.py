import keras


# Define a compatible class for loading legacy models
@keras.saving.register_keras_serializable()
class CompatibleLSTM(keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Remove the deprecated time_major argument from older Keras versions
        if "time_major" in kwargs:
            kwargs.pop("time_major")
        super().__init__(*args, **kwargs)


print("Loading legacy model...")

# Load legacy model that uses CompatibleLSTM
old_model = keras.models.load_model(
    "nsfw_classifier.h5", custom_objects={"LSTM": CompatibleLSTM}, compile=False
)

print("Cleaning configuration...")

# Grab the configuration dictionary
config = old_model.get_config()

# Deep-clean the configuration
for layer in config["layers"]:
    if layer["class_name"] == "CompatibleLSTM":
        print(f"   - Replacing {layer['name']} with standard LSTM")

        # Key fix: switch class name to the standard LSTM
        layer["class_name"] = "LSTM"

        # Key fix: explicitly set module to keras.layers so Keras treats it as the built-in LSTM
        layer["module"] = "keras.layers"

        # Key fix: drop registered_name to force Keras to forget the custom registration
        if "registered_name" in layer:
            del layer["registered_name"]

        # Ensure no stale time_major flag remains in the config
        if "config" in layer and "time_major" in layer["config"]:
            del layer["config"]["time_major"]

print("Rebuilding model from clean config...")

# Rebuild the model so Keras treats it as a clean standard model
new_model = keras.models.Model.from_config(config)

# Transfer weights (architecture is identical, so copy directly)
new_model.set_weights(old_model.get_weights())

# Save in the standard format
new_model.save("../nsfw_classifier.keras")
print("Success: Cleaned model saved to 'nsfw_classifier.keras'")
