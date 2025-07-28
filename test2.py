import transformers
print(transformers.__version__)  # Confirm version
print(transformers.TrainingArguments)  # Show the class object
print(transformers.TrainingArguments.__init__)  # Show the init method

# Inspect the init signature
import inspect
print(inspect.signature(transformers.TrainingArguments.__init__))
