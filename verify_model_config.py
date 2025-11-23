from src.model.model_loader import load_gemma_model, HF_TOKEN
import inspect

sig = inspect.signature(load_gemma_model)
default_model = sig.parameters['model_id'].default
print(f"Default model ID: {default_model}")

if default_model == "google/gemma-2b":
    print("Model ID verification passed.")
else:
    print(f"Model ID verification failed. Expected 'google/gemma-2b', got '{default_model}'")

if HF_TOKEN == "Yhf_cpzqRjgvmRvgVCLaqVgoEPoCbGkKHegVGP":
    print("HF_TOKEN placeholder verification passed.")
else:
    print(f"HF_TOKEN placeholder verification failed. Got '{HF_TOKEN}'")
