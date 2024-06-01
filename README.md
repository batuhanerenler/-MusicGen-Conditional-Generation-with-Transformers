
# MusicGen Conditional Generation with Transformers

This project demonstrates how to use the `transformers` library to generate music conditionally using the `MusicgenForConditionalGeneration` model. The script provides a step-by-step guide to load the model, generate music based on a text description, and handle various issues such as float precision, chunked generation, and silence detection.

## Requirements

- Python 3.7+
- `torch`
- `transformers`
- `numpy`
- `scipy`
- `tqdm`
- `IPython`

## Setup

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/your-username/musicgen-conditional-generation.git
   cd musicgen-conditional-generation
   ```

2. Install the required dependencies:

   ```bash
   pip install torch transformers numpy scipy tqdm ipython
   ```

3. Ensure that your environment supports GPU acceleration (optional but recommended).

## Usage

1. Open the script `musicgen_conditional_generation.py` and customize the input text description and desired length of the music.

2. Run the script:

   ```bash
   python musicgen_conditional_generation.py
   ```

### Example

Here is an example of how to run the script with a specific text description and desired music length:

```python
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from IPython.display import Audio, display
import scipy.io.wavfile as wavfile
import numpy as np
from tqdm import tqdm
import os
import time

# Check if GPU is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# Set environment variable to disable flash attention if needed
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the small model and processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Use bfloat16 to avoid issues with float16 precision
model = model.to(torch.bfloat16)

# Input text description
inputs = processor(text=["a dark and heroic theme with violin"], return_tensors="pt")

# Move tensors to GPU if available
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Set pad_token_id and eos_token_id if not already set
if model.config.pad_token_id is None:
    model.config.pad_token_id = processor.tokenizer.pad_token_id

if model.config.eos_token_id is None:
    model.config.eos_token_id = processor.tokenizer.eos_token_id

# User input: number of seconds for the music
desired_seconds = 30  # change this value as needed

# Calculate the number of tokens
tokens_per_second = 60.0  # assuming 60 tokens per second as given in the original information
max_new_tokens = int(desired_seconds * tokens_per_second)

chunk_size = 750  # Generate music in chunks if needed
num_chunks = (max_new_tokens // chunk_size) if max_new_tokens > 4000 else 1

# Estimate the production time per chunk
time_per_chunk_seconds = 31  # given information
total_estimated_time_seconds = (max_new_tokens / chunk_size) * time_per_chunk_seconds
total_estimated_time_minutes = total_estimated_time_seconds / 60

print(f"Estimated production time: {total_estimated_time_minutes:.2f} minutes.")

audio_chunks = []

start_time = time.time()

if num_chunks > 1:
    pbar = tqdm(total=num_chunks, desc="Generating Music", unit="chunk")
    for i in range(num_chunks):
        chunk_start_time = time.time()
        with torch.no_grad():
            model_kwargs = {
                "attention_mask": inputs['input_ids'].ne(model.config.pad_token_id).long().to(device)
            }
            try:
                current_max_tokens = chunk_size if max_new_tokens > 4000 else max_new_tokens
                audio_values = model.generate(
                    inputs['input_ids'], 
                    do_sample=True, 
                    guidance_scale=5,  # Increase the guidance scale
                    max_new_tokens=current_max_tokens, 
                    temperature=0.8,  # Adjust the temperature
                    top_k=50, 
                    top_p=0.95, 
                    **model_kwargs
                )
                # Convert BFloat16 to Float32 before moving to CPU
                audio_chunk = audio_values[0].to(torch.float32).cpu().numpy().ravel()

                # Check for NaN or Inf values
                if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
                    raise ValueError("Generated audio contains NaN or Inf values.")

                # Check if the chunk contains mostly silence
                if np.mean(np.abs(audio_chunk)) < 0.01:
                    print(f"Warning: Chunk {i+1} contains mostly silence.")

                audio_chunks.append(audio_chunk)
            except RuntimeError as e:
                print(f"RuntimeError during generation: {e}")
                break
        
        # Update progress bar and estimated time remaining
        chunk_end_time = time.time()
        elapsed_time = chunk_end_time - chunk_start_time
        estimated_remaining_time = (num_chunks - (i + 1)) * elapsed_time
        pbar.set_postfix({'ETA': f'{estimated_remaining_time / 60:.2f} min'})
        pbar.update(1)
    pbar.close()
else:
    with torch.no_grad():
        model_kwargs = {
            "attention_mask": inputs['input_ids'].ne(model.config.pad_token_id).long().to(device)
        }
        try:
            audio_values = model.generate(
                inputs['input_ids'], 
                do_sample=True, 
                guidance_scale=8,  # Increase the guidance scale
                max_new_tokens=max_new_tokens, 
                temperature=0.8,  # Adjust the temperature
                top_k=50, 
                top_p=0.95, 
                **model_kwargs
            )
            # Convert BFloat16 to Float32 before moving to CPU
            audio_chunk = audio_values[0].to(torch.float32).cpu().numpy().ravel()

            # Check for NaN or Inf values
            if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
                raise ValueError("Generated audio contains NaN or Inf values.")

            # Check if the chunk contains mostly silence
            if np.mean(np.abs(audio_chunk)) < 0.01:
                print("Warning: Generated audio contains mostly silence.")

            audio_chunks.append(audio_chunk)
        except RuntimeError as e:
            print(f"RuntimeError during generation: {e}")

# Concatenate all audio chunks
if audio_chunks:
    audio = np.concatenate(audio_chunks)

    # Ensure the audio data type is appropriate
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)

    # Define the sampling rate
    sampling_rate = 16000  # Replace with the correct sampling rate from your model

    # Trim the audio to the desired length
    desired_length = int(desired_seconds * sampling_rate)
    audio = audio[:desired_length]

    # Save to WAV file
    wavfile.write("musicgen_output.wav", rate=sampling_rate, data=audio)

    # Play the generated music
    display(Audio(audio, rate=sampling_rate))
else:
    print("No audio was generated due to errors.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Actual production time: {elapsed_time / 60:.2f} minutes.")
```

### Customizing the Input

You can customize the input text description to generate different types of music. For example, to generate a happy and upbeat theme with a piano, change the `inputs` variable:

```python
inputs = processor(text=["a happy and upbeat theme with piano"], return_tensors="pt")
```

### Handling Long Generation Times

The script includes logic to handle long generation times by breaking the music generation into chunks. This is particularly useful for generating music longer than a few seconds. Adjust the `chunk_size` variable and the `num_chunks` calculation as needed.

### Troubleshooting

- Ensure your environment has enough memory to handle the model and the music generation process.
- Check for any NaN or Inf values in the generated audio and handle them appropriately.
- If the generated audio contains mostly silence, consider adjusting the input parameters or guidance scale.

## Contributing

Feel free to open issues or submit pull requests to improve the script or add new features.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The `transformers` library by Hugging Face.
- The MusicGen model by Facebook.

---

This readme file provides a comprehensive overview of the project, including setup instructions, usage examples, and troubleshooting tips. Adjust the input parameters and text description as needed to suit your specific requirements.
