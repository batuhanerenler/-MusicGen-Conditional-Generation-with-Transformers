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
inputs = processor(text=["50s American Style Pop"], return_tensors="pt")

# Move tensors to GPU if available
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Set pad_token_id and eos_token_id if not already set
if model.config.pad_token_id is None:
    model.config.pad_token_id = processor.tokenizer.pad_token_id

if model.config.eos_token_id is None:
    model.config.eos_token_id = processor.tokenizer.eos_token_id

# User input: number of seconds for the music
desired_seconds = 60  # change this value as needed

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
