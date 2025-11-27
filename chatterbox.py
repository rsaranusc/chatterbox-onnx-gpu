#!/Users/ceo/.pyenv/shims/python
import argparse
import os
import sys
import glob
import pyperclip
import librosa
import soundfile as sf
import numpy as np
from chatterbox_onnx import ChatterboxOnnx


def chunk_text(text, chunk_size=300):
    """
    Split text into chunks of approximately chunk_size characters,
    respecting word boundaries and avoiding splits in the middle of words.
    
    Args:
        text (str): Input text to chunk
        chunk_size (int): Target size for each chunk
        
    Returns:
        list: List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # If remaining text is less than chunk_size, add it as final chunk
        if start + chunk_size >= len(text):
            chunks.append(text[start:].strip())
            break
            
        # Find the last space/punctuation before chunk_size limit
        end = start + chunk_size
        
        # Look backwards for word boundaries
        last_space = text.rfind(' ', start, end)
        last_punct = max(
            text.rfind('.', start, end),
            text.rfind('!', start, end),
            text.rfind('?', start, end)
        )
        
        # Prefer punctuation boundaries, then spaces
        if last_punct > start:
            split_point = last_punct + 1  # Include the punctuation
        elif last_space > start:
            split_point = last_space
        else:
            # No good split point found, force split at chunk_size
            split_point = end
        
        chunk = text[start:split_point].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        start = split_point
    
    return chunks


def combine_audio_chunks(chunk_files, output_file, sample_rate=24000):
    """
    Combine multiple audio chunk files into a single output file.
    
    Args:
        chunk_files (list): List of paths to audio chunk files
        output_file (str): Path for the combined output file
        sample_rate (int): Audio sample rate
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not chunk_files:
        print("No audio chunks to combine.")
        return False
    
    try:
        print(f"\n--- Combining {len(chunk_files)} audio chunks ---")
        
        # Load and concatenate all chunks
        combined_audio = []
        for i, chunk_file in enumerate(chunk_files, 1):
            print(f"Loading chunk {i}/{len(chunk_files)}: {chunk_file}")
            audio, _ = librosa.load(chunk_file, sr=sample_rate)
            combined_audio.append(audio)
        
        # Concatenate all audio arrays
        final_audio = np.concatenate(combined_audio)
        
        # Save combined audio
        sf.write(output_file, final_audio, sample_rate)
        print(f"\nSuccessfully saved combined audio to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error combining audio chunks: {e}")
        return False


def process_text_file(input_file, output_dir, synthesizer, target_voice, exaggeration, apply_wm):
    """
    Process a single text file and generate audio output.
    
    Args:
        input_file (str): Path to input text file
        output_dir (str): Directory for output audio file
        synthesizer: ChatterboxOnnx synthesizer instance
        target_voice (str): Path to target voice file
        exaggeration (float): Voice exaggeration parameter
        apply_wm (bool): Whether to apply watermark
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read text from file
        print(f"\n{'='*60}")
        print(f"Processing file: {os.path.basename(input_file)}")
        print(f"{'='*60}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text or not text.strip():
            print(f"Warning: File is empty or contains only whitespace: {input_file}")
            return False
        
        print(f"Text length: {len(text)} characters")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.wav")
        
        # Chunk the text
        chunks = chunk_text(text)
        print(f"Text split into {len(chunks)} chunk(s)")
        
        # Process chunks
        if len(chunks) == 1:
            # Single chunk - use original logic
            print("\n--- Processing single chunk ---")
            synthesizer.synthesize(
                text=chunks[0],
                target_voice_path=target_voice,
                exaggeration=exaggeration,
                output_file_name=output_file,
                apply_watermark=apply_wm
            )
        else:
            # Multiple chunks - process each separately and combine
            chunk_files = []
            
            print(f"\n--- Processing {len(chunks)} chunks ---")
            
            for i, chunk in enumerate(chunks, 1):
                chunk_output = f"temp_chunk_{i:03d}.wav"
                chunk_files.append(chunk_output)
                
                print(f"\nProcessing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
                print(f"Text preview: {chunk[:60]}...")
                
                try:
                    synthesizer.synthesize(
                        text=chunk,
                        target_voice_path=target_voice,
                        exaggeration=exaggeration,
                        output_file_name=chunk_output,
                        apply_watermark=apply_wm
                    )
                    print(f"✓ Chunk {i} saved to {chunk_output}")
                    
                except Exception as e:
                    print(f"✗ Error processing chunk {i}: {e}")
                    print("Continuing with remaining chunks...")
            
            # Combine all chunks
            if combine_audio_chunks(chunk_files, output_file):
                # Clean up intermediate chunk files
                print("\n--- Cleaning up intermediate chunk files ---")
                for chunk_file in chunk_files:
                    try:
                        os.remove(chunk_file)
                        print(f"Deleted {chunk_file}")
                    except Exception as e:
                        print(f"Could not delete {chunk_file}: {e}")
        
        print(f"\n✓ Successfully processed: {os.path.basename(input_file)}")
        print(f"  Output: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing file {input_file}: {e}")
        return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Convert text files to speech using Chatterbox ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process all .md and .txt files from input directory
  python chatterbox-file.py --input-dir ./input --output-dir ./output
  
  # Use clipboard content (original behavior)
  python chatterbox-file.py --clipboard
  
  # Specify custom voice file
  python chatterbox-file.py --input-dir ./input --output-dir ./output --voice custom_voice.wav
        '''
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        help='Input directory containing .md and .txt files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for .wav files'
    )
    parser.add_argument(
        '--clipboard', '-c',
        action='store_true',
        help='Use clipboard content instead of files'
    )
    parser.add_argument(
        '--voice', '-v',
        default='en-Rob_man.wav',
        help='Path to target voice file (default: en-Rob_man.wav)'
    )
    parser.add_argument(
        '--exaggeration', '-e',
        type=float,
        default=0.5,
        help='Voice exaggeration parameter (default: 0.5)'
    )
    parser.add_argument(
        '--watermark', '-w',
        action='store_true',
        help='Apply watermark to output audio'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.clipboard:
        # Use clipboard mode
        process_clipboard_mode(args)
    elif args.input_dir and args.output_dir:
        # Use directory mode
        process_directory_mode(args)
    else:
        parser.error("Must specify either --clipboard or both --input-dir and --output-dir")


def process_clipboard_mode(args):
    """Process text from clipboard (original behavior)"""
    print("=== Clipboard Mode ===")
    
    # Read text from clipboard
    clipboard_text = pyperclip.paste()
    
    # Check if clipboard is empty
    if not clipboard_text or not clipboard_text.strip():
        print("Error: Clipboard is empty or contains only whitespace.")
        sys.exit(1)
    
    print(f"Clipboard text length: {len(clipboard_text)} characters")
    
    # Initialize synthesizer
    synthesizer = ChatterboxOnnx(quantized=True)
    
    # Chunk the text
    chunks = chunk_text(clipboard_text, chunk_size=300)
    print(f"Text split into {len(chunks)} chunk(s)")
    
    # Process chunks
    if len(chunks) == 1:
        # Single chunk - use original logic
        print("\n--- Processing single chunk ---")
        synthesizer.synthesize(
            text=chunks[0],
            target_voice_path=args.voice,
            exaggeration=args.exaggeration,
            output_file_name="chatterbox_tts_output.wav",
            apply_watermark=args.watermark
        )
    else:
        # Multiple chunks - process each separately and combine
        chunk_files = []
        
        print(f"\n--- Processing {len(chunks)} chunks ---")
        
        for i, chunk in enumerate(chunks, 1):
            chunk_output = f"chunk_{i:03d}.wav"
            chunk_files.append(chunk_output)
            
            print(f"\nProcessing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            print(f"Text preview: {chunk[:60]}...")
            
            try:
                synthesizer.synthesize(
                    text=chunk,
                    target_voice_path=args.voice,
                    exaggeration=args.exaggeration,
                    output_file_name=chunk_output,
                    apply_watermark=args.watermark
                )
                print(f"✓ Chunk {i} saved to {chunk_output}")
                
            except Exception as e:
                print(f"✗ Error processing chunk {i}: {e}")
                print("Continuing with remaining chunks...")
        
        # Combine all chunks
        if combine_audio_chunks(chunk_files, "chatterbox_tts_output.wav"):
            # Clean up intermediate chunk files
            print("\n--- Cleaning up intermediate chunk files ---")
            for chunk_file in chunk_files:
                try:
                    os.remove(chunk_file)
                    print(f"Deleted {chunk_file}")
                except Exception as e:
                    print(f"Could not delete {chunk_file}: {e}")
    
    print("\n--- Processing complete ---")


def process_directory_mode(args):
    """Process all .md and .txt files from input directory"""
    print("=== Directory Mode ===")
    
    # Validate directories
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Find all .md and .txt files
    input_files = []
    for ext in ['*.md', '*.txt']:
        pattern = os.path.join(args.input_dir, ext)
        input_files.extend(glob.glob(pattern))
    
    if not input_files:
        print(f"No .md or .txt files found in {args.input_dir}")
        sys.exit(1)
    
    input_files.sort()  # Process in alphabetical order
    print(f"Found {len(input_files)} file(s) to process")
    
    # Initialize synthesizer
    synthesizer = ChatterboxOnnx(quantized=True)
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] ", end="")
        
        if process_text_file(
            input_file,
            args.output_dir,
            synthesizer,
            args.voice,
            args.exaggeration,
            args.watermark
        ):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:   {len(input_files)}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()