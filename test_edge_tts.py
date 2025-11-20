#!/usr/bin/env python3
"""
Test script to isolate Edge TTS issues.
This script tests Edge TTS functionality independently of the handler.
Tests both English and French voices.
"""

import edge_tts
import io
import time
import threading
import librosa
import numpy as np

def list_french_voices():
    """List available French voices"""
    print("=" * 60)
    print("Listing available French voices")
    print("=" * 60)
    try:
        import asyncio
        async def get_voices():
            voices = await edge_tts.list_voices()
            french_voices = [v for v in voices if v['Locale'].startswith('fr')]
            male_voices = [v for v in french_voices if 'Male' in v['Gender']]
            print(f"\nFound {len(french_voices)} French voices ({len(male_voices)} male):\n")
            for voice in male_voices:
                print(f"  {voice['ShortName']:30} - {voice['Gender']:10} - {voice['Locale']}")
            return male_voices
        return asyncio.run(get_voices())
    except Exception as e:
        print(f"Error listing voices: {e}")
        return []

def test_stream_sync(voice="en-US-AriaNeural", text="Hello, this is a test of Edge TTS. How are you today?", language="English"):
    """Test the synchronous stream_sync() method"""
    print("=" * 60)
    print(f"Testing Edge TTS stream_sync() method - {language}")
    print("=" * 60)
    
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print()
    
    try:
        print("Creating Communicate object...")
        communicate = edge_tts.Communicate(text, voice)
        print("✓ Communicate object created")
        print()
        
        print("Starting stream_sync()...")
        print("(This may take a few seconds - waiting for first chunk...)")
        start_time = time.time()
        data = b''
        chunk_count = 0
        stream_error = None
        stream_completed = False
        
        def stream_audio():
            nonlocal data, chunk_count, stream_error, stream_completed
            try:
                for chunk in communicate.stream_sync():
                    chunk_count += 1
                    elapsed = time.time() - start_time
                    chunk_type = chunk.get('type', 'unknown')
                    
                    if chunk_count == 1:
                        print(f"✓ Received first chunk after {elapsed:.2f} seconds (type: {chunk_type})")
                    
                    if chunk_type == 'audio':
                        data += chunk['data']
                        if chunk_count % 10 == 0 or chunk_count <= 3:
                            print(f"  Chunk {chunk_count}: audio data, total size: {len(data)} bytes, elapsed: {elapsed:.2f}s")
                    else:
                        print(f"  Chunk {chunk_count}: type={chunk_type}, elapsed: {elapsed:.2f}s")
                stream_completed = True
            except Exception as e:
                stream_error = e
        
        # Run in thread with timeout
        stream_thread = threading.Thread(target=stream_audio, daemon=True)
        stream_thread.start()
        
        # Wait with timeout and progress logging
        timeout = 30.0
        check_interval = 2.0
        last_check = time.time()
        
        while stream_thread.is_alive():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n✗ TIMEOUT: stream_sync() still running after {timeout} seconds")
                print("  This suggests stream_sync() is hanging")
                print(f"  Chunks received: {chunk_count}")
                return False
            
            # Log progress every check_interval seconds
            if time.time() - last_check > check_interval:
                print(f"  Still waiting... elapsed: {elapsed:.1f}s, chunks: {chunk_count}")
                last_check = time.time()
            
            stream_thread.join(timeout=check_interval)
        
        if stream_error:
            elapsed = time.time() - start_time
            print(f"\n✗ ERROR after {elapsed:.2f} seconds:")
            print(f"  {type(stream_error).__name__}: {stream_error}")
            import traceback
            traceback.print_exc()
            return False
        
        if not stream_completed:
            print(f"\n✗ WARNING: Thread completed but stream_completed flag not set")
            print(f"  Chunks received: {chunk_count}")
        
        elapsed = time.time() - start_time
        print()
        print(f"✓ stream_sync() completed in {elapsed:.2f} seconds")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total audio data: {len(data)} bytes")
        print()
        
        if len(data) == 0:
            print("✗ ERROR: No audio data received!")
            return False
        
        # Test loading with librosa
        print("Testing librosa.load()...")
        output_audio, orig_sr = librosa.load(io.BytesIO(data), sr=None)
        print(f"✓ Audio loaded successfully")
        print(f"  Original sample rate: {orig_sr} Hz")
        print(f"  Audio length: {len(output_audio)} samples ({len(output_audio)/orig_sr:.2f} seconds)")
        print()
        
        # Test resampling to 24000 Hz
        target_sr = 24000
        if orig_sr != target_sr:
            print(f"Testing resampling from {orig_sr} Hz to {target_sr} Hz...")
            resampled_audio = librosa.resample(output_audio, orig_sr=orig_sr, target_sr=target_sr)
            print(f"✓ Resampling successful")
            print(f"  Resampled length: {len(resampled_audio)} samples ({len(resampled_audio)/target_sr:.2f} seconds)")
            print()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print()
        print("=" * 60)
        print(f"✗ ERROR after {elapsed:.2f} seconds:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def test_async_stream():
    """Test the asynchronous stream() method"""
    print("=" * 60)
    print("Testing Edge TTS async stream() method")
    print("=" * 60)
    
    import asyncio
    
    text = "Hello, this is a test of Edge TTS. How are you today?"
    voice = "en-US-AriaNeural"
    
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print()
    
    async def test_async():
        try:
            print("Creating Communicate object...")
            communicate = edge_tts.Communicate(text, voice)
            print("✓ Communicate object created")
            print()
            
            print("Starting async stream()...")
            start_time = time.time()
            data = b''
            chunk_count = 0
            
            async for chunk in communicate.stream():
                chunk_count += 1
                elapsed = time.time() - start_time
                chunk_type = chunk.get('type', 'unknown')
                
                if chunk_count == 1:
                    print(f"✓ Received first chunk after {elapsed:.2f} seconds (type: {chunk_type})")
                
                if chunk_type == 'audio':
                    data += chunk['data']
                    if chunk_count % 10 == 0 or chunk_count <= 3:
                        print(f"  Chunk {chunk_count}: audio data, total size: {len(data)} bytes, elapsed: {elapsed:.2f}s")
                else:
                    print(f"  Chunk {chunk_count}: type={chunk_type}, elapsed: {elapsed:.2f}s")
            
            elapsed = time.time() - start_time
            print()
            print(f"✓ async stream() completed in {elapsed:.2f} seconds")
            print(f"  Total chunks: {chunk_count}")
            print(f"  Total audio data: {len(data)} bytes")
            print()
            
            if len(data) == 0:
                print("✗ ERROR: No audio data received!")
                return False
            
            print("=" * 60)
            print("✓ Async test passed!")
            print("=" * 60)
            return True
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            print()
            print("=" * 60)
            print(f"✗ ERROR after {elapsed:.2f} seconds:")
            print(f"  {type(e).__name__}: {e}")
            print("=" * 60)
            import traceback
            traceback.print_exc()
            return False
    
    try:
        return asyncio.run(test_async())
    except Exception as e:
        print(f"✗ ERROR running async test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_long_text():
    """Test with a longer text (similar to what the handler receives)"""
    print("=" * 60)
    print("Testing Edge TTS with longer text")
    print("=" * 60)
    
    text = "That's a normal reaction. Let me explain how it's designed to work and try to alleviate any concerns you may have. Would you like to know more about the process?"
    voice = "en-US-AriaNeural"
    
    print(f"Text length: {len(text)} characters")
    print(f"Voice: {voice}")
    print()
    
    try:
        print("Creating Communicate object...")
        communicate = edge_tts.Communicate(text, voice)
        print("✓ Communicate object created")
        print()
        
        print("Starting stream_sync()...")
        start_time = time.time()
        data = b''
        chunk_count = 0
        
        for chunk in communicate.stream_sync():
            chunk_count += 1
            elapsed = time.time() - start_time
            
            if chunk['type'] == 'audio':
                data += chunk['data']
                if chunk_count % 20 == 0 or chunk_count <= 5:
                    print(f"  Chunk {chunk_count}: {len(data)} bytes, elapsed: {elapsed:.2f}s")
            
            # Timeout check
            if elapsed > 30:
                print(f"✗ TIMEOUT: Still processing after 30 seconds")
                return False
        
        elapsed = time.time() - start_time
        print()
        print(f"✓ stream_sync() completed in {elapsed:.2f} seconds")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total audio data: {len(data)} bytes")
        print()
        
        if len(data) == 0:
            print("✗ ERROR: No audio data received!")
            return False
        
        print("=" * 60)
        print("✓ Long text test passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print()
        print("=" * 60)
        print(f"✗ ERROR after {elapsed:.2f} seconds:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def test_french_voice():
    """Test French male voice"""
    print("=" * 60)
    print("Testing French Male Voice")
    print("=" * 60)
    
    # Test with different French male voices
    french_voices = [
        "fr-FR-DenoisNeural",
        "fr-FR-HenriNeural", 
        "fr-FR-AlainNeural"
    ]
    
    text = "Bonjour, je suis l'avatar d'Alphonse Desjardins, comment puis-je vous aider aujourd'hui?"
    
    results = []
    for voice in french_voices:
        print(f"\n--- Testing voice: {voice} ---\n")
        try:
            result = test_stream_sync(voice=voice, text=text, language="French")
            results.append((voice, result))
        except Exception as e:
            print(f"✗ Error testing {voice}: {e}")
            results.append((voice, False))
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Edge TTS Test Script")
    print("=" * 60 + "\n")
    
    # First, list available French voices
    french_voices = list_french_voices()
    print("\n")
    
    results = []
    
    # Test 1: Basic stream_sync() - English
    results.append(("stream_sync() basic (English)", test_stream_sync()))
    print("\n")
    
    # Test 2: French voices
    french_results = test_french_voice()
    for voice, result in french_results:
        results.append((f"French voice: {voice}", result))
    print("\n")
    
    # Test 3: Async stream()
    results.append(("async stream()", test_async_stream()))
    print("\n")
    
    # Test 4: Long text
    results.append(("stream_sync() long text", test_long_text()))
    print("\n")
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    print("=" * 60)
    
    # Count passed/failed
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n⚠ Some tests failed. Check the output above for details.")
    else:
        print("\n✓ All tests passed!")

