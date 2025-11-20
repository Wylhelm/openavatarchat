#!/usr/bin/env python3
"""Simple Edge TTS test to check if authentication works"""

import edge_tts
import asyncio

async def test_direct():
    """Test direct TTS without listing voices"""
    print("Testing Edge TTS direct synthesis...")
    try:
        text = "Bonjour, je suis un test."
        voice = "fr-FR-DenoisNeural"
        
        print(f"Text: {text}")
        print(f"Voice: {voice}")
        print("Creating Communicate object...")
        
        communicate = edge_tts.Communicate(text, voice)
        
        print("Streaming audio...")
        data = b''
        chunk_count = 0
        
        async for chunk in communicate.stream():
            chunk_count += 1
            if chunk['type'] == 'audio':
                data += chunk['data']
                if chunk_count <= 3 or chunk_count % 10 == 0:
                    print(f"  Chunk {chunk_count}: {len(data)} bytes")
        
        print(f"\n✓ Success! Generated {len(data)} bytes of audio")
        print(f"  Total chunks: {chunk_count}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_direct())
    exit(0 if result else 1)

