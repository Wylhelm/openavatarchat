#!/usr/bin/env python3
"""Test script to list ElevenLabs voices and test French TTS"""

import os
from elevenlabs import ElevenLabs

# Get API key from config or environment
api_key = os.getenv("ELEVENLABS_API_KEY", "sk_f816c34ec8c96dfeefae46c6552a750c79314f1d59447ba8")

client = ElevenLabs(api_key=api_key)

print("=" * 60)
print("ElevenLabs Voice Test")
print("=" * 60)

# List all voices
print("\nFetching available voices...")
try:
    voices = client.voices.get_all()
    print(f"\nFound {len(voices.voices)} voices\n")
    
    # Filter French voices
    french_voices = []
    for voice in voices.voices:
        # Check if voice name or description contains French indicators
        name_lower = voice.name.lower()
        if any(keyword in name_lower for keyword in ['french', 'français', 'france', 'fr-']):
            french_voices.append(voice)
        # Also check voice settings if available
        if hasattr(voice, 'labels') and voice.labels:
            if 'french' in str(voice.labels).lower() or 'fr' in str(voice.labels).lower():
                if voice not in french_voices:
                    french_voices.append(voice)
    
    print(f"Found {len(french_voices)} potentially French voices:\n")
    for voice in french_voices:
        print(f"  Voice ID: {voice.voice_id}")
        print(f"  Name: {voice.name}")
        if hasattr(voice, 'labels'):
            print(f"  Labels: {voice.labels}")
        print()
    
    # Test current voice
    current_voice_id = "IPgYtHTNLjC7Bq7IPHrm"
    print(f"\nTesting current voice ID: {current_voice_id}")
    current_voice = None
    for voice in voices.voices:
        if voice.voice_id == current_voice_id:
            current_voice = voice
            break
    
    if current_voice:
        print(f"  Name: {current_voice.name}")
        if hasattr(current_voice, 'labels'):
            print(f"  Labels: {current_voice.labels}")
        print(f"  Description: {getattr(current_voice, 'description', 'N/A')}")
    else:
        print("  ⚠ Voice not found in your account!")
    
    # Test French text
    print(f"\nTesting French TTS with voice {current_voice_id}...")
    test_text = "Bonjour, je suis Guillaume Gagnon. Comment puis-je vous aider?"
    
    audio_data = b''
    for chunk in client.text_to_speech.stream(
        text=test_text,
        voice_id=current_voice_id,
        model_id="eleven_multilingual_v2",
        output_format="pcm_24000",
    ):
        audio_data += chunk
    
    print(f"✓ Generated {len(audio_data)} bytes of audio")
    print(f"  Text sent: '{test_text}'")
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

