# interviewinsights

# The web interface has been created by Harshita
check this url- https://interviewinsights.netlify.app/

# Emotion Detection from Video and Audio created by Rudrika
Detects human emotions by combining facial expressions, voice tone, and text analysis from video and audio inputs. It uses DeepFace for facial analysis, an SVM for audio classification, and transformer models for text analysis.

## Features
- **Facial Analysis:** Extracts emotions from video frames using DeepFace.
- **Audio Analysis:** Extracts MFCC features and classifies emotions with an SVM.
- **Text Analysis:** Transcribes audio, creates sentence embeddings, and identifies emotion-related words.
- **Fusion:** Merges results from all modalities for a final emotion prediction.

## Installation
```bash
pip install deepface opencv-python librosa numpy soundfile sklearn speechrecognition transformers torch faiss-cpu
```

## Usage Examples
**1. Extract a Video Frame:**
```python
from emotion_detection import get_video_frame
frame = get_video_frame("path/to/video.mp4")
```
**2. Process Audio:**
```python
from emotion_detection import read_audio
audio, sample_rate = read_audio("path/to/audio.wav")
```
**3. Facial Expression Analysis:**
```python
from emotion_detection import analyze_facial_expression
dominant, emotions = analyze_facial_expression(frame)
```
**4. Audio Emotion Classification:**
```python
from emotion_detection import train_audio_classifier, analyze_voice_tone
model = train_audio_classifier(feature_vectors, labels)  # Train on your data
voice_emotion, probabilities = analyze_voice_tone(audio, sample_rate, model)
```
**5. Speech-to-Text & Text Analysis:**
```python
from emotion_detection import transcribe_audio, rag_word_classification, count_emotion_words
text = transcribe_audio("path/to/audio.wav")
retrieved = rag_word_classification(text, ["happy", "sad", "angry", "anxious"])
word_counts = count_emotion_words(text, retrieved)
```
**6. Fuse Modalities:**
```python
from emotion_detection import fuse_emotions
final_emotion, confidence = fuse_emotions(dominant, voice_emotion, word_counts)
```
**7. Full Pipeline:**
```python
from emotion_detection import emotion_detection_pipeline
emotion_detection_pipeline("path/to/video.mp4", "path/to/audio.wav")
```
