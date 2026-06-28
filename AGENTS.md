# Project: haCARton

Satellite imagery analysis pipeline using AI segmentation and classification with human-in-the-loop review.

## Vision

1. Connect to Google Earth Engine or another satellite image provider (agnostic)
2. Segment satellite images into polygons using Meta's Segment Anything
3. Classify each polygon's content using a custom-trained model
4. Present results to humans for approval/rejection
5. Save approved results

## Architecture

- **Image Provider**: Agnostic interface for satellite imagery
- **Segmentation**: Meta's Segment Anything (SAM)
- **Classification**: Custom trained model (to be trained)
- **Review Interface**: Streamlit app for human validation

## Current State

- Hackathon prototype
- Processing is mocked (no actual AI calls yet)
- Streamlit review interface in progress

## Tech Stack

- Python
- Streamlit
- Mock processing layer