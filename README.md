# Fraudulent Candidate Detection Tool

This project provides an end-to-end system to detect fraudulent patterns in resumes, optionally compare with job descriptions and LinkedIn profiles, compute a fit score, and produce JSON/HTML reports. A Streamlit UI is provided for interactive use, and a CLI is available for batch runs.

<img width="1917" height="945" alt="Screenshot 2025-08-21 093157" src="https://github.com/user-attachments/assets/f53d710f-2459-4b93-a386-8adc9d5de17b" />
<img width="1532" height="718" alt="Screenshot 2025-08-21 093205" src="https://github.com/user-attachments/assets/6e779ad0-53f4-414f-883d-33c7b9094ef8" />




## Features
- AI/NLP-driven analysis of resumes
- Fraud flags: timeline overlaps, skill inflation, education mismatches, plagiarism indicators, verification discrepancies
- Optional cross-verification using LinkedIn profile text
- Fit scoring (skills, experience, education, keyword density; optional semantic similarity)
- JSON and HTML report generation
- Streamlit dashboard with visualizations

## Tech Stack
- Python 3.10+
- Streamlit for UI
- scikit-learn, sentence-transformers (optional), matplotlib
- pdfminer.six, python-docx for document parsing
- Optional: Google Gemini API for AI insights

## Setup

```bash
# From project root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want AI insights via Gemini, set `GOOGLE_API_KEY` or provide the key in the UI.

## Quick Start (CLI)

```bash
# Example with provided samples
python fraud_detector.py --resume samples/resume_sample.txt --jd samples/jd_sample.txt --linkedin samples/linkedin_sample.txt --format json --output out.json

# HTML report
python fraud_detector.py --resume samples/resume_sample.txt --jd samples/jd_sample.txt --format html --output out.html
```

## Streamlit App

```bash
streamlit run fraud_detector.py
```

- Upload a resume (PDF/DOCX/TXT)
- Optionally paste a Job Description and LinkedIn profile text or upload a LinkedIn PDF export
- Optionally enter a Gemini API key to enable AI insights

## Project Structure
- `fraud_detector.py`: Core engine, Streamlit UI, CLI
- `samples/`: Sample inputs
- `requirements.txt`: Dependencies

## Notes
- Sentence embeddings and AI insights are optional. If models or keys are unavailable, the system degrades gracefully.
- Reports include clear explanations for each flag to support human review.
