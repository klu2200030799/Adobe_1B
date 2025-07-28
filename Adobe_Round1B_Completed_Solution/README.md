# Round 1B - Persona-Driven Document Intelligence

## How it Works:
- Accepts PDFs, persona.json, and job.json from `/app/input`
- Extracts blocks of text from PDFs using PyMuPDF
- Uses TF-IDF + cosine similarity to rank sections by relevance to the persona + task
- Outputs top 10 ranked sections to `/app/output/output.json`

## Run Locally:
```bash
docker build --platform linux/amd64 -t docinsight:adobe .
docker run --rm -v %cd%/input:/app/input -v %cd%/output:/app/output --network none docinsight:adobe
