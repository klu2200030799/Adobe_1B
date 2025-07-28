## Approach Explanation - Round 1B: Persona-Driven Document Intelligence

To address the challenge of extracting persona-specific insights from a collection of PDFs, our approach is designed around the principles of offline NLP, modular parsing, and scalable similarity ranking.

### Step 1: Input Interpretation
The system begins by reading a `persona.json` and a `job.json`, which together define the role (e.g., Undergraduate Chemistry Student), their area of focus (e.g., Organic Chemistry), and their immediate task (e.g., Study for reaction kinetics exam). These are concatenated into a search query representing the user's information need.

### Step 2: Text Extraction
Using `PyMuPDF`, each PDF is parsed page by page. Text blocks are extracted and filtered to retain only meaningful sections (sentences with >5 words). Each block is tagged with its page number and source document. This yields a comprehensive but structured collection of potential knowledge chunks.

### Step 3: Relevance Scoring
To evaluate relevance without relying on web APIs or heavy models, we use `TfidfVectorizer` from scikit-learn. Each extracted text block is treated as a document, and the persona+job query as the final item in the corpus. Cosine similarity is computed between this query and all blocks, ranking them based on semantic closeness.

### Step 4: Output Generation
The top 10 most relevant sections are selected. Each is annotated with:
- document name
- page number
- raw text content
- importance rank (1 being most relevant)
- cosine similarity score

These are bundled into a structured `output.json` with metadata, timestamp, and traceability.

### Key Benefits
- Fully offline & lightweight (under 1GB RAM & <60s runtime)
- Handles multiple domains generically
- Generalizes to any persona-task pair by converting them into ranking queries
- Easily extendable to sub-section extraction or summarization in future rounds

This methodology ensures meaningful, explainable, and repeatable insights for any document-user-task triad without network dependencies or GPU requirements.
