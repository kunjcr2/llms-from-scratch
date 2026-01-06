# RAG Lec-1 Vizuara AI

## 1. Data Ingestion

- Downloading and reading PDFs.
- If we have a ONLY text in the docs, we can use - `PyMuPDF`, can save images but not read the image, for that we need OCR. And a famous one is `tesseract`, it is used to get texts from the image from hand written stuff or bills image. For tabular data, we use `docling`. It is really good at working with tables, it saves the tables as ACTUAL tables with rows and cols, and can be joined with OCR where it can get tables through the image too by preserving the tables. docling can handle markdown tables and joining with an OCR tool. `Mistral AI OCR` is a good one as well !
- for scraping websites we use `Firecrawl`, `beautifulsoup4` for html, `pyppeteer` is good as well for titles and stuff by mentioning styles of what you want.

- Lets start with Data **Download and preprocess**. **PyMuPDF**. it is 10-15 times faster than docling.

### **Data Preparation**

1. We install packages.
2. We download the pdf and use `fitz` which is pymupdf basically and open it. we fo some preprocess on it and maintain a list of each page with text, nymber of chars; etc in a form of dictionary. we can get some stats of it as well.

### **Chunking**

- Chunk is basically a small corpus of entire book or pdf. We split the document in chunks and these are exactly what that are retrieved theough vector search ! We split the entire data in chunks and that is must !
- There are infinite possiblities of chubking !

- _**5 types of chunking**_ !
  1. `Fixed size chunking`. Chunks are of SAME SIZE. lets say 200 words. everything 200 words. There can be Overlap but it ysually creates issues with incomplete context provided or important information being passed to next chunk; etc.
     - For quick and fast processing. Semantic breaks and context losses.
  2. `Semantic Chunking`. First we add first sentence to the vector and get its embedding and add it to the chunk then we take second one and see if the second sent has embedding with higher cosine similarity with first one, then we add it. Should be above a threshold. and so on. The moment were done with semantic going below threshold, we stop the chunk and move ahead. We use it at lets say Parliament Debate. Unstructured but a good usecase.
     - Extremely complex and takes a lots of compute. Inconsistent Chunk sizes.
  3. `Structural Chunking`. We solit the report at proper structured points. Lets say for a single page stories book, we make chunk 1 at Intro, second at preface, third at story 1, fourth at story 2; and so on. Subsections can also be used as a chunk.
     - Some chunks may get SUPER big. But good for RAG Metadata, and can be human readable. Also fast and well structured.
  4. `Recusrive Chunking`. It uses structural Chunking but takes cars of Chunk size as well ! Having a textbook. We do structal chunking on it but lets say if one chapter of the textbook is 1200 words while over chunk limit is 1000 words so we basically do the chunking on another level. We chunk at lets say at paraphraph level. This is how we slowly make everything structured as well as proper chunk size. We are using multiple chunking under one structure chunking, and that's why it is called `Recursive`.
     - Apparently the **BEST**.
  5. `LLM Chunking`: Context drift happens during talks with the LLM. We move from a context to a context. Semantics are maintained. We tell the LLM by giving it entire context and tell it to break at proper points where context drift is happening. LLMs are good at doing this and human gave up at this point !
     - High semantic accuracy, good for docs with rapid changing contexts, unstructred texts are taken care of but  expensicve as well as context window limitations could happen. Stocastic outputs (LLM itself is a based on probabilities).
