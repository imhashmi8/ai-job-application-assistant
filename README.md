# AI Job Application Assistant

An intelligent job application tool powered by LLMs. Paste any job posting URL and the assistant scrapes the listing, extracts structured requirements, streams a tailored cover letter, and generates resume bullet points aligned to the role.

> Built as part of my LLM Engineering journey, demonstrating practical use of the OpenAI API, web scraping, streaming, structured JSON output, and multi-step LLM pipelines.

---

## Features

- **Live job scraping** — fetches and cleans any job posting URL using `requests` + `BeautifulSoup`
- **Structured job analysis** — extracts role, required skills, responsibilities, and culture keywords as JSON
- **Streaming cover letter** — generates a personalized cover letter token-by-token in real time
- **Resume bullet points** — produces tailored, action-verb resume bullets aligned to the job
- **Token awareness** — shows token count per request using `tiktoken`
- **Model flexibility** — switch between OpenAI (cloud) and Ollama (local, no API key) by changing one line

---

## How It Works

```
Job Posting URL
    └─► Scraper          →  clean job text        (BeautifulSoup)
          └─► Analyzer   →  structured JSON        (GPT + json_object)
                └─► Generator  →  cover letter + resume bullets  (streamed)
```

**Three-step LLM pipeline:**

1. **Scrape** — strip HTML noise and extract readable job posting text
2. **Analyze** — extract structured requirements (skills, responsibilities, culture) as JSON
3. **Generate** — stream a tailored cover letter and 5 resume bullet points

---

## Project Structure

```
ai-job-application-assistant/
├── smart_job_assistant.ipynb   # Interactive notebook (main demo)
├── job_assistant.py            # CLI version with streaming
├── scraper.py                  # Reusable web scraping utility
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)

| Library | Purpose |
|---|---|
| `openai` | Chat Completions API, streaming, JSON structured output |
| `requests` + `beautifulsoup4` | Scrape and parse live job postings |
| `tiktoken` | Token counting for cost and context awareness |
| `python-dotenv` | Secure API key management via `.env` |
| `IPython.display` | Real-time streaming in Jupyter notebooks |

---

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/imhashmi8/ai-job-application-assistant.git
cd ai-job-application-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your API key**
```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

**4. Run the notebook**
```bash
jupyter notebook smart_job_assistant.ipynb
```

Or use the CLI version with streaming:
```bash
python job_assistant.py
```

---

## Using a Local Model (No API Key Required)

The assistant supports [Ollama](https://ollama.com) for fully local, private inference:

```bash
ollama pull llama3.2
```

In `job_assistant.py`, change:
```python
MODE = "ollama"
```

That's it — the same code runs against your local model with no API costs.

---

## LLM Concepts Demonstrated

| Concept | Where Used |
|---|---|
| Chat Completions API | All LLM calls |
| System vs user prompts | Analyst, coach, and resume writer personas |
| JSON structured output | Job analysis step (`response_format: json_object`) |
| Streaming responses | Cover letter and resume bullet generation |
| Multi-step LLM pipeline | Scrape → Analyze → Generate |
| Token counting | `count_tokens()` with `tiktoken` |
| OpenAI-compatible endpoints | Ollama local model support via `base_url` |
| Prompt engineering | Role-specific system prompts per task |

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `OLLAMA_BASE_URL` | Ollama endpoint (default: `http://localhost:11434`) |

---

## Author

**Md Qamar Hashmi** — DevOps / SRE / Platform Engineer  
5+ years in cloud infrastructure, automation, and observability  
AWS · Azure · Kubernetes · Terraform · Python

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-qamar-hashmi/)
[![GitHub](https://img.shields.io/badge/GitHub-imhashmi8-181717?logo=github&logoColor=white)](https://github.com/imhashmi8)

---

## License

MIT
