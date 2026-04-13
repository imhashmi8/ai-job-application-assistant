import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from scraper import scrape_job_posting
import tiktoken

# --- Load environment variables --- #
load_dotenv(override=True)


# --- Model switcher: change to "ollama" to use local model --- #
MODE = "openai" # or "ollama"

if MODE == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL = "gpt-4o-mini"
else:
    client = OpenAI(base_url=os.getenv("OLLAMA_BASE_URL", api_key="ollama"))
    MODEL = "llama3.2"

# --- Token counting utility --- #
def count_tokens(text, model=MODEL):
    """Count tokens in a text string for the specified model."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# --- Core function to stream response from the model --- #
def stream_response(messages):
    """"Stream the response from the model."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000,
        temperature=0.7,
        stream=True
    )
    result = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            result += delta
    print()  # for newline after streaming
    return result

# --- Main function to analyze job posting --- #
def analyze_job(job_text):
    """Analyze the job posting and generate a tailored cover letter."""
    system_prompt = (
        "You are a helpful assistant that analyzes job postings and generates tailored cover letters. "
        "Extract key requirements, responsibilities, and company culture from the job posting. "
        "Then, create a personalized cover letter that highlights how the applicant's skills and experience align with the job."
    )

    user_prompt = (
        f"Here is the job posting:\n\n{job_text}\n\nPlease analyze it and write a cover letter."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    print(json.dumps(result, indent=2))
    return result


# --- Function to generate cover letter --- #
def generate_cover_letter(job_analysis, candidate_backgroud):
    """ Generate a cover letter based on job analysis and candidate backgroud."""
    system_prompt = (
        "You are a helpful assistant that generates tailored cover letters based on job analysis and candidate background. "
        "Use the job analysis to highlight the most relevant skills and experiences from the candidate's background. "
        "Write a personalized cover letter that effectively connects the candidate's qualifications to the job requirements."
    )

    user_prompt = (
        f"Here is the job analysis:\n{job_analysis}\n\n"
        f"Here is the candidate background:\n{candidate_backgroud}\n\n"
        "Please write a cover letter based on this information."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return stream_response(messages)


# --- Generate resume bullet points --- #
def generate_resume_bullets(job_analysis, candidate_backgroud):
    """ Generate resume bullet points based on job analysis and candidate background."""
    system_prompt = (
        "You are an expert resume writer. Generate 5 strong, quantified resume bullet points "
        "that align the candidate's experience with the job requirements. "
        "Use the format: 'Action verb + what you did + measurable result'. "
        "Tailor keywords to match the job posting."
    )

    user_prompt = (
        f"Here is the job analysis:\n{job_analysis}\n\n"
        f"Here is the candidate background:\n{candidate_backgroud}\n\n"
        "Please write 5 resume bullet points based on this information."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return stream_response(messages)


# --- Main execution flow --- #
if __name__ == "__main__":
    print(" === Smart Job Application Assistant === ")
    print(f"Model: {MODEL} ({MODE})\n")

    job_url= input("Enter the job posting URL (or press Enter to use sample text): ").strip()

    if job_url:
        print("\nFetching and analyzing job posting...")
        job_text = scrape_job_posting(job_url)
        print(f"Fetched {count_tokens(job_text)} tokens of job text.")
    else:
        job_text = input("Enter the job posting text:\n")

    candidate_background = input("\nEnter your background (skills, experience, etc.):\n")
    print("\nAnalyzing job posting and generating cover letter...")
    job_analysis = analyze_job(job_text)
    print("\nGenerating cover letter...")
    cover_letter = generate_cover_letter(job_analysis, candidate_background)
    print("\nGenerating resume bullet points...")
    resume_bullets = generate_resume_bullets(job_analysis, candidate_background)    
    print("\n=== Cover Letter ===\n")
    print(cover_letter)
    print("\n=== Resume Bullet Points ===\n")
    print(resume_bullets)

    print("\nDone!")