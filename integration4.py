import os
import json
import re
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Any, Optional, Type
from pathlib import Path

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, PDFSearchTool # Using PDFSearchTool for extraction
from langchain_community.llms import Ollama # Correct import for Ollama with LangChain integration

# Other necessary imports from the original script (adapt as needed)
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Load email configuration either from .env or from email_config.json
email_config_path = "email_config.json"
if os.path.exists(email_config_path):
    try:
        with open(email_config_path, 'r') as config_file:
            email_config = json.load(config_file)
            logger.info(f"Loaded email configuration from {email_config_path}")
            SMTP_CONFIG = {
                "smtp_server": email_config.get("smtp_server"),
                "smtp_port": int(email_config.get("smtp_port", 587)),
                "sender_email": email_config.get("sender_email"),
                "sender_password": email_config.get("sender_password"),
                "sender_name": email_config.get("sender_name", "Recruitment Team")
            }
    except Exception as e:
        logger.error(f"Error loading email config from {email_config_path}: {e}")
        # Fallback to environment variables
        SMTP_CONFIG = {
            "smtp_server": os.getenv("SMTP_SERVER"),
            "smtp_port": int(os.getenv("SMTP_PORT", 587)),
            "sender_email": os.getenv("SENDER_EMAIL"),
            "sender_password": os.getenv("SENDER_PASSWORD"),
            "sender_name": os.getenv("SENDER_NAME", "Recruitment Team")
        }
else:
    # Use environment variables
    SMTP_CONFIG = {
        "smtp_server": os.getenv("SMTP_SERVER"),
        "smtp_port": int(os.getenv("SMTP_PORT", 587)),
        "sender_email": os.getenv("SENDER_EMAIL"),
        "sender_password": os.getenv("SENDER_PASSWORD"),
        "sender_name": os.getenv("SENDER_NAME", "Recruitment Team")
    }

# Validate SMTP configuration
if all([SMTP_CONFIG["smtp_server"], SMTP_CONFIG["sender_email"], SMTP_CONFIG["sender_password"]]):
    logger.info(f"SMTP configuration loaded. Server: {SMTP_CONFIG['smtp_server']}, Email: {SMTP_CONFIG['sender_email']}")
else:
    logger.warning("SMTP configuration incomplete. Email sending functionality will be limited.")

# --- Enhanced Helper Functions ---

def extract_email_from_text(text):
    """Extracts email addresses from text using regex."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None

def extract_phone_from_text(text):
    """Extracts phone numbers from text using regex."""
    # Multiple patterns to catch different phone formats
    phone_patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}',  # International format
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US/Canada format
        r'\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}'  # European format
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            return phones[0]
    return None

def extract_name_from_text(text, max_lines=5):
    """
    Attempts to extract a name from the beginning of a CV.
    This is a heuristic approach assuming the name is one of the first lines.
    """
    # Get first few lines where the name is likely to be
    lines = text.strip().split('\n')[:max_lines]
    
    # Look for a line with just a name (typically 2-3 words, not too long)
    for line in lines:
        clean_line = line.strip()
        # Skip lines that look like headers, emails, or are too long
        if (
            len(clean_line.split()) <= 3 and
            len(clean_line) < 40 and
            '@' not in clean_line and
            'CV' not in clean_line.upper() and
            'RESUME' not in clean_line.upper() and
            'CURRICULUM' not in clean_line.upper()
        ):
            return clean_line
    
    # Fallback: just return the first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()
    
    return None

def _extract_json_from_llm_response(response_text: str) -> Dict:
    """
    Enhanced JSON extraction with better error handling and recovery mechanisms.
    """
    logger.debug(f"Attempting to extract JSON from: {response_text[:1000]}...") # Log more of the response
    
    # First try: Look for JSON within ```json ... ```
    json_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```' 
    json_match = re.search(json_pattern, response_text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
        logger.debug("Found JSON block inside code block")
    else:
        # Second try: Look for a complete JSON object anywhere in the text
        json_pattern_bare = r'({[\s\S]*})'
        json_match_bare = re.search(json_pattern_bare, response_text)
        if json_match_bare:
            json_str = json_match_bare.group(1)
            logger.debug("Found JSON block without markdown")
        else:
            # Last resort: Try to find anything that looks remotely like a JSON object
            logger.warning("No complete JSON pattern found in LLM response. Attempting recovery.")
            # Look for patterns like {"key": "value"}
            fragments = re.findall(r'{[^{}]*}', response_text)
            if fragments:
                logger.debug(f"Found {len(fragments)} potential JSON fragments")
                json_str = fragments[0]  # Take the first fragment as our best guess
            else:
                logger.error("No JSON pattern found in LLM response.")
                return {}

    try:
        # Basic cleaning
        json_str = json_str.strip()
        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        parsed_json = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON: {list(parsed_json.keys())}")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}. String was: {json_str[:500]}...")
        
        # Advanced recovery attempts
        try:
            # Replace single quotes with double quotes (common LLM mistake)
            json_str_fixed = json_str.replace("'", '"')
            # Remove trailing commas more aggressively
            json_str_fixed = re.sub(r',\s*([\]}])', r'\1', json_str_fixed)
            # Add missing quotes to keys
            json_str_fixed = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str_fixed)
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            logger.error("Failed to repair JSON even with aggressive fixes.")
            
            # Last resort: Try to build a minimal valid JSON with regex
            try:
                # Extract key-value pairs with regex
                pairs = re.findall(r'"([^"]+)"\s*:\s*("([^"]*)"|\d+|\[.*?\]|{.*?})', json_str)
                if pairs:
                    minimal_json = {pair[0]: pair[1].strip('"') for pair in pairs}
                    logger.info(f"Created minimal JSON with keys: {list(minimal_json.keys())}")
                    return minimal_json
                else:
                    return {}
            except Exception:
                logger.error("All JSON recovery methods failed.")
                return {}

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Enhanced text extraction from PDF with better error handling and logging.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Starting text extraction from: {pdf_path}")
    
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    char_count = len(page_text)
                    logger.debug(f"Page {i+1}/{total_pages}: Extracted {char_count} characters")
                else:
                    logger.warning(f"Page {i+1}/{total_pages}: No text extracted, possible scanned image")
        
        text = text.strip()
        logger.info(f"Total extracted: {len(text)} characters from {pdf_path}")
        
        # Log a sample of the text for debugging
        text_sample = text[:500].replace('\n', ' ')
        logger.debug(f"Text sample: {text_sample}...")
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise RuntimeError(f"Error extracting text from PDF: {str(e)}")

def validate_cv_data(cv_data: Dict) -> Dict:
    """
    Validates and repairs the CV data structure, ensuring all expected fields exist.
    """
    if not cv_data:
        logger.warning("CV data is empty, creating minimal structure")
        cv_data = {}
    
    # Ensure personal_info exists
    if 'personal_info' not in cv_data:
        logger.warning("Missing personal_info section in CV data")
        cv_data['personal_info'] = {}
    
    # Ensure each expected key exists in personal_info
    expected_personal_keys = ['name', 'email', 'phone', 'address', 'linkedin', 'github', 'website']
    for key in expected_personal_keys:
        if key not in cv_data['personal_info']:
            logger.warning(f"Missing {key} in personal_info")
            cv_data['personal_info'][key] = ''
    
    # Ensure other main sections exist
    expected_sections = ['education', 'experience', 'skills', 'certifications', 'projects', 'summary']
    for section in expected_sections:
        if section not in cv_data:
            logger.warning(f"Missing {section} section in CV data")
            if section in ['education', 'experience', 'certifications', 'projects']:
                cv_data[section] = []
            elif section == 'skills':
                cv_data[section] = {'technical_skills': [], 'languages': [], 'soft_skills': []}
            else:
                cv_data[section] = ''
    
    return cv_data

# --- CrewAI Tools ---

class PDFTextExtractorTool(BaseTool):
    name: str = "PDF Text Extractor"
    description: str = "Extracts all text content from a given PDF file path. Input must be the full path to the PDF."

    def _run(self, pdf_path: str) -> str:
        try:
            return extract_text_from_pdf(pdf_path)
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"

class ContactInfoExtractorTool(BaseTool):
    name: str = "Contact Information Extractor"
    description: str = "Extracts contact information (email, phone, name) from CV text. Input must be the full CV text."

    def _run(self, cv_text: str) -> str:
        try:
            email = extract_email_from_text(cv_text)
            phone = extract_phone_from_text(cv_text)
            name = extract_name_from_text(cv_text)
            
            result = {
                "name": name,
                "email": email,
                "phone": phone
            }
            
            return json.dumps(result)
        except Exception as e:
            return f"Error extracting contact info: {str(e)}"

class EmailSenderTool(BaseTool):
    name: str = "Email Sender"
    description: str = "Sends an email using preconfigured SMTP settings. Input must be a dictionary with 'recipient_email', 'subject', and 'body'."

    def _run(self, email_data: Dict[str, str]) -> str:
        recipient = email_data.get("recipient_email")
        subject = email_data.get("subject")
        body = email_data.get("body")

        if not all([recipient, subject, body]):
            return "Error: Missing recipient_email, subject, or body in input."

        if not all([SMTP_CONFIG.get("smtp_server"), SMTP_CONFIG.get("sender_email"), SMTP_CONFIG.get("sender_password")]):
             return "Error: SMTP settings are not fully configured in environment variables or config file."

        try:
            msg = MIMEMultipart()
            sender_name = SMTP_CONFIG.get("sender_name", "Recruitment Team")
            sender_email = SMTP_CONFIG.get("sender_email")
            # Format sender as "Name <email>" if name is provided
            if sender_name:
                msg['From'] = f"{sender_name} <{sender_email}>"
            else:
                msg['From'] = sender_email
                
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_CONFIG["smtp_server"], SMTP_CONFIG["smtp_port"])
            server.starttls()
            server.login(SMTP_CONFIG["sender_email"], SMTP_CONFIG["sender_password"])
            text = msg.as_string()
            server.sendmail(SMTP_CONFIG["sender_email"], recipient, text)
            server.quit()
            logger.info(f"Email sent successfully to {recipient}")
            return f"Email successfully sent to {recipient}"
        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {str(e)}")
            return f"Error sending email: {str(e)}"

# --- Initialize LLM ---
try:
    ollama_llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    # Test connection
    ollama_llm.invoke("Respond with OK")
    logger.info(f"Successfully connected to Ollama model '{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}")
except Exception as e:
    logger.error(f"Failed to initialize or connect to Ollama LLM: {e}")
    logger.error("Please ensure Ollama is running and the model is available.")
    exit(1) # Exit if LLM connection fails


# --- Instantiate Tools ---
pdf_tool = PDFTextExtractorTool()
contact_tool = ContactInfoExtractorTool()
email_tool = EmailSenderTool()

# --- Define Agents ---

cv_analyst = Agent(
    role='Expert CV Analyst',
    goal='Accurately parse candidate CVs (provided as text) and extract key information into a structured JSON format, including personal details, education, experience, skills (categorized), certifications, projects, and generate a professional summary.',
    backstory=(
        "You are a meticulous analyst with years of experience in HR tech. "
        "You specialize in transforming unstructured CV text into clean, structured data. "
        "You understand the nuances of CV layouts and terminology across various industries. "
        "Your goal is to provide data ready for comparison and analysis."
        "You ALWAYS respond ONLY with the required JSON structure."
    ),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    tools=[contact_tool]  # Add the contact info extractor tool
)

job_analyst = Agent(
    role='Job Description Specialist',
    goal='Analyze job descriptions (provided as text) and extract key requirements (job title, technical skills, experience years, education, soft skills, languages) into a structured JSON format.',
    backstory=(
        "You are an expert in understanding recruitment needs. "
        "You can quickly identify the core requirements and qualifications from dense job descriptions. "
        "Your extraction focuses on quantifiable and comparable data points essential for candidate matching. "
        "You ALWAYS respond ONLY with the required JSON structure."
    ),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False
)

candidate_matcher = Agent(
    role='Candidate Fitness Assessor',
    goal=(
        "Critically evaluate a candidate's structured CV data against structured job requirements. "
        "Calculate percentage match scores for technical skills, education, experience, languages, and soft skills. "
        "Crucially, assess the DOMAIN MATCH between the candidate's expertise and the job's field. "
        "Identify missing skills and key strengths. "
        "Produce a final JSON output containing all scores, missing skills, strengths, and a concise overall assessment. "
        "Be VERY strict on domain and technical skills match."
    ),
    backstory=(
        "You are a seasoned technical recruiter with a keen eye for talent. "
        "You don't just look for keyword matches; you assess genuine fit, especially technical depth and domain relevance. "
        "You provide objective, data-driven evaluations to aid hiring decisions. Low technical or domain match scores significantly impact your overall assessment. "
        "You ALWAYS respond ONLY with the required JSON structure."
    ),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False
)

recruitment_communicator = Agent(
    role='Recruitment Communication Specialist',
    goal='Draft professional and personalized emails for candidates based on the matching results. Generate acceptance emails (for high matches), interview invitations (if accepted), or constructive rejection emails (for low matches), using provided templates and incorporating details from the match analysis (scores, strengths, missing skills).',
    backstory=(
        "You are an experienced HR communicator focused on candidate experience. "
        "You write clear, concise, and empathetic emails. "
        "For rejections, you provide brief, constructive feedback based on the match analysis. "
        "For acceptances/interviews, you convey enthusiasm and next steps clearly. "
        "You tailor the tone appropriately for each scenario."
        "You output ONLY the generated email text."
    ),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    tools=[email_tool]
)


# --- Define Tasks ---

# Task 2: Analyze CV
analyze_cv = Task(
    description=(
        "Analyze the provided CV text ({cv_text}) and extract the following information into a STRICT JSON format:\n"
        "- personal_info: (name, email, phone, address, linkedin, github, website) - use regex and LLM for name/address.\n"
        "- education: list of objects (degree, institution, period, field, location, grade)\n"
        "- experience: list of objects (title, company, period, location, responsibilities list, contract_type)\n"
        "- skills: object with keys (technical_skills list, languages list, soft_skills list) - Ensure soft skills have proper spacing.\n"
        "- certifications: list of objects (name, issuer, date, field, id)\n"
        "- projects: list of objects (title, description, technologies list, period, results, link)\n"
        "Generate a professional summary (~150 words, first person) based on the extracted info.\n\n"
        "IMPORTANT: Use the Contact Information Extractor tool to reliably extract email, phone, and name.\n\n"
        "FINAL OUTPUT MUST BE a single JSON object containing all these keys: "
        "{{\"personal_info\": ..., \"education\": ..., \"experience\": ..., \"skills\": ..., \"certifications\": ..., \"projects\": ..., \"summary\": \"...\"}}"
        "DO NOT include any other text before or after the JSON."
    ),
    expected_output=(
        "A single JSON string containing the structured CV data including keys: "
        "personal_info, education, experience, skills, certifications, projects, and summary."
    ),
    agent=cv_analyst,
    # Input context will be provided during crew kickoff
)

# Task 3: Analyze Job Description
analyze_job_description = Task(
    description=(
        "Analyze the provided Job Description text ({job_description}) and extract the key requirements into a STRICT JSON format:\n"
        "- job_title: string\n"
        "- company: string\n"
        "- location: string\n"
        "- technical_skills: list of strings\n"
        "- experience_years: string (e.g., '3-5 years', 'Senior', 'Minimum 2 years')\n"
        "- education: list of strings (required degrees or fields)\n"
        "- soft_skills: list of strings\n"
        "- languages: list of strings (e.g., 'English (Fluent)', 'French (Professional)')\n\n"
        "FINAL OUTPUT MUST BE a single JSON object containing these keys: "
        "{{\"job_title\": ..., \"company\": ..., \"location\": ..., \"technical_skills\": ..., \"experience_years\": ..., \"education\": ..., \"soft_skills\": ..., \"languages\": ...}}"
         "DO NOT include any other text before or after the JSON."
    ),
    expected_output=(
        "A single JSON string containing the structured job requirements including keys: "
        "job_title, company, location, technical_skills, experience_years, education, soft_skills, languages."
    ),
    agent=job_analyst,
    # Input context will be provided during crew kickoff
)

# Task 4: Match Candidate
match_candidate = Task(
    description=(
        "Compare the structured CV data and the structured Job Requirements provided in the context. "
        "Perform a CRITICAL analysis focusing heavily on the match between the candidate's technical skills and experience DOMAIN versus the job requirements. "
        "Calculate percentage scores (0-100) for: technical_skills_match, education_match, experience_match, languages_match, soft_skills_match, and domain_match. "
        "Identify key 'missing_skills' (job requirements not met by CV) and 'strengths' (CV aspects strongly matching requirements). "
        "Generate a brief 'final_assessment' text (max 150 words) summarizing the fit and recommending 'interview', 'consider', or 'reject'. Low technical (<40) or domain (<50) match should generally lead to 'reject'.\n\n"
        "FINAL OUTPUT MUST BE a single JSON object containing these keys: "
        "{{\"technical_skills_match\": ..., \"education_match\": ..., \"experience_match\": ..., \"languages_match\": ..., \"soft_skills_match\": ..., \"domain_match\": ..., \"missing_skills\": [...], \"strengths\": [...], \"final_assessment\": \"...\"}}"
        "DO NOT include any other text before or after the JSON."
    ),
    expected_output=(
        "A single JSON string containing the matching scores, missing skills, strengths, and final assessment including keys: "
        "technical_skills_match, education_match, experience_match, languages_match, soft_skills_match, domain_match, missing_skills, strengths, final_assessment."
    ),
    agent=candidate_matcher,
    context=[analyze_cv, analyze_job_description] # Depends on output of previous tasks
)

# Task 5: Draft Communication (Enhanced with company info)
draft_communication = Task(
    description=(
        "Based on the candidate's CV info, job requirements, and the detailed match results provided in the context, draft an appropriate email. "
        "Use the 'final_assessment' field from the match results to decide the email type:\n"
        "- If assessment suggests 'interview' or 'consider' (and overall technical_skills_match > 60 and domain_match > 60): Draft an **Acceptance/Interview Invitation Email**. Mention 1-2 strengths from the match results. State next steps (scheduling interview).\n"
        "- If assessment suggests 'reject' OR technical_skills_match < 50 OR domain_match < 50: Draft a **Rejection Email**. Be polite, thank the candidate. Briefly mention 1-2 general areas of mismatch based on 'missing_skills' or low scores. Wish them luck.\n\n"
        "Address the candidate by name (use 'personal_info.name' from CV data). Include the company name and job title from job requirements.\n\n"
        "IMPORTANT: Output ONLY the email text itself. Start with 'Subject:' line followed by the email body.\n"
        "Example format:\n"
        "Subject: Your Application for [Job Title] at [Company]\n\n"
        "Dear [Candidate Name],\n\n"
        "[Email body...]\n\n"
        "Regards,\n"
        "[Company Name] Recruitment Team"
    ),
    expected_output=(
        "The full text of the drafted email (Subject + Body)."
    ),
    agent=recruitment_communicator,
    context=[analyze_cv, analyze_job_description, match_candidate]
)

# --- Create Crew ---
recruitment_crew = Crew(
    agents=[cv_analyst, job_analyst, candidate_matcher, recruitment_communicator],
    tasks=[analyze_cv, analyze_job_description, match_candidate, draft_communication],
    process=Process.sequential,
    verbose=2 # verbose=2 logs agent actions and tool usage
)

# --- Main Execution Logic ---
def main():
    print("📄 CrewAI Virtual Recruitment Assistant 📄")
    print("=" * 80)

    pdf_path = input("📂 Enter the full path to the candidate's CV (PDF): ")
    if not pdf_path or not os.path.exists(pdf_path):
        print("❌ Invalid PDF path. Exiting.")
        return

    print("\n📋 Enter the job description (paste text, then press Enter twice to finish):")
    job_description_lines = []
    while True:
        try:
            line = input()
            if line:
                job_description_lines.append(line)
            else:
                break
        except EOFError: # Handles case where input is piped
             break
    job_description = "\n".join(job_description_lines)

    if not job_description.strip():
        print("❌ Job description cannot be empty. Exiting.")
        return

    print("\n⏳ Extracting text from CV...")
    try:
        cv_text = extract_text_from_pdf(pdf_path)
        if not cv_text:
            print("❌ Failed to extract text from CV or CV is empty. Exiting.")
            return
        print("✅ CV Text Extracted.")
    except Exception as e:
        print(f"❌ Error extracting CV text: {e}")
        return

    # Prepare inputs for the Crew
    inputs = {
        'cv_text': cv_text[:15000], # Limit context window for LLM
        'job_description': job_description[:15000] # Limit context window
    }

    print("\n🚀 Launching Recruitment Crew...")
    try:
        crew_result = recruitment_crew.kickoff(inputs=inputs)

        print("\n" + "=" * 80)
        print("✅ Crew Execution Finished!")
        print("=" * 80)

        # --- Process Crew Results ---
        # The final result 'crew_result' should be the output of the last task (email draft)

        print("\n📧 Generated Communication Draft:")
        print("-" * 80)
        print(crew_result) # This should be the email text
        print("-" * 80)

        # --- Attempt to retrieve intermediate results ---
        cv_data_json = {}
        job_req_json = {}
        match_details_json = {}

        try:
            # Access task outputs (assuming sequential execution)
            cv_analysis_output = analyze_cv.output.raw_output if analyze_cv.output else "{}"
            job_analysis_output = analyze_job_description.output.raw_output if analyze_job_description.output else "{}"
            match_output = match_candidate.output.raw_output if match_candidate.output else "{}"

            logger.info("Attempting to parse intermediate JSON outputs...")
            cv_data_json = _extract_json_from_llm_response(cv_analysis_output)
            job_req_json = _extract_json_from_llm_response(job_analysis_output)
            match_details_json = _extract_json_from_llm_response(match_output)

            print("\n📊 Intermediate Results Summary:")
            if cv_data_json:
                print(f"- Candidate Name: {cv_data_json.get('personal_info', {}).get('name', 'N/A')}")
                print(f"- CV Summary: {cv_data_json.get('summary', 'N/A')[:100]}...")
            if job_req_json:
                print(f"- Job Title: {job_req_json.get('job_title', 'N/A')}")
                print(f"- Company: {job_req_json.get('company', 'N/A')}")
            if match_details_json:
                 print(f"- Technical Match: {match_details_json.get('technical_skills_match', 'N/A')}%")
                 print(f"- Domain Match: {match_details_json.get('domain_match', 'N/A')}%")
                 print(f"- Final Assessment: {match_details_json.get('final_assessment', 'N/A')}")

        except Exception as e:
            logger.warning(f"Could not reliably parse intermediate task outputs: {e}")
            print("\n⚠️ Could not display all intermediate results.")


        # --- Optional: Save results / Send Email (Improved) ---
        candidate_email = cv_data_json.get('personal_info', {}).get('email')
        job_title = job_req_json.get('job_title', 'the position')
        company_name = job_req_json.get('company', 'Our Company')

        # Debug information for email parsing
        print("\n🔍 Email Generation Debug:")
        print(f"Email brut généré (début): {crew_result[:200]}...")

        if candidate_email:
            # Improved email subject and body extraction - handles multiline better
            email_subject_match = re.search(r"Subject:(.*?)(\n|\r\n)+", crew_result, re.IGNORECASE)
            if email_subject_match:
                email_subject = email_subject_match.group(1).strip()
                subject_end_pos = email_subject_match.end()
                email_body = crew_result[subject_end_pos:].strip()
            else:
                # Fallback if no subject found
                email_subject = f"Regarding your application for {job_title} at {company_name}"
                email_body = crew_result.strip()
            
            # Debug the extraction
            print(f"Sujet extrait: {email_subject}")
            print(f"Corps extrait (début): {email_body[:200]}...")

            send_option = input(f"\n📤 Send the generated email to {candidate_email}? (y/n): ")
            if send_option.lower() == 'y':
                if not all([SMTP_CONFIG.get("smtp_server"), SMTP_CONFIG.get("sender_email"), SMTP_CONFIG.get("sender_password")]):
                     print("❌ Cannot send email. SMTP settings incomplete in .env file or email_config.json.")
                else:
                    email_data = {
                        "recipient_email": candidate_email,
                        "subject": email_subject,
                        "body": email_body
                    }
                    sender_tool = EmailSenderTool()
                    send_status = sender_tool._run(email_data)
                    print(f"✉️ Email Send Status: {send_status}")
        else:
            print("\nℹ️ Candidate email not found in parsed CV data, cannot offer to send email.")

        save_option = input("\n💾 Save extracted data and match results to JSON? (y/n): ")
        if save_option.lower() == 'y':
            # Create a data directory if it doesn't exist
            data_dir = os.getenv("DATA_DIR", "recruitment_data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Create filename based on candidate name (if available) and job
            candidate_name = cv_data_json.get('personal_info', {}).get('name', 'unknown_candidate')
            candidate_name = re.sub(r'[^\w\s-]', '', candidate_name).replace(' ', '_').lower()
            
            output_data = {
                "cv_analysis": cv_data_json,
                "job_requirements": job_req_json,
                "match_details": match_details_json,
                "generated_email": crew_result
            }
            
            output_filename = os.path.join(data_dir, f"{candidate_name}_{Path(pdf_path).stem}_analysis.json")
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
                print(f"✅ Results saved to {output_filename}")
            except Exception as e:
                print(f"❌ Error saving results: {e}")


    except Exception as e:
        logger.error(f"An error occurred during Crew execution: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred: {e}")

    print("\n" + "=" * 80)
    print("🏁 Recruitment process finished.")
    print("=" * 80)

if __name__ == "__main__":
    main()