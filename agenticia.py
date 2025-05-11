import os
import json
import re
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import time
import traceback

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools.base_tool import BaseTool
from crewai_tools import PDFSearchTool
from langchain_community.llms import Ollama

# PDF extraction imports
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
        r'[+]?[\d]{1,3}[-\s]?[(]?[\d]{1,4}[)]?[-\s]?[\d]{1,4}[-\s]?[\d]{1,4}',  # International format
        r'[(]?[\d]{3}[)]?[-\s]?[\d]{3}[-\s]?[\d]{4}',  # US/Canada format
        r'[\d]{2}[-\s]?[\d]{2}[-\s]?[\d]{2}[-\s]?[\d]{2}[-\s]?[\d]{2}',  # European format
        r'\+?\d{8,15}'  # Simple long digits
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            return phones[0]
    return None

def extract_name_from_text(text, max_lines=8):
    """
    Attempts to extract a name from the beginning of a CV with improved logic.
    """
    # Get first few lines where the name is likely to be
    lines = text.strip().split('\n')[:max_lines]
    
    # First pass: Look for lines that appear to be a name (not too long, not common headers)
    for i, line in enumerate(lines):
        clean_line = line.strip()
        # Skip empty lines
        if not clean_line:
            continue
            
        # Check if this looks like a name (2-3 words, not too long, no special characters)
        word_count = len(clean_line.split())
        if (
            1 <= word_count <= 4 and
            len(clean_line) < 45 and
            '@' not in clean_line and
            not re.search(r'[0-9]', clean_line) and
            not any(keyword in clean_line.upper() for keyword in ['CV', 'RESUME', 'CURRICULUM', 'VITAE', 'PHONE', 'EMAIL', 'ADDRESS', 'PROFILE'])
        ):
            # Check if the next line looks like a title/role - if so, this is likely the name
            if i+1 < len(lines) and lines[i+1]:
                next_line = lines[i+1].strip()
                if len(next_line.split()) <= 5 and len(next_line) < 50:
                    return clean_line
            else:
                return clean_line
    
    # Second pass - less strict: just return the first non-empty line that's reasonably sized
    for line in lines:
        clean_line = line.strip()
        if clean_line and len(clean_line) < 60:
            return clean_line
    
    # Fallback - first non-empty line regardless
    for line in lines:
        if line.strip():
            return line.strip()
    
    return "Unknown Candidate"

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

def improved_extract_text_from_pdf(pdf_path: str) -> str:
    """
    Improved text extraction from PDF that handles complex layouts better.
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
            
            # First pass - standard extraction
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text(y_tolerance=3)  # Adjust y_tolerance for better line merging
                    if page_text:
                        text += page_text + "\n"
                        char_count = len(page_text)
                        logger.debug(f"Page {i+1}/{total_pages}: Extracted {char_count} characters")
                    else:
                        logger.warning(f"Page {i+1}/{total_pages}: No text extracted with standard method")
                        # Try alternative extraction if standard method failed
                        try:
                            # Extract words individually and reconstruct text
                            words = page.extract_words(x_tolerance=3, y_tolerance=3)
                            if words:
                                reconstructed_text = ""
                                prev_y = None
                                for word in words:
                                    if prev_y is not None and abs(word['top'] - prev_y) > 5:  # New line
                                        reconstructed_text += "\n"
                                    reconstructed_text += word['text'] + " "
                                    prev_y = word['top']
                                
                                text += reconstructed_text.strip() + "\n"
                                logger.info(f"Page {i+1}/{total_pages}: Recovered text using word extraction")
                            else:
                                logger.warning(f"Page {i+1}/{total_pages}: Could not extract words")
                        except Exception as e:
                            logger.warning(f"Failed alternative extraction on page {i+1}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
                    logger.warning("Continuing with next page...")
        
        # Clean up the extracted text
        cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Replace multiple newlines with double newlines
        cleaned_text = re.sub(r'^\s+', '', cleaned_text, flags=re.MULTILINE)  # Remove leading whitespace from each line
        
        final_text = cleaned_text.strip()
        logger.info(f"Total extracted: {len(final_text)} characters from {pdf_path}")
        
        if len(final_text) < 200:
            logger.warning(f"Extracted text is suspiciously short ({len(final_text)} chars). PDF may be image-based or have security restrictions.")
        
        return final_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        traceback.print_exc()
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
    
    # Ensure 'skills' has all expected subsections
    if 'skills' in cv_data:
        expected_skill_sections = ['technical_skills', 'languages', 'soft_skills']
        for skill_section in expected_skill_sections:
            if skill_section not in cv_data['skills'] or not isinstance(cv_data['skills'][skill_section], list):
                logger.warning(f"Missing or invalid {skill_section} in skills")
                cv_data['skills'][skill_section] = []
    
    return cv_data

# --- Helper functions for email generation ---
def generate_email_template(cv_data: Dict, job_req: Dict, match_details: Dict) -> str:
    """
    Generate an email template based on match scores without using LLM to improve performance.
    This function uses pre-defined templates based on match criteria.
    """
    # Extract relevant information
    candidate_name = cv_data.get('personal_info', {}).get('name', 'Candidate')
    job_title = job_req.get('job_title', 'the position')
    company_name = job_req.get('company', 'Our Company')
    
    # Get match scores
    tech_score = match_details.get('technical_skills_match', 0)
    domain_score = match_details.get('domain_match', 0)
    
    # Extract strengths and missing skills
    strengths = match_details.get('strengths', [])
    missing_skills = match_details.get('missing_skills', [])
    
    # Create strength text for acceptance emails
    strengths_text = ""
    if strengths:
        strengths_text = "We were particularly impressed with "
        if len(strengths) == 1:
            strengths_text += f"your {strengths[0]}."
        else:
            strengths_text += f"your {', '.join(strengths[:-1])} and {strengths[-1]}."
    
    # Create missing skills text for rejection emails
    missing_skills_text = ""
    if missing_skills and len(missing_skills) > 0:
        missing_skills_text = "For your future applications, you may want to develop skills in "
        if len(missing_skills) == 1:
            missing_skills_text += f"{missing_skills[0]}."
        else:
            missing_skills_text += f"{', '.join(missing_skills[:-1])} and {missing_skills[-1]}."
    
    # Determine email type based on match scores
    if tech_score > 60 and domain_score > 60:
        # Acceptance/Interview Email
        subject = f"Interview Invitation for {job_title} at {company_name}"
        body = f"""Dear {candidate_name},

Thank you for your application for the {job_title} position at {company_name}. We have reviewed your qualifications and are pleased to invite you for an interview.

{strengths_text}

We would like to schedule an interview with you to discuss your background and the position in more detail. Please respond to this email with your availability in the next two weeks.

We look forward to speaking with you.

Best regards,
{company_name} Recruitment Team
"""
    else:
        # Rejection Email
        subject = f"Regarding your application for {job_title} at {company_name}"
        body = f"""Dear {candidate_name},

Thank you for your interest in the {job_title} position at {company_name} and for the time you invested in applying.

After careful consideration of your profile, we regret to inform you that we have decided to move forward with other candidates whose qualifications more closely match our current needs.

{missing_skills_text}

We appreciate your interest in {company_name} and wish you success in your job search.

Best regards,
{company_name} Recruitment Team
"""
    
    return f"Subject: {subject}\n\n{body}"

# --- CrewAI Tools ---

class PDFTextExtractorTool(BaseTool):
    name: str = "PDF Text Extractor"
    description: str = "Extracts all text content from a given PDF file path. Input must be the full path to the PDF."

    def _run(self, pdf_path: str) -> str:
        try:
            return improved_extract_text_from_pdf(pdf_path)
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
def initialize_llm():
    try:
        ollama_llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        # Test connection with a short prompt
        ollama_llm.invoke("OK")
        logger.info(f"Successfully connected to Ollama model '{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}")
        return ollama_llm
    except Exception as e:
        logger.error(f"Failed to initialize or connect to Ollama LLM: {e}")
        logger.error("Please ensure Ollama is running and the model is available.")
        raise RuntimeError(f"Failed to initialize LLM: {e}")

# --- Instantiate Tools ---
pdf_tool = PDFTextExtractorTool()
contact_tool = ContactInfoExtractorTool()
email_tool = EmailSenderTool()

# --- Define Agents ---
def create_agents(llm):
    cv_analyst = Agent(
        role='Expert CV Analyst',
        goal='Accurately parse candidate CVs (provided as text) and extract key information into a structured JSON format, including personal details, education, experience, skills (categorized), certifications, projects, and generate a professional summary.',
        backstory=(
            "You are a meticulous analyst with years of experience in HR tech. "
            "You specialize in transforming unstructured CV text into clean, structured data. "
            "You understand the nuances of CV layouts and terminology across various industries. "
            "Your goal is to provide data ready for comparison and analysis. "
            "You ALWAYS respond ONLY with the required JSON structure."
        ),
        llm=llm,
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
        llm=llm,
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
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    return cv_analyst, job_analyst, candidate_matcher

# --- Define Tasks ---
def create_tasks(cv_analyst, job_analyst, candidate_matcher):
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

    return analyze_cv, analyze_job_description, match_candidate

# --- Create Crew ---
def create_crew(cv_analyst, job_analyst, candidate_matcher, analyze_cv, analyze_job_description, match_candidate):
    recruitment_crew = Crew(
        agents=[cv_analyst, job_analyst, candidate_matcher],
        tasks=[analyze_cv, analyze_job_description, match_candidate],
        process=Process.sequential,
        verbose=2 # verbose=2 logs agent actions and tool usage
    )
    return recruitment_crew

# --- Process application function ---
def process_application(pdf_path, job_description):
    """
    Process a job application by analyzing CV, job description, and matching them.
    Returns structured data for all steps of the process.
    """
    try:
        # Extract text from CV with improved extraction
        logger.info(f"Extracting text from CV: {pdf_path}")
        try:
            cv_text = improved_extract_text_from_pdf(pdf_path)
            if not cv_text or len(cv_text) < 100:  # Minimum viable CV length
                return {"error": "Failed to extract sufficient text from CV. The PDF may be image-based or corrupted."}
        except Exception as e:
            logger.error(f"Error during CV text extraction: {str(e)}")
            return {"error": f"Failed to extract text from CV: {str(e)}"}

        # Extract basic contact info directly - as backup in case LLM fails
        basic_info = {
            "email": extract_email_from_text(cv_text),
            "phone": extract_phone_from_text(cv_text),
            "name": extract_name_from_text(cv_text)
        }
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        try:
            ollama_llm = initialize_llm()
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            
            # Create a minimal CV structure with the extracted basic info
            minimal_cv_data = {
                "personal_info": basic_info,
                "education": [],
                "experience": [],
                "skills": {"technical_skills": [], "languages": [], "soft_skills": []},
                "certifications": [],
                "projects": [],
                "summary": "CV processing failed due to LLM connection issues."
            }
            
            return {
                "error": f"LLM initialization failed: {str(e)}. Using minimal CV structure with extracted contact info.",
                "cv_data": minimal_cv_data,
                "job_data": {
                    "job_title": "Unknown Position",
                    "company": "Unknown Company",
                    "location": "",
                    "technical_skills": [],
                    "experience_years": "",
                    "education": [],
                    "soft_skills": [],
                    "languages": []
                },
                "match_data": {
                    "technical_skills_match": 0,
                    "education_match": 0,
                    "experience_match": 0,
                    "languages_match": 0,
                    "soft_skills_match": 0,
                    "domain_match": 0,
                    "missing_skills": [],
                    "strengths": [],
                    "final_assessment": "CV processing failed due to LLM connection issues. Cannot perform matching."
                },
                "email": {
                    "subject": f"Regarding your application",
                    "body": f"Dear {basic_info['name'] if basic_info['name'] else 'Candidate'},\n\nThank you for your application. We are currently reviewing your profile and will be in touch soon.\n\nBest regards,\nRecruitment Team"
                }
            }

        # Create agents and tasks
        logger.info("Creating agents and tasks...")
        cv_analyst, job_analyst, candidate_matcher = create_agents(ollama_llm)
        analyze_cv, analyze_job_description, match_candidate = create_tasks(
            cv_analyst, job_analyst, candidate_matcher
        )

        # Create crew
        logger.info("Creating crew...")
        recruitment_crew = create_crew(
            cv_analyst, 
            job_analyst, 
            candidate_matcher,
            analyze_cv, 
            analyze_job_description, 
            match_candidate
        )

        # Prepare inputs and run crew
        logger.info("Running recruitment crew...")
        inputs = {
            'cv_text': cv_text[:20000],  # Limit context window but allow more text than before
            'job_description': job_description[:15000]  # Limit context window
        }
        
        start_time = time.time()
        crew_result = recruitment_crew.kickoff(inputs=inputs)
        processing_time = time.time() - start_time
        logger.info(f"Crew execution completed in {processing_time:.2f} seconds")
        
        # Process results
        try:
            # Access task outputs
            cv_analysis_output = analyze_cv.output.raw_output if analyze_cv.output else "{}"
            job_analysis_output = analyze_job_description.output.raw_output if analyze_job_description.output else "{}"
            match_output = match_candidate.output.raw_output if match_candidate.output else "{}"
            
            logger.info("Parsing output JSONs...")
            cv_data_json = _extract_json_from_llm_response(cv_analysis_output)
            job_req_json = _extract_json_from_llm_response(job_analysis_output)
            match_details_json = _extract_json_from_llm_response(match_output)
            
            # Validate and repair data
            cv_data_json = validate_cv_data(cv_data_json)
            
            # If we have basic info but LLM didn't extract it, use our direct extraction
            if not cv_data_json['personal_info'].get('email') and basic_info['email']:
                cv_data_json['personal_info']['email'] = basic_info['email']
            if not cv_data_json['personal_info'].get('phone') and basic_info['phone']:
                cv_data_json['personal_info']['phone'] = basic_info['phone']
            if not cv_data_json['personal_info'].get('name') and basic_info['name']:
                cv_data_json['personal_info']['name'] = basic_info['name']
            
            # Generate email using the template function instead of LLM
            logger.info("Generating email template...")
            email_content = generate_email_template(cv_data_json, job_req_json, match_details_json)
            
            # Extract email subject and body
            email_subject_match = re.search(r"Subject:(.*?)(\n|\r\n)+", email_content, re.IGNORECASE)
            if email_subject_match:
                email_subject = email_subject_match.group(1).strip()
                subject_end_pos = email_subject_match.end()
                email_body = email_content[subject_end_pos:].strip()
            else:
                job_title = job_req_json.get("job_title", "the position")
                company_name = job_req_json.get("company", "Our Company")
                email_subject = f"Regarding your application for {job_title} at {company_name}"
                email_body = email_content.strip()
            
            # Return complete results
            return {
                "success": True,
                "cv_data": cv_data_json,
                "job_data": job_req_json,
                "match_data": match_details_json,
                "email": {
                    "subject": email_subject,
                    "body": email_body
                },
                "processing_time": f"{processing_time:.2f} seconds"
            }
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}", exc_info=True)
            # Use our basic extraction info if LLM parsing failed
            backup_cv = {
                "personal_info": basic_info,
                "education": [],
                "experience": [],
                "skills": {"technical_skills": [], "languages": [], "soft_skills": []},
                "certifications": [],
                "projects": [],
                "summary": "Failed to fully parse CV structure. Basic contact info extracted."
            }
            
            return {
                "partial_success": True,
                "error": f"Error processing results: {str(e)}",
                "cv_data": backup_cv,
                "job_data": {},
                "match_data": {
                    "technical_skills_match": 0,
                    "education_match": 0,
                    "experience_match": 0,
                    "languages_match": 0,
                    "soft_skills_match": 0,
                    "domain_match": 0,
                    "missing_skills": [],
                    "strengths": [],
                    "final_assessment": "Processing error occurred. Manual review required."
                },
                "email": {
                    "subject": f"Regarding your application",
                    "body": f"Dear {basic_info['name'] if basic_info['name'] else 'Candidate'},\n\nThank you for your application. We are currently reviewing your profile and will be in touch soon.\n\nBest regards,\nRecruitment Team"
                }
            }
            
    except Exception as e:
        logger.error(f"Error in process_application: {str(e)}", exc_info=True)
        return {"error": f"Application processing failed: {str(e)}"}

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

    print("\n⏳ Processing application...")
    result = process_application(pdf_path, job_description)
    
    if "error" in result and not result.get("partial_success"):
        print(f"❌ {result['error']}")
        return
        
    print("\n✅ Application processed successfully!")
    
    # Print summary
    cv_data = result["cv_data"]
    job_data = result.get("job_data", {})
    match_data = result.get("match_data", {})
    
    print("\n📊 Results Summary:")
    print(f"Candidate: {cv_data.get('personal_info', {}).get('name', 'N/A')}")
    print(f"Position: {job_data.get('job_title', 'N/A')} at {job_data.get('company', 'N/A')}")
    
    if "partial_success" in result:
        print("\n⚠️ Notice: Only partial data could be extracted. Some information may be missing.")
    
    print(f"Technical Skills Match: {match_data.get('technical_skills_match', 'N/A')}%")
    print(f"Domain Match: {match_data.get('domain_match', 'N/A')}%")
    print(f"Assessment: {match_data.get('final_assessment', 'N/A')}")
    
    # Show generated email
    print("\n📧 Generated Email:")
    print("-" * 80)
    print(f"Subject: {result['email']['subject']}")
    print(f"\n{result['email']['body']}")
    print("-" * 80)
    
    candidate_email = cv_data.get('personal_info', {}).get('email')
    if candidate_email:
        send_option = input(f"\n📤 Send this email to {candidate_email}? (y/n): ")
        if send_option.lower() == 'y':
            if not all([SMTP_CONFIG.get("smtp_server"), SMTP_CONFIG.get("sender_email"), SMTP_CONFIG.get("sender_password")]):
                print("❌ Cannot send email. SMTP settings incomplete in .env file or email_config.json.")
            else:
                email_data = {
                    "recipient_email": candidate_email,
                    "subject": result['email']['subject'],
                    "body": result['email']['body']
                }
                sender_tool = EmailSenderTool()
                send_status = sender_tool._run(email_data)
                print(f"✉️ Email Send Status: {send_status}")
    
    # Save results
    save_option = input("\n💾 Save results to JSON file? (y/n): ")
    if save_option.lower() == 'y':
        data_dir = os.getenv("DATA_DIR", "recruitment_data")
        os.makedirs(data_dir, exist_ok=True)
        
        candidate_name = cv_data.get('personal_info', {}).get('name', 'unknown_candidate')
        candidate_name = re.sub(r'[^\w\s-]', '', candidate_name).replace(' ', '_').lower()
        
        output_filename = os.path.join(data_dir, f"{candidate_name}_{Path(pdf_path).stem}_analysis.json")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"✅ Results saved to {output_filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    print("\n" + "=" * 80)
    print("🏁 Recruitment process finished.")
    print("=" * 80)

if __name__ == "__main__":
    main()