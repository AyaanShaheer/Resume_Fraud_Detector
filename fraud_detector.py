"""
Enhanced Fraudulent Candidate Detection Tool (Complete Implementation)

Features:
- Resume-only analysis with optional Job Description
- Experience parsing with date validation and overlap detection
- LinkedIn/public profile cross-verification
- Advanced fraud pattern detection
- AI-powered fit scoring and insights
- Comprehensive report generation
- Streamlit UI with visualizations

Key Deliverables Addressed:
1. Fraud Detection Engine ✓
2. LinkedIn Profile Verification ✓
3. Fit Scoring ✓
4. Report Generation ✓
"""

import io
import re
import json
import math
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any, Union
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

# UI Framework
try:
    import streamlit as st
except ImportError:
    st = None

# Core ML/NLP libraries
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    TfidfVectorizer = CountVectorizer = cosine_similarity = MinMaxScaler = None

# Advanced embeddings (optional)
_HAVE_SBERT = False
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_SBERT = True
except ImportError:
    pass

# Document processing
try:
    import docx
except ImportError:
    docx = None

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except ImportError:
    pdf_extract_text = None

# AI Integration
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    plt = None

# Date parsing
try:
    from dateutil import parser as date_parser
    from dateutil.relativedelta import relativedelta
except ImportError:
    date_parser = None
    relativedelta = None

# Web scraping (for LinkedIn verification)
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = BeautifulSoup = None


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class FraudFlag:
    """Represents a detected fraud indicator"""
    flag_type: str
    severity: str  # HIGH, MEDIUM, LOW
    description: str
    evidence: str
    confidence_score: float
    category: str
    recommendation: str = ""


@dataclass
class ExperienceEntry:
    """Structured experience entry"""
    text: str
    company: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_months: Optional[int] = None
    is_full_time: bool = True
    is_current: bool = False
    skills_mentioned: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)


@dataclass
class EducationEntry:
    """Structured education entry"""
    text: str
    degree: Optional[str] = None
    institution: Optional[str] = None
    field: Optional[str] = None
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None


@dataclass
class CandidateProfile:
    """Complete candidate profile"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    locations: List[str] = field(default_factory=list)
    education: List[EducationEntry] = field(default_factory=list)
    experience: List[ExperienceEntry] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    total_experience_years: float = 0.0
    career_level: str = "Entry"  # Entry, Mid, Senior, Executive
    

@dataclass
class FitScore:
    """Fit scoring results"""
    overall_score: float
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    keyword_density_score: float
    semantic_similarity_score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    profile: CandidateProfile
    flags: List[FraudFlag]
    fit_score: Optional[FitScore]
    verification_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    verdict: str
    summary: str
    recommendations: List[str]
    ai_insights: Optional[str] = None


# =============================================================================
# ENHANCED FRAUD DETECTION ENGINE
# =============================================================================

class EnhancedFraudDetector:
    """Advanced fraud detection with comprehensive pattern matching"""
    
    # Enhanced pattern libraries
    MONTHS = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9, 'oct': 10,
        'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    TECHNICAL_SKILLS = {
        'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin'],
        'web': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'express'],
        'data': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'spark'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
        'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch']
    }
    
    LEADERSHIP_INDICATORS = {
        'explicit': ['manager', 'director', 'vp', 'ceo', 'cto', 'head of', 'lead', 'principal'],
        'actions': ['managed', 'led', 'supervised', 'mentored', 'architected', 'owned']
    }
    
    SUSPICIOUS_PATTERNS = {
        'generic_descriptions': ['responsible for', 'worked on', 'involved in', 'participated'],
        'vague_achievements': ['improved', 'enhanced', 'optimized', 'streamlined'],
        'buzzwords': ['synergy', 'paradigm', 'leverage', 'utilize', 'facilitate']
    }

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        self.sentence_model = None
        
        # Initialize Gemini if available
        if gemini_api_key and genai:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            except Exception as e:
                logging.error(f"Gemini initialization failed: {e}")
        
        # Initialize sentence transformers if available
        if _HAVE_SBERT:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logging.error(f"SentenceTransformer initialization failed: {e}")

    # =========================================================================
    # DOCUMENT PROCESSING
    # =========================================================================
    
    def extract_text_from_file(self, file_bytes: bytes, filename: str) -> str:
        """Enhanced document text extraction"""
        filename_lower = filename.lower()
        
        try:
            if filename_lower.endswith('.txt'):
                return file_bytes.decode('utf-8', errors='ignore')
            
            elif filename_lower.endswith('.pdf') and pdf_extract_text:
                with io.BytesIO(file_bytes) as pdf_file:
                    text = pdf_extract_text(pdf_file)
                    return text if text else ""
            
            elif filename_lower.endswith('.docx') and docx:
                doc = docx.Document(io.BytesIO(file_bytes))
                paragraphs = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        paragraphs.append(para.text.strip())
                return '\n'.join(paragraphs)
            
            else:
                # Fallback to UTF-8 decoding
                return file_bytes.decode('utf-8', errors='ignore')
                
        except Exception as e:
            logging.error(f"File extraction error: {e}")
            return file_bytes.decode('latin-1', errors='ignore')

    # =========================================================================
    # ENHANCED PROFILE PARSING
    # =========================================================================
    
    def parse_candidate_profile(self, text: str) -> CandidateProfile:
        """Advanced profile parsing with structured data extraction"""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        profile = CandidateProfile()
        
        # Extract basic info
        profile.name = self._extract_name(lines)
        profile.email = self._extract_email(text)
        profile.phone = self._extract_phone(text)
        profile.locations = self._extract_locations(text)
        
        # Extract structured data
        profile.education = self._parse_education(text)
        profile.experience = self._parse_experience(text)
        profile.skills = self._extract_skills(text)
        profile.certifications = self._extract_certifications(text)
        profile.languages = self._extract_languages(text)
        
        # Calculate derived metrics
        profile.total_experience_years = self._calculate_total_experience(profile.experience)
        profile.career_level = self._determine_career_level(profile)
        
        return profile
    
    def _extract_name(self, lines: List[str]) -> Optional[str]:
        """Extract candidate name from document header"""
        for line in lines[:5]:
            # Skip lines with email or phone
            if '@' in line or re.search(r'\d{3,}', line):
                continue
            
            # Look for name patterns
            words = line.split()
            if 2 <= len(words) <= 4 and all(word.replace('.', '').isalpha() for word in words):
                return line
        return None
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, text)
        return matches[0] if matches else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number"""
        pattern = r'(\+?\d[\d\s\-\(\)]{7,}\d)'
        matches = re.findall(pattern, text)
        return matches[0] if matches else None
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract location information"""
        locations = []
        
        # Common location patterns
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',  # City, State
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+)\b',  # City, Country
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                location = f"{match[0]}, {match[1]}"
                if location not in locations:
                    locations.append(location)
        
        return locations[:5]  # Limit to 5 locations
    
    def _parse_education(self, text: str) -> List[EducationEntry]:
        """Parse education entries"""
        education_entries = []
        
        # Education keywords
        edu_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'diploma', 'certificate',
            'b.s.', 'm.s.', 'b.a.', 'm.a.', 'mba', 'btech', 'mtech'
        ]
        
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in edu_keywords):
                # Extract education details
                entry = EducationEntry(text=line.strip())
                
                # Try to extract degree, institution, year
                entry.degree = self._extract_degree(line)
                entry.institution = self._extract_institution(line, lines, i)
                entry.graduation_year = self._extract_year(line)
                entry.field = self._extract_field_of_study(line)
                entry.gpa = self._extract_gpa(line)
                
                education_entries.append(entry)
        
        return education_entries
    
    def _parse_experience(self, text: str) -> List[ExperienceEntry]:
        """Parse work experience entries"""
        experience_entries = []
        
        # Date range pattern
        date_pattern = r'(?P<start>(?:\d{1,2}[/-]\d{4}|[A-Za-z]{3,9}\s+\d{4}|\d{4}))\s*[-–—]\s*(?P<end>(?:\d{1,2}[/-]\d{4}|[A-Za-z]{3,9}\s+\d{4}|\d{4}|present|current))'
        
        lines = text.splitlines()
        i = 0
        
        while i < len(lines):
            line = lines[i]
            date_match = re.search(date_pattern, line, re.IGNORECASE)
            
            if date_match:
                # Found a date range, parse the experience block
                entry = self._parse_experience_block(lines, i, date_match)
                if entry:
                    experience_entries.append(entry)
                i += 1
            else:
                # Look for role-like lines
                if self._looks_like_job_title(line):
                    entry = self._parse_experience_without_dates(lines, i)
                    if entry:
                        experience_entries.append(entry)
                i += 1
        
        return experience_entries[:15]  # Limit to 15 entries
    
    def _parse_experience_block(self, lines: List[str], start_idx: int, date_match) -> Optional[ExperienceEntry]:
        """Parse a complete experience block"""
        entry = ExperienceEntry(text="")
        
        # Parse dates
        entry.start_date = self._parse_date(date_match.group('start'))
        entry.end_date = self._parse_date(date_match.group('end'))
        
        if entry.start_date and entry.end_date:
            entry.duration_months = self._calculate_months_between(entry.start_date, entry.end_date)
            entry.is_current = date_match.group('end').lower() in ['present', 'current']
        
        # Collect the experience block text
        block_lines = [lines[start_idx]]
        
        # Look ahead for related content
        for i in range(start_idx + 1, min(start_idx + 8, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
            
            # Stop if we hit another date range
            if re.search(r'\d{4}.*[-–—].*\d{4}', line):
                break
                
            # Stop if we hit what looks like a new section
            if re.match(r'^[A-Z][A-Za-z\s]+:?$', line) and len(line) < 30:
                break
                
            block_lines.append(line)
        
        full_text = ' '.join(block_lines)
        entry.text = full_text
        
        # Extract structured information
        entry.company = self._extract_company(full_text)
        entry.role = self._extract_role(full_text)
        entry.is_full_time = self._infer_employment_type(full_text)
        entry.skills_mentioned = self._extract_skills_from_text(full_text)
        entry.achievements = self._extract_achievements(full_text)
        
        return entry
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical and soft skills"""
        skills = set()
        text_lower = text.lower()
        
        # Look for explicit skills sections
        skill_patterns = [
            r'(?:technical\s+)?skills?\s*[:\-]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'technologies?\s*[:\-]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'competencies\s*[:\-]\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by common delimiters
                skill_items = re.split(r'[,;|•\n]', match)
                for item in skill_items:
                    item = item.strip()
                    if item and len(item) < 50:  # Reasonable skill name length
                        skills.add(item.lower())
        
        # If no explicit skills section, look for technical terms
        if not skills:
            for category, skill_list in self.TECHNICAL_SKILLS.items():
                for skill in skill_list:
                    if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                        skills.add(skill)
        
        return list(skills)[:50]  # Limit to 50 skills
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        cert_patterns = [
            r'certifications?\s*[:\-]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'certified\s+(.+?)(?:\n|\s-|\s\()',
            r'\b[A-Z]{2,}\s+certified\b',
        ]
        
        certifications = set()
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and len(match) < 100:
                    certifications.add(match.strip())
        
        return list(certifications)
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract language skills"""
        lang_patterns = [
            r'languages?\s*[:\-]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'\b(english|spanish|french|german|chinese|japanese|korean|hindi|arabic|russian|portuguese|italian)\b'
        ]
        
        languages = set()
        for pattern in lang_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    # Split by common delimiters
                    lang_items = re.split(r'[,;|•\n]', match)
                    for item in lang_items:
                        item = item.strip().lower()
                        if item and len(item) < 20:
                            languages.add(item.title())
        
        return list(languages)
    
    # =========================================================================
    # FRAUD DETECTION ALGORITHMS
    # =========================================================================
    
    def detect_timeline_inconsistencies(self, profile: CandidateProfile) -> List[FraudFlag]:
        """Detect timeline and chronological issues"""
        flags = []
        
        # Check for overlapping full-time positions
        dated_experiences = [exp for exp in profile.experience if exp.start_date and exp.end_date]
        
        for i in range(len(dated_experiences)):
            for j in range(i + 1, len(dated_experiences)):
                exp1, exp2 = dated_experiences[i], dated_experiences[j]
                
                if self._experiences_overlap(exp1, exp2):
                    overlap_months = self._calculate_overlap_months(exp1, exp2)
                    
                    if exp1.is_full_time and exp2.is_full_time and overlap_months >= 2:
                        flags.append(FraudFlag(
                            flag_type="Overlapping Full-Time Positions",
                            severity="HIGH",
                            description=f"Two full-time positions overlap by {overlap_months} months",
                            evidence=f"{exp1.role} at {exp1.company} and {exp2.role} at {exp2.company}",
                            confidence_score=0.9,
                            category="Timeline",
                            recommendation="Verify employment dates and ask about concurrent roles"
                        ))
        
        # Check for unrealistic experience progression
        total_years = profile.total_experience_years
        if total_years > 50:
            flags.append(FraudFlag(
                flag_type="Unrealistic Experience Duration",
                severity="MEDIUM",
                description=f"Claims {total_years} years of experience",
                evidence="Calculated from resume dates",
                confidence_score=0.7,
                category="Timeline",
                recommendation="Verify actual work history and dates"
            ))
        
        return flags
    
    def detect_skill_inflation(self, profile: CandidateProfile) -> List[FraudFlag]:
        """Detect inflated or suspicious skill claims"""
        flags = []
        
        # Check for overly broad skill set
        if len(profile.skills) > 40:
            flags.append(FraudFlag(
                flag_type="Overly Broad Skill Set",
                severity="LOW",
                description=f"Lists {len(profile.skills)} skills - may indicate lack of focus",
                evidence=f"Skills: {', '.join(profile.skills[:10])}...",
                confidence_score=0.6,
                category="Skills",
                recommendation="Assess depth of knowledge in key skills through technical interview"
            ))
        
        # Check for leadership claims in junior roles
        for exp in profile.experience:
            if self._is_junior_role(exp) and self._has_leadership_claims(exp):
                flags.append(FraudFlag(
                    flag_type="Leadership Claims in Junior Role",
                    severity="HIGH",
                    description="Claims leadership responsibilities in intern/junior position",
                    evidence=exp.text[:200],
                    confidence_score=0.85,
                    category="Experience",
                    recommendation="Verify scope of responsibilities and team size"
                ))
        
        return flags
    
    def detect_education_inconsistencies(self, profile: CandidateProfile, jd_text: str = "") -> List[FraudFlag]:
        """Detect education-related fraud indicators"""
        flags = []
        
        # Check for degree-role mismatch
        if jd_text:
            required_background = self.analyze_required_background(jd_text)
            candidate_background = self.analyze_candidate_background(profile)
            
            if required_background and candidate_background:
                if not self._backgrounds_compatible(required_background, candidate_background):
                    flags.append(FraudFlag(
                        flag_type="Education-Role Mismatch",
                        severity="MEDIUM",
                        description="Educational background doesn't align with role requirements",
                        evidence=f"Required: {required_background}, Candidate: {candidate_background}",
                        confidence_score=0.7,
                        category="Education",
                        recommendation="Assess transferable skills and relevant experience"
                    ))
        
        # Check for unrealistic GPA claims
        for edu in profile.education:
            if edu.gpa and edu.gpa > 4.0:
                flags.append(FraudFlag(
                    flag_type="Unrealistic GPA",
                    severity="MEDIUM",
                    description=f"Claims GPA of {edu.gpa} (above 4.0 scale)",
                    evidence=edu.text,
                    confidence_score=0.8,
                    category="Education",
                    recommendation="Verify GPA scale and request official transcripts"
                ))
        
        return flags
    
    def detect_plagiarism_indicators(self, resume_text: str, jd_text: str = "") -> List[FraudFlag]:
        """Detect potential plagiarism from job descriptions or templates"""
        flags = []
        
        if not jd_text:
            return flags
        
        # Check for excessive overlap with job description
        overlap_score = self._calculate_text_overlap(resume_text, jd_text)
        
        if overlap_score > 0.3:
            severity = "HIGH" if overlap_score > 0.5 else "MEDIUM"
            flags.append(FraudFlag(
                flag_type="Job Description Plagiarism",
                severity=severity,
                description=f"Resume shows {overlap_score:.1%} textual overlap with job description",
                evidence="Shared phrases and sentences detected",
                confidence_score=0.8 + (overlap_score - 0.3) * 0.2,
                category="Plagiarism",
                recommendation="Request original work samples and clarify experience"
            ))
        
        # Check for generic template usage
        generic_score = self._assess_generic_content(resume_text)
        if generic_score > 0.7:
            flags.append(FraudFlag(
                flag_type="Generic Template Usage",
                severity="LOW",
                description="Resume contains many generic phrases and templates",
                evidence="High usage of common resume templates",
                confidence_score=generic_score,
                category="Authenticity",
                recommendation="Look for specific, quantified achievements"
            ))
        
        return flags
    
    def detect_verification_discrepancies(self, profile: CandidateProfile, linkedin_text: str = "") -> List[FraudFlag]:
        """Compare resume with LinkedIn or other public profiles"""
        flags = []
        
        if not linkedin_text:
            return flags
        
        linkedin_profile = self.parse_candidate_profile(linkedin_text)
        
        # Compare companies
        resume_companies = {exp.company for exp in profile.experience if exp.company}
        linkedin_companies = {exp.company for exp in linkedin_profile.experience if exp.company}
        
        missing_companies = resume_companies - linkedin_companies
        if len(missing_companies) >= 2:
            flags.append(FraudFlag(
                flag_type="Company Verification Mismatch",
                severity="MEDIUM",
                description=f"Companies on resume not found in LinkedIn profile",
                evidence=f"Missing: {', '.join(list(missing_companies)[:3])}",
                confidence_score=0.75,
                category="Verification",
                recommendation="Verify employment history through references"
            ))
        
        # Compare skills
        resume_skills = set(skill.lower() for skill in profile.skills)
        linkedin_skills = set(skill.lower() for skill in linkedin_profile.skills)
        
        skill_discrepancy = len(resume_skills - linkedin_skills)
        if skill_discrepancy > 10:
            flags.append(FraudFlag(
                flag_type="Skill Verification Mismatch",
                severity="LOW",
                description=f"Many skills on resume not reflected in LinkedIn profile",
                evidence=f"{skill_discrepancy} skills not found in LinkedIn",
                confidence_score=0.6,
                category="Verification",
                recommendation="Assess skill claims through technical evaluation"
            ))
        
        return flags
    
    # =========================================================================
    # FIT SCORING ENGINE
    # =========================================================================
    
    def calculate_fit_score(self, profile: CandidateProfile, jd_text: str) -> FitScore:
        """Calculate comprehensive fit score"""
        if not jd_text.strip():
            return FitScore(
                overall_score=0.0,
                skill_match_score=0.0,
                experience_match_score=0.0,
                education_match_score=0.0,
                keyword_density_score=0.0
            )
        
        # Extract job requirements
        jd_requirements = self._extract_job_requirements(jd_text)
        
        # Calculate component scores
        skill_score = self._calculate_skill_match(profile.skills, jd_requirements['skills'])
        exp_score = self._calculate_experience_match(profile.experience, jd_requirements['experience'])
        edu_score = self._calculate_education_match(profile.education, jd_requirements['education'])
        keyword_score = self._calculate_keyword_density(profile, jd_text)
        
        # Semantic similarity using embeddings
        semantic_score = None
        if self.sentence_model:
            try:
                resume_text = self._profile_to_text(profile)
                embeddings = self.sentence_model.encode([resume_text, jd_text])
                semantic_score = float(np.dot(embeddings[0], embeddings[1]))
            except Exception:
                pass
        
        # Calculate overall score (weighted average)
        weights = {
            'skills': 0.3,
            'experience': 0.3,
            'education': 0.15,
            'keywords': 0.15,
            'semantic': 0.1
        }
        
        overall = (
            skill_score * weights['skills'] +
            exp_score * weights['experience'] +
            edu_score * weights['education'] +
            keyword_score * weights['keywords'] +
            (semantic_score or 0) * weights['semantic']
        )
        
        return FitScore(
            overall_score=round(overall * 100, 1),
            skill_match_score=round(skill_score * 100, 1),
            experience_match_score=round(exp_score * 100, 1),
            education_match_score=round(edu_score * 100, 1),
            keyword_density_score=round(keyword_score * 100, 1),
            semantic_similarity_score=round(semantic_score * 100, 1) if semantic_score else None,
            details={
                'jd_requirements': jd_requirements,
                'weights_used': weights
            }
        )
    
    def _calculate_skill_match(self, candidate_skills: List[str], required_skills: List[str]) -> float:
        """Calculate skill matching score"""
        if not required_skills:
            return 0.5  # Neutral score if no requirements
        
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        required_skills_lower = [skill.lower() for skill in required_skills]
        
        matches = 0
        for req_skill in required_skills_lower:
            if any(req_skill in cand_skill or cand_skill in req_skill for cand_skill in candidate_skills_lower):
                matches += 1
        
        return min(matches / len(required_skills), 1.0)
    
    def _calculate_experience_match(self, experiences: List[ExperienceEntry], required_exp: Dict[str, Any]) -> float:
        """Calculate experience matching score"""
        if not required_exp:
            return 0.5
        
        score = 0.0
        factors = 0
        
        # Years of experience
        if 'years' in required_exp and required_exp['years']:
            candidate_years = sum(exp.duration_months or 0 for exp in experiences) / 12
            required_years = required_exp['years']
            year_score = min(candidate_years / required_years, 1.0)
            score += year_score
            factors += 1
        
        # Industry experience
        if 'industries' in required_exp and required_exp['industries']:
            industry_match = self._calculate_industry_match(experiences, required_exp['industries'])
            score += industry_match
            factors += 1
        
        # Role level
        if 'level' in required_exp and required_exp['level']:
            level_match = self._calculate_level_match(experiences, required_exp['level'])
            score += level_match
            factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def _calculate_education_match(self, education: List[EducationEntry], required_edu: Dict[str, Any]) -> float:
        """Calculate education matching score"""
        if not required_edu or not education:
            return 0.5
        
        score = 0.0
        
        # Degree level match
        if 'degree_level' in required_edu:
            candidate_level = self._get_highest_degree_level(education)
            required_level = required_edu['degree_level']
            if candidate_level >= required_level:
                score += 0.5
        
        # Field match
        if 'fields' in required_edu:
            field_match = self._calculate_field_match(education, required_edu['fields'])
            score += field_match * 0.5
        
        return min(score, 1.0)
    
    def _calculate_keyword_density(self, profile: CandidateProfile, jd_text: str) -> float:
        """Calculate keyword density score using TF-IDF"""
        if not TfidfVectorizer or not cosine_similarity:
            return 0.5
        
        try:
            resume_text = self._profile_to_text(profile)
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.5
    
    # =========================================================================
    # AI INSIGHTS GENERATION
    # =========================================================================
    
    def generate_ai_insights(self, profile: CandidateProfile, flags: List[FraudFlag], 
                           fit_score: Optional[FitScore], jd_text: str = "") -> str:
        """Generate AI-powered insights using Gemini"""
        if not self.gemini_model:
            return ""
        
        try:
            resume_text = self._profile_to_text(profile)[:3000]  # Truncate for API limits
            flags_summary = "\n".join([f"- {f.flag_type} ({f.severity}): {f.description}" 
                                     for f in flags[:10]])
            
            prompt = f"""
            As an expert recruiter, analyze this candidate profile and provide insights:

            CANDIDATE PROFILE:
            Name: {profile.name or 'Not specified'}
            Experience: {profile.total_experience_years} years
            Skills: {', '.join(profile.skills[:15])}
            Education: {len(profile.education)} entries
            
            RESUME TEXT (truncated):
            {resume_text}
            
            JOB DESCRIPTION:
            {jd_text[:2000] if jd_text else 'Not provided'}
            
            DETECTED FLAGS:
            {flags_summary or 'None detected'}
            
            FIT SCORE: {fit_score.overall_score if fit_score else 'Not calculated'}
            
            Please provide:
            1. Overall candidate assessment (2-3 sentences)
            2. Top 3 strengths
            3. Top 3 concerns or areas to verify
            4. Interview recommendations
            5. Final hiring recommendation
            
            Be concise and actionable.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"AI insights unavailable: {str(e)}"
    
    # =========================================================================
    # MAIN ANALYSIS ORCHESTRATION
    # =========================================================================
    
    def analyze_candidate(self, resume_text: str, jd_text: str = "", 
                         linkedin_text: str = "") -> AnalysisResult:
        """Main analysis orchestration method"""
        
        # Parse candidate profile
        profile = self.parse_candidate_profile(resume_text)
        
        # Run all fraud detection algorithms
        all_flags = []
        
        all_flags.extend(self.detect_timeline_inconsistencies(profile))
        all_flags.extend(self.detect_skill_inflation(profile))
        all_flags.extend(self.detect_education_inconsistencies(profile, jd_text))
        all_flags.extend(self.detect_plagiarism_indicators(resume_text, jd_text))
        all_flags.extend(self.detect_verification_discrepancies(profile, linkedin_text))
        
        # Calculate fit score
        fit_score = None
        if jd_text.strip():
            fit_score = self.calculate_fit_score(profile, jd_text)
        
        # Calculate verification score
        verification_score = self._calculate_verification_score(all_flags, linkedin_text)
        
        # Determine risk level and verdict
        risk_level = self._determine_risk_level(all_flags, fit_score)
        verdict = self._generate_verdict(risk_level, all_flags, fit_score)
        
        # Generate summary and recommendations
        summary = self._generate_summary(all_flags, risk_level)
        recommendations = self._generate_recommendations(all_flags, risk_level)
        
        # Generate AI insights
        ai_insights = self.generate_ai_insights(profile, all_flags, fit_score, jd_text)
        
        return AnalysisResult(
            profile=profile,
            flags=all_flags,
            fit_score=fit_score,
            verification_score=verification_score,
            risk_level=risk_level,
            verdict=verdict,
            summary=summary,
            recommendations=recommendations,
            ai_insights=ai_insights
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        if not date_str:
            return None
            
        date_str = date_str.strip().lower()
        
        if date_str in ['present', 'current', 'now']:
            return datetime.now()
        
        # Try various patterns
        patterns = [
            r'(\d{1,2})[/-](\d{4})',  # MM/YYYY or MM-YYYY
            r'([a-z]{3,9})\s+(\d{4})',  # Month YYYY
            r'(\d{4})',  # YYYY only
        ]
        
        for pattern in patterns:
            match = re.match(pattern, date_str)
            if match:
                if len(match.groups()) == 2:
                    if match.group(1).isdigit():
                        # MM/YYYY format
                        month = int(match.group(1))
                        year = int(match.group(2))
                        month = max(1, min(12, month))  # Clamp month
                        return datetime(year, month, 1)
                    else:
                        # Month YYYY format
                        month_name = match.group(1)
                        if month_name in self.MONTHS:
                            year = int(match.group(2))
                            return datetime(year, self.MONTHS[month_name], 1)
                else:
                    # YYYY only
                    year = int(match.group(1))
                    return datetime(year, 1, 1)
        
        # Fallback to dateutil if available
        if date_parser:
            try:
                return date_parser.parse(date_str, fuzzy=True)
            except Exception:
                pass
        
        return None
    
    def _calculate_months_between(self, start: datetime, end: datetime) -> int:
        """Calculate months between two dates"""
        if not start or not end:
            return 0
        return (end.year - start.year) * 12 + (end.month - start.month)
    
    def _calculate_total_experience(self, experiences: List[ExperienceEntry]) -> float:
        """Calculate total years of experience"""
        if not experiences:
            return 0.0
        
        total_months = 0
        for exp in experiences:
            if exp.duration_months:
                total_months += exp.duration_months
            elif exp.start_date and exp.end_date:
                months = self._calculate_months_between(exp.start_date, exp.end_date)
                total_months += months
        
        return round(total_months / 12.0, 1)
    
    def _determine_career_level(self, profile: CandidateProfile) -> str:
        """Determine career level based on experience and roles"""
        years = profile.total_experience_years
        
        # Check for leadership roles
        has_leadership = any(
            self._has_leadership_indicators(exp.text) for exp in profile.experience
        )
        
        if years < 2:
            return "Entry"
        elif years < 5:
            return "Junior" if not has_leadership else "Mid"
        elif years < 10:
            return "Mid" if not has_leadership else "Senior"
        else:
            return "Senior" if not has_leadership else "Executive"
    
    def _has_leadership_indicators(self, text: str) -> bool:
        """Check if text contains leadership indicators"""
        text_lower = text.lower()
        
        for indicator in self.LEADERSHIP_INDICATORS['explicit']:
            if indicator in text_lower:
                return True
        
        for action in self.LEADERSHIP_INDICATORS['actions']:
            if action in text_lower:
                return True
        
        return False
    
    def _calculate_verification_score(self, flags: List[FraudFlag], linkedin_text: str) -> float:
        """Calculate overall verification score"""
        base_score = 100.0
        
        # Deduct points for flags
        for flag in flags:
            if flag.severity == "HIGH":
                base_score -= 25
            elif flag.severity == "MEDIUM":
                base_score -= 15
            else:
                base_score -= 5
        
        # Bonus for LinkedIn verification
        if linkedin_text:
            base_score += 10
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_risk_level(self, flags: List[FraudFlag], fit_score: Optional[FitScore]) -> str:
        """Determine overall risk level"""
        high_flags = [f for f in flags if f.severity == "HIGH"]
        medium_flags = [f for f in flags if f.severity == "MEDIUM"]
        
        if len(high_flags) >= 2:
            return "CRITICAL"
        elif len(high_flags) >= 1 and len(medium_flags) >= 2:
            return "HIGH"
        elif len(high_flags) >= 1 or len(medium_flags) >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_verdict(self, risk_level: str, flags: List[FraudFlag], 
                         fit_score: Optional[FitScore]) -> str:
        """Generate final verdict"""
        verdicts = {
            "CRITICAL": "REJECT - Critical fraud indicators detected",
            "HIGH": "PROCEED WITH EXTREME CAUTION - Verify all claims",
            "MEDIUM": "PROCEED WITH CAUTION - Additional verification needed",
            "LOW": "SAFE TO PROCEED - Minor concerns only"
        }
        
        base_verdict = verdicts.get(risk_level, "UNKNOWN")
        
        # Modify based on fit score
        if fit_score and fit_score.overall_score:
            if fit_score.overall_score >= 80 and risk_level in ["LOW", "MEDIUM"]:
                base_verdict += " - Strong candidate fit"
            elif fit_score.overall_score < 50:
                base_verdict += " - Poor role fit"
        
        return base_verdict
    
    def _generate_summary(self, flags: List[FraudFlag], risk_level: str) -> str:
        """Generate analysis summary"""
        if not flags:
            return "No significant fraud indicators detected. Candidate appears authentic."
        
        summary_parts = [f"Risk Level: {risk_level}"]
        
        by_severity = {}
        for flag in flags:
            if flag.severity not in by_severity:
                by_severity[flag.severity] = []
            by_severity[flag.severity].append(flag)
        
        for severity in ["HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                count = len(by_severity[severity])
                summary_parts.append(f"{count} {severity} severity flag{'s' if count > 1 else ''}")
        
        return " | ".join(summary_parts)
    
    def _generate_recommendations(self, flags: List[FraudFlag], risk_level: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_level == "CRITICAL":
            recommendations.append("Do not proceed with this candidate")
            recommendations.append("Consider reporting if fraud is suspected")
        
        elif risk_level == "HIGH":
            recommendations.append("Require extensive verification before proceeding")
            recommendations.append("Request official documentation for all claims")
            recommendations.append("Conduct thorough reference checks")
        
        elif risk_level == "MEDIUM":
            recommendations.append("Verify specific claims mentioned in flags")
            recommendations.append("Conduct detailed technical/behavioral interviews")
            recommendations.append("Check employment history with previous employers")
        
        else:
            recommendations.append("Proceed with standard interview process")
            recommendations.append("Address minor concerns during interview")
        
        # Add specific recommendations based on flag types
        flag_types = {f.flag_type for f in flags}
        
        if "Job Description Plagiarism" in flag_types:
            recommendations.append("Request original work samples")
        
        if "Overlapping Full-Time Positions" in flag_types:
            recommendations.append("Clarify employment timeline and commitments")
        
        if "Skill Verification Mismatch" in flag_types:
            recommendations.append("Conduct hands-on technical assessment")
        
        return recommendations
    
    # Additional utility methods for missing functionality
    def _profile_to_text(self, profile: CandidateProfile) -> str:
        """Convert profile to text for analysis"""
        parts = []
        
        if profile.name:
            parts.append(f"Name: {profile.name}")
        
        if profile.skills:
            parts.append(f"Skills: {', '.join(profile.skills)}")
        
        for exp in profile.experience:
            parts.append(exp.text)
        
        for edu in profile.education:
            parts.append(edu.text)
        
        return "\n".join(parts)
    
    def _extract_job_requirements(self, jd_text: str) -> Dict[str, Any]:
        """Extract requirements from job description"""
        requirements = {
            'skills': [],
            'experience': {},
            'education': {}
        }
        
        # Extract skills (basic implementation)
        jd_lower = jd_text.lower()
        for category, skills in self.TECHNICAL_SKILLS.items():
            for skill in skills:
                if skill in jd_lower:
                    requirements['skills'].append(skill)
        
        # Extract years of experience
        years_match = re.search(r'(\d+)\+?\s*years?\s+(?:of\s+)?experience', jd_text, re.IGNORECASE)
        if years_match:
            requirements['experience']['years'] = int(years_match.group(1))
        
        return requirements
    
    def _looks_like_job_title(self, line: str) -> bool:
        """Check if line looks like a job title"""
        line_lower = line.lower()
        job_keywords = ['engineer', 'developer', 'analyst', 'manager', 'director', 'specialist', 'consultant']
        return any(keyword in line_lower for keyword in job_keywords)
    
    def _parse_experience_without_dates(self, lines: List[str], start_idx: int) -> Optional[ExperienceEntry]:
        """Parse experience entry without explicit dates"""
        if start_idx >= len(lines):
            return None
        
        entry = ExperienceEntry(text=lines[start_idx])
        entry.role = self._extract_role(lines[start_idx])
        entry.is_full_time = True  # Default assumption
        
        return entry
    
    def _extract_company(self, text: str) -> Optional[str]:
        """Extract company name from text"""
        # Simple pattern matching for company names
        company_patterns = [
            r'\bat\s+([A-Z][A-Za-z\s&.,]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Company)?)',
            r'([A-Z][A-Za-z\s&.,]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Company))'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                company = match.group(1).strip()
                if len(company) < 50:  # Reasonable company name length
                    return company
        
        return None
    
    def _extract_role(self, text: str) -> Optional[str]:
        """Extract job role from text"""
        # Look for common job title patterns
        role_patterns = [
            r'^([A-Za-z\s]+(?:Engineer|Developer|Analyst|Manager|Director|Specialist|Consultant))',
            r'(?:as\s+)?([A-Za-z\s]+(?:Engineer|Developer|Analyst|Manager|Director|Specialist|Consultant))'
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                if len(role) < 50:
                    return role
        
        return None
    
    def _infer_employment_type(self, text: str) -> bool:
        """Infer if position was full-time"""
        text_lower = text.lower()
        
        part_time_indicators = ['intern', 'internship', 'part-time', 'contract', 'freelance', 'temporary']
        full_time_indicators = ['full-time', 'permanent', 'staff']
        
        if any(indicator in text_lower for indicator in part_time_indicators):
            return False
        elif any(indicator in text_lower for indicator in full_time_indicators):
            return True
        
        # Default to full-time if unclear
        return True
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills mentioned in text"""
        skills = []
        text_lower = text.lower()
        
        # Check against known technical skills
        for category, skill_list in self.TECHNICAL_SKILLS.items():
            for skill in skill_list:
                if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                    skills.append(skill)
        
        return skills
    
    def _extract_achievements(self, text: str) -> List[str]:
        """Extract achievements from text"""
        achievements = []
        
        # Look for quantified achievements
        patterns = [
            r'(?:increased|improved|reduced|achieved|delivered|saved|generated)\s+[^.]*\d+[%$]?[^.]*',
            r'\d+[%$]\s+(?:increase|improvement|reduction|growth|savings?)',
            r'(?:led|managed)\s+(?:team of\s+)?\d+\s+(?:people|engineers|developers)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            achievements.extend(matches[:3])  # Limit to 3 per pattern
        
        return achievements[:10]  # Overall limit
    
    def _extract_degree(self, text: str) -> Optional[str]:
        """Extract degree from education text"""
        degree_patterns = [
            r'\b(Bachelor|Master|PhD|Doctorate|B\.?[ASE]\.?|M\.?[ASE]\.?|MBA|PhD)\b[^,\n]*',
        ]
        
        for pattern in degree_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return None
    
    def _extract_institution(self, line: str, lines: List[str], index: int) -> Optional[str]:
        """Extract institution name"""
        # Look in current line and nearby lines
        contexts = [line]
        if index + 1 < len(lines):
            contexts.append(lines[index + 1])
        
        for context in contexts:
            # Look for university/college patterns
            institution_patterns = [
                r'\b([A-Z][A-Za-z\s]+(?:University|College|Institute|School))\b',
                r'\bUniversity of ([A-Za-z\s]+)',
            ]
            
            for pattern in institution_patterns:
                match = re.search(pattern, context)
                if match:
                    return match.group(0).strip()
        
        return None
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract graduation year"""
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            return int(year_match.group(0))
        return None
    
    def _extract_field_of_study(self, text: str) -> Optional[str]:
        """Extract field of study"""
        # Common field patterns
        field_patterns = [
            r'\bin\s+([A-Za-z\s]+(?:Science|Engineering|Studies|Arts))',
            r'(?:Bachelor|Master|PhD)\s+(?:of|in)\s+([A-Za-z\s]+)',
        ]
        
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                field = match.group(1).strip()
                if len(field) < 50:
                    return field
        
        return None
    
    def _extract_gpa(self, text: str) -> Optional[float]:
        """Extract GPA from text"""
        gpa_patterns = [
            r'GPA:?\s*(\d+\.?\d*)',
            r'(\d\.\d{1,2})\s*/\s*4\.0',
            r'Grade:?\s*(\d+\.?\d*)(?:/100)?'
        ]
        
        for pattern in gpa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    gpa = float(match.group(1))
                    if 0 <= gpa <= 10:  # Reasonable GPA range
                        return gpa
                except ValueError:
                    continue
        
        return None


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

class VisualizationEngine:
    """Handle all visualization components"""
    
    @staticmethod
    def create_fit_gauge(score: float) -> Optional[object]:
        """Create fit score gauge visualization"""
        if plt is None:
            return None
        
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(0, 1.2)
        ax.axis('off')
        
        # Create colored segments
        segments = [
            (0, 40, '#d9534f'),    # Red (Poor)
            (40, 70, '#f0ad4e'),   # Yellow (Average) 
            (70, 100, '#5cb85c')   # Green (Good)
        ]
        
        for start, end, color in segments:
            theta1 = 180 - (start / 100) * 180
            theta2 = 180 - (end / 100) * 180
            wedge = Wedge((0, 0), 1, theta2, theta1, width=0.3, 
                         facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(wedge)
        
        # Add needle
        angle = 180 - (score / 100) * 180
        needle_x = 0.8 * math.cos(math.radians(angle))
        needle_y = 0.8 * math.sin(math.radians(angle))
        
        ax.plot([0, needle_x], [0, needle_y], color='#333', linewidth=3)
        ax.add_patch(Circle((0, 0), 0.05, color='#333'))
        
        # Add score text
        ax.text(0, -0.2, f'{score:.1f}%', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        
        plt.title('Candidate Fit Score', fontsize=12, pad=20)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def create_risk_radar(flags: List[FraudFlag]) -> Optional[object]:
        """Create fraud risk radar chart"""
        if plt is None:
            return None
        
        categories = ['Timeline', 'Skills', 'Experience', 'Education', 'Verification', 'Authenticity']
        
        # Calculate risk scores by category
        risk_scores = {cat: 0 for cat in categories}
        
        for flag in flags:
            category = flag.category
            if category in risk_scores:
                weight = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(flag.severity, 0)
                risk_scores[category] += weight
        
        # Normalize scores
        max_score = max(risk_scores.values()) if risk_scores.values() else 1
        normalized_scores = [risk_scores[cat] / max_score * 5 for cat in categories]
        
        # Create radar chart
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, polar=True)
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        scores = normalized_scores + [normalized_scores[0]]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles, scores, 'o-', linewidth=2, color='#e74c3c')
        ax.fill(angles, scores, alpha=0.25, color='#e74c3c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.grid(True)
        
        plt.title('Fraud Risk by Category', y=1.08, fontsize=12)
        plt.tight_layout()
        
        return fig


# =============================================================================
# REPORT GENERATION
# =============================================================================


    # =========================================================================
    # BACKGROUND ANALYZERS (added to fix AttributeError)
    # =========================================================================
    def _analyze_required_background(self, jd_text: str) -> dict:
        """Very lightweight parser that extracts degree level and fields
        from a job description. It intentionally errs on the side of
        being permissive so it won't block the pipeline if a JD is vague.
        Returns a dict like {"degree_level": int|None, "fields": [str]}.
        Levels: 0=unspecified, 1=Bachelor, 2=Master/MBA, 3=PhD/Doctorate.
        """
        if not jd_text:
            return {}
        t = jd_text.lower()
        level = 0
        # Degree level heuristics
        if re.search(r'ph\.?d|doctorate|dphil', t):
            level = 3
        elif re.search(r"master'?s|m\.a\.|m\.s\.|mba|mtech|m\.eng", t):
            level = 2
        elif re.search(r"bachelor'?s|b\.a\.|b\.s\.|btech|b\.eng|be\b|bs\b|ba\b", t):
            level = 1
        
        # Extract common fields/majors
        fields_vocab = [
            'computer science','information technology','software engineering','electrical engineering',
            'electronics','mechanical engineering','civil engineering','data science','statistics',
            'mathematics','physics','chemistry','biology','economics','finance','accounting','business',
            'marketing','operations','human resources','psychology','design','architecture','law'
        ]
        requested_fields = []
        for f in fields_vocab:
            if re.search(rf"\b{re.escape(f)}\b", t):
                requested_fields.append(f)
        # Also capture patterns like "degree in X"
        extra = re.findall(r'(?:degree|major|background)\s+in\s+([a-zA-Z\s]{3,40})', t)
        for e in extra:
            e = e.strip().lower()
            if e and e not in requested_fields:
                requested_fields.append(e)
        
        return {"degree_level": level if level else None, "fields": requested_fields}
    
    def _analyze_candidate_background(self, profile) -> dict:
        """Summarizes the candidate's education into comparable pieces."""
        if not getattr(profile, 'education', None):
            return {}
        # Map degree strings to level
        def degree_to_level(deg: str) -> int:
            if not deg:
                return 0
            d = deg.lower()
            if re.search(r'ph\.?d|doctorate|dphil', d):
                return 3
            if re.search(r"master|m\.a\.|m\.s\.|mba|mtech|m\.eng", d):
                return 2
            if re.search(r"bachelor|b\.a\.|b\.s\.|btech|b\.eng|be\b|bs\b|ba\b", d):
                return 1
            return 0
        
        highest = 0
        fields = []
        for edu in profile.education:
            highest = max(highest, degree_to_level(getattr(edu, 'degree', None)))
            field = getattr(edu, 'field', None)
            if field:
                f = str(field).strip().lower()
                if f and f not in fields and len(f) < 50:
                    fields.append(f)
        return {"degree_level": highest if highest else None, "fields": fields}
    
    def _backgrounds_compatible(self, required: dict, candidate: dict) -> bool:
        """Checks if candidate education satisfies the JD requirements.
        - Degree level: candidate must have >= required level (if specified).
        - Fields: at least one overlap if JD lists fields; if JD specifies none,
          we consider it compatible.
        """
        if not required:
            return True
        # Degree level check
        req_level = required.get("degree_level") or 0
        cand_level = candidate.get("degree_level") or 0
        if req_level and cand_level < req_level:
            return False
        # Field overlap
        req_fields = set((required.get("fields") or []))
        cand_fields = set((candidate.get("fields") or []))
        if req_fields:
            # Normalize by stripping plurals and whitespace crudely
            def norm_set(s):
                out = set()
                for x in s:
                    x = re.sub(r'\s+', ' ', x.lower()).strip()
                    x = re.sub(r's$', '', x)  # crude singularization
                    out.add(x)
                return out
            return bool(norm_set(req_fields) & norm_set(cand_fields))
        return True
class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    @staticmethod
    def generate_json_report(result: AnalysisResult) -> str:
        """Generate JSON report"""
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'candidate_profile': asdict(result.profile),
            'fraud_flags': [asdict(flag) for flag in result.flags],
            'fit_score': asdict(result.fit_score) if result.fit_score else None,
            'verification_score': result.verification_score,
            'risk_level': result.risk_level,
            'verdict': result.verdict,
            'summary': result.summary,
            'recommendations': result.recommendations,
            'ai_insights': result.ai_insights
        }
        
        return json.dumps(report_data, indent=2, default=serialize_datetime)
    
    @staticmethod
    def generate_html_report(result: AnalysisResult) -> str:
        """Generate HTML report"""
        def escape_html(text):
            return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        profile = result.profile
        
        # Generate flags HTML
        flags_html = ""
        if result.flags:
            for flag in result.flags:
                severity_class = f"alert-{flag.severity.lower()}"
                flags_html += f"""
                <div class="alert {severity_class}">
                    <h4>{escape_html(flag.flag_type)} <span class="badge">{flag.severity}</span></h4>
                    <p>{escape_html(flag.description)}</p>
                    <small><strong>Evidence:</strong> {escape_html(flag.evidence)}</small>
                    <br><small><strong>Recommendation:</strong> {escape_html(flag.recommendation)}</small>
                </div>
                """
        else:
            flags_html = '<div class="alert alert-success">No fraud flags detected</div>'
        
        # Generate recommendations HTML
        recommendations_html = ""
        for rec in result.recommendations:
            recommendations_html += f"<li>{escape_html(rec)}</li>"
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fraud Detection Report - {escape_html(profile.name or 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .alert {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .alert-high {{ background: #f8d7da; border-left: 5px solid #dc3545; }}
                .alert-medium {{ background: #fff3cd; border-left: 5px solid #ffc107; }}
                .alert-low {{ background: #d1ecf1; border-left: 5px solid #17a2b8; }}
                .alert-success {{ background: #d4edda; border-left: 5px solid #28a745; }}
                .badge {{ background: #6c757d; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }}
                .score-box {{ background: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }}
                .verdict {{ font-size: 18px; font-weight: bold; padding: 15px; border-radius: 5px; text-align: center; }}
                .verdict-low {{ background: #d4edda; color: #155724; }}
                .verdict-medium {{ background: #fff3cd; color: #856404; }}
                .verdict-high {{ background: #f8d7da; color: #721c24; }}
                .verdict-critical {{ background: #f5c6cb; color: #491217; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .experience-item {{ border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fraud Detection Report</h1>
                <div class="grid">
                    <div>
                        <p><strong>Candidate:</strong> {escape_html(profile.name or 'Not specified')}</p>
                        <p><strong>Email:</strong> {escape_html(profile.email or 'Not specified')}</p>
                        <p><strong>Phone:</strong> {escape_html(profile.phone or 'Not specified')}</p>
                        <p><strong>Location:</strong> {escape_html(', '.join(profile.locations) if profile.locations else 'Not specified')}</p>
                    </div>
                    <div>
                        <p><strong>Total Experience:</strong> {profile.total_experience_years} years</p>
                        <p><strong>Career Level:</strong> {profile.career_level}</p>
                        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>Verification Score:</strong> {result.verification_score:.1f}/100</p>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="verdict verdict-{result.risk_level.lower()}">
                    {escape_html(result.verdict)}
                </div>
            </div>

            <div class="section">
                <h2>Fit Score Analysis</h2>
                {f'''
                <div class="grid">
                    <div class="score-box">
                        <h3>Overall Fit</h3>
                        <h2>{result.fit_score.overall_score}%</h2>
                    </div>
                    <div>
                        <table>
                            <tr><th>Component</th><th>Score</th></tr>
                            <tr><td>Skill Match</td><td>{result.fit_score.skill_match_score}%</td></tr>
                            <tr><td>Experience Match</td><td>{result.fit_score.experience_match_score}%</td></tr>
                            <tr><td>Education Match</td><td>{result.fit_score.education_match_score}%</td></tr>
                            <tr><td>Keyword Density</td><td>{result.fit_score.keyword_density_score}%</td></tr>
                            {f"<tr><td>Semantic Similarity</td><td>{result.fit_score.semantic_similarity_score}%</td></tr>" if result.fit_score.semantic_similarity_score else ""}
                        </table>
                    </div>
                </div>
                ''' if result.fit_score else '<p>Fit score not calculated (no job description provided)</p>'}
            </div>

            <div class="section">
                <h2>Fraud Risk Flags</h2>
                {flags_html}
            </div>

            <div class="section">
                <h2>Summary</h2>
                <p>{escape_html(result.summary)}</p>
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {recommendations_html}
                </ul>
            </div>

            <div class="section">
                <h2>Experience History</h2>
                <div>
                    {chr(10).join([f'<div class="experience-item"><strong>{escape_html(exp.role or "Role not specified")}</strong> at {escape_html(exp.company or "Company not specified")}<br><small>{exp.start_date.strftime("%Y-%m") if exp.start_date else "Start date unknown"} - {exp.end_date.strftime("%Y-%m") if exp.end_date and not exp.is_current else "Present" if exp.is_current else "End date unknown"} ({exp.duration_months or 0} months)</small><br>{escape_html(exp.text[:200])}...</div>' for exp in profile.experience]) if profile.experience else '<p>No experience entries found</p>'}
                </div>
            </div>

            <div class="section">
                <h2>Education History</h2>
                <div>
                    {chr(10).join([f'<div class="experience-item"><strong>{escape_html(edu.degree or "Degree not specified")}</strong><br>{escape_html(edu.institution or "Institution not specified")} {f"({edu.graduation_year})" if edu.graduation_year else ""}<br><small>{escape_html(edu.text[:150])}...</small></div>' for edu in profile.education]) if profile.education else '<p>No education entries found</p>'}
                </div>
            </div>

            <div class="section">
                <h2>Skills</h2>
                <p>{escape_html(', '.join(profile.skills)) if profile.skills else 'No skills extracted'}</p>
            </div>

            {f'''
            <div class="section">
                <h2>AI Insights</h2>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <pre style="white-space: pre-wrap;">{escape_html(result.ai_insights)}</pre>
                </div>
            </div>
            ''' if result.ai_insights else ''}

            <div class="section">
                <h2>Technical Details</h2>
                <p><strong>Risk Level:</strong> {result.risk_level}</p>
                <p><strong>Total Flags:</strong> {len(result.flags)}</p>
                <p><strong>High Severity:</strong> {len([f for f in result.flags if f.severity == 'HIGH'])}</p>
                <p><strong>Medium Severity:</strong> {len([f for f in result.flags if f.severity == 'MEDIUM'])}</p>
                <p><strong>Low Severity:</strong> {len([f for f in result.flags if f.severity == 'LOW'])}</p>
            </div>
        </body>
        </html>
        """
        
        return html


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    """Main Streamlit application"""
    if st is None:
        print("Streamlit not available. Install with: pip install streamlit")
        print("Run with: streamlit run fraud_detector.py")
        return
    
    st.set_page_config(
        page_title="Enhanced Fraud Detection System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .flag-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    .flag-high {
        background: #ffeaea;
        border-color: #dc3545;
    }
    .flag-medium {
        background: #fff8e1;
        border-color: #ffc107;
    }
    .flag-low {
        background: #e3f2fd;
        border-color: #2196f3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Enhanced Fraud Detection System</h1>
        <p>AI-Powered Candidate Verification & Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Keys
        gemini_key = st.text_input(
            "Gemini API Key (Optional)", 
            type="password",
            help="Enable AI insights with Google Gemini"
        )
        
        enable_ai_insights = st.checkbox(
            "Enable AI Insights", 
            value=bool(gemini_key),
            disabled=not bool(gemini_key)
        )
        
        st.divider()
        
        # Additional settings
        st.subheader("Analysis Settings")
        include_visualization = st.checkbox("Include Visualizations", value=True)
        detailed_analysis = st.checkbox("Detailed Analysis Mode", value=True)
        
        st.divider()
        
        # LinkedIn Profile Input
        st.subheader("Verification Data")
        linkedin_text = st.text_area(
            "LinkedIn Profile Text (Optional)",
            height=100,
            help="Paste LinkedIn profile content for cross-verification"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Document Upload")
        
        # File uploads
        resume_file = st.file_uploader(
            "Upload Resume",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        linkedin_file = st.file_uploader(
            "Upload LinkedIn Profile (Optional)",
            type=['pdf'],
            help="PDF export of LinkedIn profile"
        )
        
    with col2:
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste Job Description (Optional)",
            height=200,
            help="Job description for fit scoring and plagiarism detection"
        )
    
    # Analysis button
    if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
        
        if not resume_file:
            st.error("Please upload a resume to analyze.")
            st.stop()
        
        # Initialize detector
        detector = EnhancedFraudDetector(gemini_api_key=gemini_key if enable_ai_insights else None)
        
        try:
            with st.spinner("Processing documents..."):
                # Extract text from resume
                resume_bytes = resume_file.read()
                resume_text = detector.extract_text_from_file(resume_bytes, resume_file.name)
                
                # Extract LinkedIn text if provided
                complete_linkedin_text = linkedin_text
                if linkedin_file:
                    linkedin_bytes = linkedin_file.read()
                    linkedin_extracted = detector.extract_text_from_file(linkedin_bytes, linkedin_file.name)
                    complete_linkedin_text = f"{linkedin_text}\n{linkedin_extracted}" if linkedin_text else linkedin_extracted
                
            with st.spinner("Running fraud detection analysis..."):
                # Run complete analysis
                result = detector.analyze_candidate(
                    resume_text=resume_text,
                    jd_text=job_description.strip(),
                    linkedin_text=complete_linkedin_text.strip() if complete_linkedin_text else ""
                )
            
            # Display results
            display_analysis_results(result, include_visualization, detailed_analysis)
            
            # Generate reports
            col1, col2, col3 = st.columns(3)
            
            with col1:
                json_report = ReportGenerator.generate_json_report(result)
                st.download_button(
                    "📥 Download JSON Report",
                    data=json_report,
                    file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                html_report = ReportGenerator.generate_html_report(result)
                st.download_button(
                    "📥 Download HTML Report",
                    data=html_report,
                    file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
            
            with col3:
                # Summary stats
                st.metric("Risk Level", result.risk_level)
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)


def display_analysis_results(result: AnalysisResult, include_viz: bool = True, detailed: bool = True):
    """Display comprehensive analysis results"""
    
    # Main verdict
    risk_colors = {
        'LOW': 'success',
        'MEDIUM': 'warning', 
        'HIGH': 'error',
        'CRITICAL': 'error'
    }
    
    risk_color = risk_colors.get(result.risk_level, 'info')
    
    if risk_color == 'success':
        st.success(f"✅ {result.verdict}")
    elif risk_color == 'warning':
        st.warning(f"⚠️ {result.verdict}")
    else:
        st.error(f"🚨 {result.verdict}")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Verification Score",
            f"{result.verification_score:.1f}/100",
            delta=None
        )
    
    with col2:
        st.metric(
            "Experience Years",
            f"{result.profile.total_experience_years}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Career Level",
            result.profile.career_level,
            delta=None
        )
    
    with col4:
        st.metric(
            "Total Flags",
            len(result.flags),
            delta=None
        )
    
    # Fit Score Section
    if result.fit_score:
        st.subheader("📊 Fit Score Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if include_viz:
                viz_engine = VisualizationEngine()
                fig = viz_engine.create_fit_gauge(result.fit_score.overall_score)
                if fig:
                    st.pyplot(fig)
                else:
                    st.metric("Overall Fit Score", f"{result.fit_score.overall_score}%")
        
        with col2:
            st.write("**Component Breakdown:**")
            
            components = [
                ("Skill Match", result.fit_score.skill_match_score),
                ("Experience Match", result.fit_score.experience_match_score),
                ("Education Match", result.fit_score.education_match_score),
                ("Keyword Density", result.fit_score.keyword_density_score),
            ]
            
            if result.fit_score.semantic_similarity_score:
                components.append(("Semantic Similarity", result.fit_score.semantic_similarity_score))
            
            for name, score in components:
                st.progress(score / 100, text=f"{name}: {score}%")
    
    # Fraud Flags Section
    st.subheader("🚩 Fraud Risk Analysis")
    
    if result.flags:
        # Risk radar chart
        if include_viz:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                viz_engine = VisualizationEngine()
                radar_fig = viz_engine.create_risk_radar(result.flags)
                if radar_fig:
                    st.pyplot(radar_fig)
            
            with col2:
                st.write("**Flag Summary:**")
                flag_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
                for flag in result.flags:
                    flag_counts[flag.severity] = flag_counts.get(flag.severity, 0) + 1
                
                for severity, count in flag_counts.items():
                    if count > 0:
                        color = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🔵'}[severity]
                        st.write(f"{color} {severity}: {count} flag{'s' if count > 1 else ''}")
        
        # Detailed flags
        st.write("**Detailed Flags:**")
        
        # Sort flags by severity
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        sorted_flags = sorted(result.flags, key=lambda x: severity_order.get(x.severity, 3))
        
        for flag in sorted_flags:
            with st.expander(f"{flag.flag_type} ({flag.severity})", expanded=(flag.severity == 'HIGH')):
                st.write(f"**Description:** {flag.description}")
                st.write(f"**Evidence:** {flag.evidence}")
                st.write(f"**Confidence:** {flag.confidence_score:.0%}")
                if flag.recommendation:
                    st.write(f"**Recommendation:** {flag.recommendation}")
    else:
        st.success("No fraud flags detected. Candidate appears authentic.")
    
    # Recommendations Section
    st.subheader("💡 Recommendations")
    for i, rec in enumerate(result.recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # AI Insights Section
    if result.ai_insights:
        st.subheader("🤖 AI Insights")
        st.info(result.ai_insights)
    
    # Detailed Profile Section (if enabled)
    if detailed:
        with st.expander("👤 Detailed Profile Analysis"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Personal Information:**")
                st.write(f"Name: {result.profile.name or 'Not specified'}")
                st.write(f"Email: {result.profile.email or 'Not specified'}")
                st.write(f"Phone: {result.profile.phone or 'Not specified'}")
                st.write(f"Locations: {', '.join(result.profile.locations) if result.profile.locations else 'Not specified'}")
                
                st.write("**Skills:**")
                if result.profile.skills:
                    skills_text = ", ".join(result.profile.skills[:20])
                    if len(result.profile.skills) > 20:
                        skills_text += f"... and {len(result.profile.skills) - 20} more"
                    st.write(skills_text)
                else:
                    st.write("No skills extracted")
            
            with col2:
                st.write("**Experience Summary:**")
                st.write(f"Total Experience: {result.profile.total_experience_years} years")
                st.write(f"Number of Positions: {len(result.profile.experience)}")
                st.write(f"Career Level: {result.profile.career_level}")
                
                if result.profile.certifications:
                    st.write("**Certifications:**")
                    for cert in result.profile.certifications[:5]:
                        st.write(f"• {cert}")
            
            # Experience timeline
            if result.profile.experience:
                st.write("**Experience Timeline:**")
                for exp in result.profile.experience[:5]:
                    with st.container():
                        st.write(f"**{exp.role or 'Role not specified'}** at {exp.company or 'Company not specified'}")
                        if exp.start_date and exp.end_date:
                            duration_text = f"{exp.start_date.strftime('%Y-%m')} - "
                            duration_text += "Present" if exp.is_current else exp.end_date.strftime('%Y-%m')
                            if exp.duration_months:
                                duration_text += f" ({exp.duration_months} months)"
                            st.caption(duration_text)
                        st.caption(exp.text[:200] + "..." if len(exp.text) > 200 else exp.text)
                        st.divider()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def run_cli():
    """Command line interface for the fraud detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Fraud Detection System")
    parser.add_argument("--resume", required=True, help="Path to resume file")
    parser.add_argument("--jd", help="Path to job description file")
    parser.add_argument("--linkedin", help="Path to LinkedIn profile file")
    parser.add_argument("--gemini-key", help="Gemini API key for AI insights")
    parser.add_argument("--output", help="Output file path for report")
    parser.add_argument("--format", choices=["json", "html"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = EnhancedFraudDetector(gemini_api_key=args.gemini_key)
    
    # Load resume
    with open(args.resume, 'rb') as f:
        resume_text = detector.extract_text_from_file(f.read(), args.resume)
    
    # Load job description
    jd_text = ""
    if args.jd:
        with open(args.jd, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    
    # Load LinkedIn profile
    linkedin_text = ""
    if args.linkedin:
        with open(args.linkedin, 'rb') as f:
            linkedin_text = detector.extract_text_from_file(f.read(), args.linkedin)
    
    # Run analysis
    result = detector.analyze_candidate(resume_text, jd_text, linkedin_text)
    
    # Generate report
    if args.format == "json":
        report = ReportGenerator.generate_json_report(result)
    else:
        report = ReportGenerator.generate_html_report(result)
    
    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        run_cli()
    else:
        # Streamlit mode
        main()