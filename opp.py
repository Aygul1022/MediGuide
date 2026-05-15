import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
MIN_QUERY_WORDS = 3
MIN_DETAIL_TOKENS_FOR_RETRIEVAL = 5
STOPWORDS = {
    "a", "about", "am", "an", "and", "are", "for", "have", "has", "i",
    "in", "is", "it", "me", "my", "of", "on", "patient", "the", "to",
    "with"
}
GENERIC_SYMPTOMS = {
    "ache", "aches", "cough", "coughing", "diarrhea", "dizzy", "dizziness",
    "fatigue", "fever", "headache", "hurt", "hurts", "nausea", "pain",
    "rash", "sick", "tired", "vomit", "vomiting", "weak", "weakness"
}
ABDOMINAL_TERMS = {
    "abdomen", "abdominal", "belly", "gastric", "stomach", "tummy"
}
LOWER_ABDOMINAL_TERMS = {
    "below", "lower", "pelvic", "rlq", "suprapubic"
}
FOLLOWUP_DETAIL_TERMS = {
    "ago", "day", "days", "hour", "hours", "week", "weeks", "started",
    "start", "since", "yesterday", "today", "morning", "night",
    "upper", "right", "left", "middle", "center", "centre", "part",
    "mild", "moderate", "severe", "worse", "better", "high", "low"
}
ASSOCIATED_SYMPTOM_TERMS = {
    "vomit", "vomiting", "diarrhea", "urinary", "urination", "burning",
    "nausea", "rash", "confusion", "stiff", "stiffness", "cough",
    "breathing", "blood"
}
TYPO_CORRECTIONS = {
    "achey": "achy",
    "bellyache": "belly pain",
    "its": "it is",
    "stomache": "stomach",
    "stomachache": "stomach pain",
    "stoamch": "stomach",
    "temprature": "temperature",
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="MediGuide AI | MediSync Health", page_icon="⚕️", layout="wide")

# --- DATABASE CONNECTION ---
@st.cache_resource
def load_database():
    if not os.path.isdir(VECTOR_DB_DIR):
        raise FileNotFoundError(
            f"Vector DB directory not found: {VECTOR_DB_DIR}.\nRun `python buid_vectordb.py` first to create the Chroma database."
        )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# --- LLM SETUP ---
@st.cache_resource
def load_llm():
    """Load a lightweight local LLM for conversational responses"""
    try:
        # Try using Ollama if available locally
        from langchain_community.llms import Ollama
        ollama_model = os.getenv("OLLAMA_MODEL", "neural-chat")
        llm = Ollama(model=ollama_model)
        llm.invoke("Respond with OK.")
        return llm
    except Exception:
        pass
    
    try:
        # Fallback: Use Azure OpenAI if configured
        from langchain_openai import AzureChatOpenAI
        if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
            return AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-08-01-preview",
                model="gpt-4"
            )
    except Exception:
        pass
    
    # If no LLM available, return None and use rule-based response
    return None

def normalize_query(query: str) -> str:
    """Normalize common keyboard/typing issues before safety checks and retrieval."""
    normalized = query.lower().replace("ı", "i")
    normalized = normalized.replace(",", " ")
    words = []
    for raw_word in normalized.split():
        prefix = raw_word[:len(raw_word) - len(raw_word.lstrip(".,;:!?()[]{}"))]
        suffix = raw_word[len(raw_word.rstrip(".,;:!?()[]{}")):]
        word = raw_word.strip(".,;:!?()[]{}")
        words.append(f"{prefix}{TYPO_CORRECTIONS.get(word, word)}{suffix}")
    return " ".join(words)


def tokenize_medical_query(query: str) -> list:
    words = [word.strip(".,;:!?()[]{}").lower() for word in normalize_query(query).split()]
    return [word for word in words if word and word not in STOPWORDS]


def is_query_too_broad(query: str) -> bool:
    """Detect generic symptom prompts that are unsafe to send directly to vector search."""
    words = tokenize_medical_query(query)
    if len(words) < MIN_QUERY_WORDS:
        return True
    if all(word in GENERIC_SYMPTOMS or word in ABDOMINAL_TERMS for word in words):
        return True
    return len(words) < MIN_DETAIL_TOKENS_FOR_RETRIEVAL and bool(set(words) & GENERIC_SYMPTOMS)


def broad_query_response(query: str) -> str:
    return f"""`{query}` is too broad for reliable clinical case matching.

Please add a few details so I can search more safely, for example:

- Age range and sex
- Duration and temperature range
- Associated symptoms such as cough, rash, headache, neck stiffness, abdominal pain, urinary symptoms, travel, exposures, or immune status
- Any abnormal vitals, labs, imaging, or exam findings

**Urgent warning signs:** seek immediate medical evaluation for fever with confusion, stiff neck, breathing difficulty, chest pain, persistent low blood pressure, seizure, severe dehydration, non-blanching rash, or fever in an infant or immunocompromised patient.

**Important Notice:**
This is decision support only. A physician must make the final diagnosis and treatment plan."""


def symptom_triage_response(query: str) -> str | None:
    """Return a safer template for common symptom clusters before retrieval."""
    words = set(tokenize_medical_query(query))
    has_fever = "fever" in words or "temperature" in words
    has_headache = "headache" in words
    has_cough = bool(words & {"cough", "coughing"})
    has_abdominal_area = bool(words & ABDOMINAL_TERMS)
    has_abdominal_pain = has_abdominal_area and bool(words & {"ache", "aches", "hurt", "hurts", "pain"})

    if has_fever and has_headache:
        return """Fever with headache is common, but it needs screening for warning signs before matching cases.

**Common categories clinicians consider include:**
- Viral respiratory illness or influenza-like illness
- Sinus, ear, or throat infection depending on local symptoms
- Migraine or dehydration occurring alongside an infection
- Meningitis or serious infection if red flags are present

**Useful details to add:**
How high is the fever, when did it start, is there neck stiffness, rash, confusion, vomiting, light sensitivity, severe sudden headache, cough/sore throat, recent travel, immune suppression, or abnormal vitals?

**Seek urgent care now** if there is stiff neck, confusion, fainting, seizure, purple/non-blanching rash, severe sudden headache, repeated vomiting, weakness/numbness, trouble breathing, or fever above 39.4 C / 103 F.

**Important Notice:**
This is decision support only. A physician must make the final diagnosis and treatment plan."""

    if has_fever and has_abdominal_pain:
        return fever_abdominal_pain_response(query)

    if has_fever and has_cough:
        return """Fever with cough should be triaged by severity and respiratory risk before matching cases.

**Common categories clinicians consider include:**
- Viral upper respiratory infection, influenza, COVID-like illness, or bronchitis
- Pneumonia, especially with shortness of breath, chest pain, low oxygen, or persistent high fever
- Asthma/COPD flare with infection in patients with chronic lung disease

**Useful details to add:**
Temperature, duration, cough type, shortness of breath, chest pain, oxygen level if available, age, pregnancy status, immune status, lung/heart disease, and any abnormal exam or imaging findings.

**Seek urgent care now** for trouble breathing, blue lips, chest pain, confusion, oxygen saturation below 92%, coughing blood, severe dehydration, or persistent high fever.

**Important Notice:**
This is decision support only. A physician must make the final diagnosis and treatment plan."""

    return None


def extract_duration(query: str) -> str | None:
    tokens = tokenize_medical_query(query)
    for index, token in enumerate(tokens):
        if token.isdigit() and index + 1 < len(tokens):
            unit = tokens[index + 1]
            if unit in {"hour", "hours", "day", "days", "week", "weeks"}:
                return f"{token} {unit}"
    if "yesterday" in tokens:
        return "since yesterday"
    if "today" in tokens:
        return "today"
    return None


def extract_temperature(query: str) -> str | None:
    tokens = tokenize_medical_query(query)
    for index, token in enumerate(tokens):
        if token.replace(".", "", 1).isdigit():
            value = float(token)
            next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
            if next_token in {"c", "celsius", "degree", "degrees"} or 34 <= value <= 43:
                display_value = int(value) if value.is_integer() else value
                return f"{display_value} C"
    return None


def extract_symptom_profile(query: str) -> dict:
    words = set(tokenize_medical_query(query))
    return {
        "has_fever": "fever" in words or "temperature" in words,
        "has_abdominal_pain": bool(words & ABDOMINAL_TERMS) and bool(words & {"ache", "aches", "hurt", "hurts", "pain"}),
        "is_lower": bool(words & LOWER_ABDOMINAL_TERMS),
        "duration": extract_duration(query),
        "temperature": extract_temperature(query),
        "associated": sorted(words & ASSOCIATED_SYMPTOM_TERMS),
    }


def format_symptom_summary(profile: dict) -> str:
    summary = []
    if profile["has_fever"]:
        fever_text = f"fever ({profile['temperature']})" if profile["temperature"] else "fever"
        summary.append(fever_text)
    if profile["has_abdominal_pain"]:
        location = "lower abdominal pain" if profile["is_lower"] else "abdominal/stomach pain"
        summary.append(location)
    if profile["duration"]:
        summary.append(f"duration: {profile['duration']}")
    if profile["associated"]:
        summary.append("associated symptoms mentioned: " + ", ".join(profile["associated"]))
    return "; ".join(summary) if summary else "not enough clinical detail yet"


def next_questions_for_profile(profile: dict) -> list:
    questions = []
    if profile["has_fever"] and not profile["temperature"]:
        questions.append("What is the temperature value, and does it come down with medication?")
    if profile["has_abdominal_pain"] and not profile["is_lower"]:
        questions.append("Where exactly is the pain: right lower, left lower, upper, or middle?")
    if not profile["duration"]:
        questions.append("When did it start?")
    if not profile["associated"]:
        questions.append("Any vomiting, diarrhea, burning urination, blood in stool/vomit, rash, stiff neck, confusion, or trouble breathing?")
    questions.append("Age and sex?")
    return questions[:3]


def general_symptom_response(query: str) -> str | None:
    """Avoid case retrieval for symptom-only prompts that do not include enough detail."""
    words = set(tokenize_medical_query(query))
    symptom_words = words & (GENERIC_SYMPTOMS | ABDOMINAL_TERMS)
    if not symptom_words:
        return None

    has_enough_context = len(words) >= MIN_DETAIL_TOKENS_FOR_RETRIEVAL and bool(
        words - GENERIC_SYMPTOMS - ABDOMINAL_TERMS
    )
    if has_enough_context:
        return None

    symptoms = ", ".join(sorted(symptom_words))
    return f"""I see these symptoms: **{symptoms}**.

This is not enough detail for reliable clinical case matching, so I will not use the case database yet.

**Please add:**
- Age and sex
- How long symptoms have been present
- Temperature value if fever is present
- Severity and location of pain, if any
- Associated symptoms such as cough, rash, vomiting, diarrhea, neck stiffness, urinary symptoms, confusion, or shortness of breath
- Any abnormal vitals, labs, imaging, pregnancy possibility, immune suppression, or major medical history

**Seek urgent care now** for confusion, stiff neck, trouble breathing, chest pain, seizure, fainting, severe or worsening pain, dehydration, non-blanching rash, or very high fever.

**Important Notice:**
This is decision support only. A physician must make the final diagnosis and treatment plan."""


def build_contextual_query(prompt: str) -> str:
    """Blend short follow-ups with recent user symptom context."""
    recent_user_messages = [
        message["content"]
        for message in st.session_state.messages[-6:]
        if message.get("role") == "user"
    ]
    if recent_user_messages and recent_user_messages[-1] == prompt:
        recent_user_messages = recent_user_messages[:-1]
    context = " ".join(recent_user_messages)
    normalized_prompt = normalize_query(prompt)
    normalized_context = normalize_query(context)
    prompt_tokens = set(tokenize_medical_query(prompt))
    context_tokens = set(tokenize_medical_query(context))

    has_prior_symptom_context = bool(context_tokens & (GENERIC_SYMPTOMS | ABDOMINAL_TERMS))
    has_location_followup = bool(prompt_tokens & LOWER_ABDOMINAL_TERMS)
    has_detail_followup = bool(prompt_tokens & FOLLOWUP_DETAIL_TERMS) or any(token.isdigit() for token in prompt_tokens)
    is_short_followup = len(prompt_tokens) <= 5

    if has_prior_symptom_context and (has_location_followup or has_detail_followup or is_short_followup):
        return f"{context} {prompt}"
    return prompt


def has_fever_and_abdominal_pain(query: str) -> bool:
    words = set(tokenize_medical_query(query))
    has_fever = "fever" in words or "temperature" in words
    has_abdominal_area = bool(words & ABDOMINAL_TERMS)
    has_pain = bool(words & {"ache", "aches", "hurt", "hurts", "pain"})
    return has_fever and has_abdominal_area and has_pain


def has_lower_abdominal_context(query: str) -> bool:
    words = set(tokenize_medical_query(query))
    return has_fever_and_abdominal_pain(query) and bool(words & LOWER_ABDOMINAL_TERMS)


def fever_abdominal_pain_response(query: str) -> str:
    profile = extract_symptom_profile(query)
    summary = format_symptom_summary(profile)
    questions = "\n".join([f"- {question}" for question in next_questions_for_profile(profile)])

    concern = "This combination needs clinical assessment rather than a database-only match."
    if profile["is_lower"]:
        concern = "Lower abdominal pain with fever raises concern for appendicitis, urinary/kidney infection, and pelvic causes depending on age/sex/pregnancy possibility."
    if profile["duration"]:
        concern += f" Since it has been going on for {profile['duration']}, worsening pain or persistent fever should be taken seriously."

    return f"""Got it. Current picture: **{summary}**.

{concern}

**Next details I need:**
{questions}

**Seek urgent care now** if the pain is severe or worsening, localized to the right lower abdomen, the abdomen is rigid, there is repeated vomiting, blood in stool/vomit, fainting, confusion, dehydration, pregnancy, immune suppression, or fever above 39.4 C / 103 F.

**Important Notice:**
This is decision support only. A physician must make the final diagnosis and treatment plan."""


def should_show_reference_cases(query: str) -> bool:
    """Show reference cases after enough clinical detail has accumulated."""
    profile = extract_symptom_profile(query)
    detail_count = sum([
        bool(profile["temperature"]),
        bool(profile["duration"]),
        bool(profile["is_lower"]),
        bool(profile["associated"]),
    ])
    return profile["has_fever"] and profile["has_abdominal_pain"] and detail_count >= 1


def build_reference_query(query: str) -> str:
    """Expand patient wording into clinical concepts for better reference retrieval."""
    profile = extract_symptom_profile(query)
    terms = ["fever", "abdominal pain"]

    if profile["is_lower"]:
        terms.extend([
            "lower abdominal pain",
            "right lower quadrant pain",
            "appendicitis",
            "urinary tract infection",
            "pyelonephritis",
            "pelvic inflammatory disease",
        ])
    else:
        terms.extend([
            "gastroenteritis",
            "intra-abdominal infection",
            "appendicitis",
            "urinary tract infection",
        ])

    if profile["associated"]:
        terms.extend(profile["associated"])
    if profile["duration"]:
        terms.append(f"symptoms for {profile['duration']}")

    return " ".join(terms)


def retrieve_relevant_cases(db, query: str, k: int = 3) -> list:
    """Use MMR for more varied results, then remove near-duplicate findings."""
    candidate_cases = db.max_marginal_relevance_search(normalize_query(query), k=8, fetch_k=24)
    unique_cases = []
    seen_findings = set()

    for case in candidate_cases:
        findings = parse_case_content(case.page_content)["findings"]
        fingerprint = " ".join(findings.lower().split())[:180]
        if fingerprint in seen_findings:
            continue
        seen_findings.add(fingerprint)
        unique_cases.append(case)
        if len(unique_cases) == k:
            break

    return unique_cases


def retrieve_reference_cases(db, query: str, k: int = 3) -> list:
    """Retrieve clinician references and rerank by clinical keyword overlap."""
    query_terms = set(tokenize_medical_query(query))
    candidate_cases = []
    for focused_query in [query, "appendicitis", "abdominal pain fever diarrhea", "urinary tract infection abdominal pain fever"]:
        candidate_cases.extend(db.similarity_search(normalize_query(focused_query), k=8))

    try:
        keyword_results = db.get(where_document={"$contains": "appendicitis"}, limit=4)
        for document, metadata in zip(keyword_results.get("documents", []), keyword_results.get("metadatas", [])):
            candidate_cases.append(type(candidate_cases[0])(page_content=document, metadata=metadata or {}))
    except Exception:
        pass

    scored_cases = []

    for case in candidate_cases:
        case_data = parse_case_content(case.page_content)
        case_terms = set(tokenize_medical_query(f"{case_data.get('symptoms') or ''} {case_data['findings']}"))
        score = len(query_terms & case_terms)
        findings_lower = case_data["findings"].lower()
        for diagnosis in ["appendicitis", "pyelonephritis", "urinary tract infection", "intra-abdominal infection"]:
            if diagnosis in normalize_query(query) and diagnosis in findings_lower:
                score += 5
        if "q fever" in findings_lower and "q fever" not in normalize_query(query):
            score -= 3
        if score >= 1:
            scored_cases.append((score, case))

    scored_cases.sort(key=lambda item: item[0], reverse=True)

    unique_cases = []
    seen_findings = set()
    for _, case in scored_cases:
        findings = parse_case_content(case.page_content)["findings"]
        fingerprint = " ".join(findings.lower().split())[:180]
        if fingerprint in seen_findings:
            continue
        seen_findings.add(fingerprint)
        unique_cases.append(case)
        if len(unique_cases) == k:
            break

    return unique_cases

def parse_case_content(content: str) -> dict:
    """Return a natural summary of the stored clinical case content."""
    summary = {"symptoms": None, "findings": content.strip()}

    if "Clinical Case / Symptoms:" in content and "Recommended Treatment / Response:" in content:
        try:
            symptoms_part = content.split("Clinical Case / Symptoms:", 1)[1].split("Recommended Treatment / Response:", 1)[0].strip()
            findings_part = content.split("Recommended Treatment / Response:", 1)[1].strip()
            summary = {"symptoms": symptoms_part, "findings": findings_part}
        except Exception:
            summary = {"symptoms": None, "findings": content.strip()}
    elif "Symptoms/Presentation:" in content and "Findings/Outcome:" in content:
        try:
            symptoms_part = content.split("Symptoms/Presentation:", 1)[1].split("Findings/Outcome:", 1)[0].strip()
            findings_part = content.split("Findings/Outcome:", 1)[1].strip()
            summary = {"symptoms": symptoms_part, "findings": findings_part}
        except Exception:
            summary = {"symptoms": None, "findings": content.strip()}
    elif "Question:" in content and "Answer:" in content:
        try:
            question_part = content.split("Question:", 1)[1].split("Answer:", 1)[0].strip()
            answer_part = content.split("Answer:", 1)[1].strip()
            summary = {"symptoms": question_part, "findings": answer_part}
        except Exception:
            summary = {"symptoms": None, "findings": content.strip()}
    elif "Answer:" in content:
        answer_part = content.split("Answer:", 1)[1].strip()
        summary = {"symptoms": None, "findings": answer_part}

    return summary


def build_basic_response(retrieved_cases: list) -> str:
    num_cases = len(retrieved_cases)
    response = f"""Based on your symptoms, I found **{num_cases} similar clinical cases** in our database.

**Relevant Clinical Findings:**
"""
    for i, case in enumerate(retrieved_cases[:3], 1):
        case_data = parse_case_content(case.page_content)
        findings = case_data["findings"].replace("\n", " ").strip()
        findings = findings.replace("Clinical case summary:", "").replace("Symptoms/Presentation:", "").replace("Clinical Case / Symptoms:", "").replace("Recommended Treatment / Response:", "").strip()
        response += f"\n**Case {i}:** {findings[:280].rstrip()}...\n"

    response += f"""

**Possible conditions to consider:**
Review the findings above for likely diagnoses and complications.

**Recommendation:**
These cases suggest you should consider further evaluation for the conditions mentioned above. Please consult with a physician for proper diagnosis and treatment.

**Important Notice:**
⚠️ This is decision support only. Treatment protocols must be evaluated by a physician according to the patient's specific condition."""
    return response


def generate_conversational_response(query: str, retrieved_cases: list, llm) -> str:
    """Generate a natural conversational response based on retrieved clinical cases"""
    
    if not retrieved_cases:
        return "I couldn't find any matching clinical cases in the database. Please try rephrasing your symptoms."
    
    # Format the retrieved cases as natural clinical summaries
    cases_text = "\n\n".join([
        f"**Case {i+1}:**\nFindings: {parse_case_content(case.page_content)['findings']}\n(Source: {case.metadata.get('source', 'Unknown')})"
        for i, case in enumerate(retrieved_cases[:3])
    ])
    
    if llm:
        # Use LLM for sophisticated response generation without requiring langchain.chains imports
        prompt = f"""You are MediGuide, a helpful medical assistant that provides decision support based on clinical knowledge.

A patient or healthcare provider has described these symptoms or findings:
{query}

Here are similar clinical cases from our database:
{cases_text}

Based on these cases, provide a helpful, conversational response that:
1. Summarizes the key clinical findings from the similar cases
2. Suggests what conditions or disease categories might be involved
3. Gives the user an idea of possible next evaluation steps
4. Reminds the user that this is decision support and a physician must make the final diagnosis

Respond naturally, without repeating the cases verbatim or presenting them as a question-and-answer dialog."""
        try:
            response = llm(prompt)
            if hasattr(response, 'text'):
                response = response.text
            return response
        except TypeError:
            try:
                generated = llm.generate([prompt])
                response = generated[0].text if hasattr(generated[0], 'text') else str(generated)
                return response
            except Exception:
                return build_basic_response(retrieved_cases) + "\n\n*Note: advanced local LLM unavailable; using basic summarization instead.*"
        except Exception:
            return build_basic_response(retrieved_cases) + "\n\n*Note: advanced local LLM unavailable; using basic summarization instead.*"
    else:
        return build_basic_response(retrieved_cases)

try:
    db = load_database()
    init_error = None
except Exception as exc:
    db = None
    init_error = str(exc)

# Try to load LLM
try:
    llm = load_llm()
except Exception:
    llm = None

# --- UI DESIGN ---
st.title("⚕️ MediGuide Clinical Assistant")
st.markdown("""
**MediSync Health Network - AI-Enabled Knowledge System**
This system scans thousands of internal clinical cases to provide *Decision Support* for healthcare professionals.
*Please use your own clinical wisdom for final decisions.*
""")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    if init_error:
        st.error("MediSync Knowledge Base: unavailable")
        st.warning(init_error)
        st.info("Run `python buid_vectordb.py` first to build the vector database.")
    else:
        st.success("MediSync Knowledge Base: Active")
        st.info("Thousands of clinical cases ready to be scanned.")
    
    st.divider()
    st.subheader("AI Response Mode")
    if llm:
        st.success("✅ Advanced LLM Mode: Generating conversational responses")
    else:
        st.info("ℹ️ Basic Mode: Using template responses (install Ollama for advanced mode)")

# Chat Interface
st.subheader("Consultation with MediGuide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am MediGuide, your clinical decision support assistant. Please describe the patient's symptoms, and I will find similar clinical cases and provide insights based on them."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if init_error:
    st.stop()

# User Query
if prompt := st.chat_input("E.g., Patient presented with high fever and severe cough..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Scanning MediSync database for similar cases..."):
            contextual_prompt = build_contextual_query(prompt)
            triage_response = symptom_triage_response(contextual_prompt)
            general_response = general_symptom_response(contextual_prompt)

            if triage_response:
                if should_show_reference_cases(contextual_prompt):
                    reference_query = build_reference_query(contextual_prompt)
                    retrieved_cases = retrieve_reference_cases(db, reference_query, k=3)
                else:
                    retrieved_cases = []
                response = triage_response
            elif general_response:
                retrieved_cases = []
                response = general_response
            elif is_query_too_broad(contextual_prompt):
                retrieved_cases = []
                response = broad_query_response(prompt)
            elif has_fever_and_abdominal_pain(contextual_prompt):
                retrieved_cases = []
                response = fever_abdominal_pain_response(contextual_prompt)
            else:
                # Retrieve varied cases so one repeated diagnosis does not dominate the answer.
                retrieved_cases = retrieve_relevant_cases(db, contextual_prompt, k=3)

                # Generate conversational response
                response = generate_conversational_response(contextual_prompt, retrieved_cases, llm)
            st.markdown(response)

            if retrieved_cases:
                with st.expander("Show theoretical reference cases for clinician review"):
                    for i, case in enumerate(retrieved_cases, 1):
                        case_data = parse_case_content(case.page_content)
                        st.markdown(f"**Case {i}:** {case_data['findings']}")
                        if case_data['symptoms']:
                            st.caption(f"Symptoms: {case_data['symptoms']} • Source: {case.metadata.get('source', 'Unknown')}")
                        else:
                            st.caption(f"Source: {case.metadata.get('source', 'Unknown')}")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
