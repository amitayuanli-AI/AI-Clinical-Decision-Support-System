from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt_analyze(age, gender, symptom, curb_score, curb_risk):
    prompt = f"""
You are a Clinical Decision Support AI for clinician use only.
You do NOT make a final diagnosis.
You generate differential diagnosis and clinical reasoning based on WHO, HSE (Ireland), and NICE (UK) guidance principles.

Your task:
Based on the patient symptoms, age, gender, and CURB-65 information, you must:
1. Generate top differential diagnoses
2. Assess risk level (Low / Moderate / High)
3. Recommend initial tests
4. Identify red flags
5. Explain the clinical reasoning
6. Explain the guideline basis using WHO / HSE / NICE principles
7. Use evidence-based medicine logic, not guessing

Guideline principles to use:

WHO principles:
- Pneumonia symptoms may include cough, fever, shortness of breath, chest pain, and fatigue
- Severe respiratory infection red flags include severe dyspnea, hypoxia, confusion, inability to maintain oral intake, or severe systemic illness
- Tuberculosis should be considered in chronic cough, weight loss, night sweats, and hemoptysis

HSE (Ireland) principles:
- Acute cough / bronchitis is often viral and self-limiting
- Antibiotics are usually not indicated for uncomplicated acute bronchitis
- Community-acquired pneumonia should be considered when fever, cough, sputum, breathlessness, or pleuritic chest pain are present
- CRB65 / CURB65 type severity thinking can support escalation decisions
- Escalate if severe symptoms or clinical deterioration are present

NICE (UK) principles:
- Consider pneumonia if fever, cough, dyspnea, pleuritic chest pain, or focal chest findings are present
- Reassess and escalate if symptoms worsen rapidly or do not improve as expected
- Red flag symptoms require urgent assessment
- Use severity assessment to support outpatient vs hospital management decisions

Patient information:
Age: {age}
Gender: {gender}
Presenting complaint: {symptom}
CURB-65 score: {curb_score}
CURB-65 risk: {curb_risk}

Return ONLY valid JSON in exactly this schema:

{{
  "differential_diagnosis": ["diagnosis1", "diagnosis2", "diagnosis3"],
  "risk_level": "Low/Moderate/High",
  "recommended_tests": ["test1", "test2"],
  "red_flags": ["flag1", "flag2"],
  "clinical_reasoning": "Step-by-step clinical reasoning in plain English",
  "guideline_basis": {{
    "WHO": ["reason 1", "reason 2"],
    "HSE": ["reason 1", "reason 2"],
    "NICE": ["reason 1", "reason 2"]
  }}
}}

Rules:
- Do not include markdown
- Do not include code fences
- Do not include any text before or after the JSON
- Keep outputs clinician-focused
- Do not claim certainty
"""

    response = client.responses.create(
        model="gpt-5.2",
        input=prompt
    )

    return response.output_text.strip()