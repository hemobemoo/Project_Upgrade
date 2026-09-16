from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

LABEL_FULL = {
    "NORM": "Normal Sinus Rhythm",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T-Wave Change",
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
}

def create_system_prompt() -> str:
    return """
You are an expert cardiology AI assistant specialized in ECG interpretation.
Your responsibilities include:
1. Interpreting predicted ECG diagnoses.
2. Generating professional medical reports.
3. Explaining cardiac conditions clearly.
4. Providing clinically relevant insights.

The report must include:
- Condition overview
- ECG interpretation
- Possible symptoms
- Clinical significance
- Suggested medical follow-up

The explanation must be medically accurate and easy to understand.
"""

def create_user_prompt(predictions: list[tuple[str, float, bool]], meta: dict = None) -> str:
    active = [(name, prob) for name, prob, is_pos in predictions if is_pos]
    if not active:
        top = max(predictions, key=lambda x: x[1])
        active = [(top[0], top[1])]

    named = [(LABEL_FULL.get(name, name), round(prob, 2)) for name, prob in active]

    patient_info = ""
    if meta:
        patient_info = f"""
### Patient Context:
- Age: {meta.get('age', 'N/A')}
- Sex: {meta.get('sex', 'N/A')}
- Heart Rate: {meta.get('hr', 'N/A')} bpm
- Heart Rhythm: {meta.get('heart_rhythm', 'N/A')}
- Rhythm Regularity: {meta.get('rhythm_regularity', 'N/A')}
"""

    prompt = f"""
ECG Analysis Data:
{patient_info}

### Predicted Conditions:
"""
    for diagnosis, confidence in named:
        prompt += f"- {diagnosis} (Model Confidence Score: {confidence})\n"

    prompt += """
### Instructions for the Expert Report:
Using the clinical context and predicted conditions above, generate a professional medical report. 
- Integrate the patient's heart rate and rhythm findings into your interpretation of the diagnoses.
- If multiple conditions are predicted, explain their potential clinical relationship.
- Focus on medical reasoning rather than the numerical confidence scores.
- Structure the report with: 1. Diagnostic Summary, 2. Clinical Correlation, 3. Possible Symptoms, 4. Recommendations.
- Write in a professional, concise medical style.
"""
    return prompt

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def generate_ecg_report(
    predictions: list[tuple[str, float, bool]],
    api_key: str,
    model_id: str,
    meta: dict = None,  
    temperature: float = 0.3,
) -> str:
    if not api_key or not model_id:
        raise EnvironmentError("OpenAI API key or model ID not configured.")

    client = OpenAI(api_key=api_key)

    stream = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": create_system_prompt()},
            {"role": "user",   "content": create_user_prompt(predictions, meta)},
        ],
        stream=True,
        temperature=temperature,
        max_tokens=900,
    )

    report = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            report += delta

    report = report.replace("```markdown", "").replace("```", "").strip()
    return report