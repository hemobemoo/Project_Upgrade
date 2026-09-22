from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

LABEL_FULL = {
    "NORM": "Normal Sinus Rhythm",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T-Wave Change",
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
}

LABEL_FULL_AR = {
    "NORM": "نظم جيبي طبيعي",
    "MI":   "احتشاء عضلة القلب",
    "STTC": "تغيرات الموجة ST-T",
    "CD":   "اضطراب التوصيل القلبي",
    "HYP":  "تضخم عضلة القلب",
}

ARABIC_SYSTEM_PROMPT = """
الٞبيايلثبنة.
تشمل مسؤولياتك ما يلي:
1. تفسير التشخيصات المتوقعة لتخطيط القلب.
2. إعداد تقارير طبية احترافية.
3. شرح الحالات القلبية بوضوح.
4. تقديم رؤى ذات صلة سريرية.

يجب أن يتضمن التقرير:
- نظرة عامة على الحالة
- تفسير تخطيط القلب
- الأعراض المحتملة
- الأهمية السريرية
- التوصية بالمتابعة الطبية

يجب أن يكون الشرح دقيقا طبيا وسهل الفهم، وأن تكتب كامل استجابتك باللغة
العربية الفصحى، مع الحفاظ على المصطلحات الطبية الدقيقة، حتى لو كان النموذج
قد تم ضبطه (fine-tuned) على تقارير باللغة الإنجليزية.

"""

ARABIC_USER_PROMPT_TEMPLATE = """
### سياق المريض:
- العمر: {age}
- الجنس: {sex}
- معدل ضربات القلب: {hr} نبضة/دقيقة
- نظم القلب: {heart_rhythm}
- انتظام النظم: {rhythm_regularity}

### الحالات المتوقعة:
- {diagnosis} (درجة ثقة النموذج: {confidence})

### تعليمات لإعداد التقرير الطبي:
بالاعتماد على السياق السريري والحالات المتوقعة أعلاه، أعد تقريرًا طبيًا احترافيًا.
- ادمج معدل ضربات القلب ونظمه في تفسيرك للتشخيصات.
- إذا كانت هناك أكثر من حالة متوقعة، اشرح العلاقة السريرية المحتملة بينها.
- ركّز على التحليل الطبي بدلاً من درجات الثقة الرقمية.
- نظّم التقرير وفق الهيكل التالي: 1. الملخص التشخيصي  2. الارتباط السريري
  3. الأعراض المحتملة  4. التوصيات.
- اكتب بأسلوب طبي احترافي وموجز، باللغة العربية الفصحى فقط.
"""

def create_system_prompt(language: str = "en") -> str:
    if language == "ar":
        return ARABIC_SYSTEM_PROMPT.strip()
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

def create_user_prompt(
    predictions: list[tuple[str, float, bool]],
    meta: dict = None,
    language: str = "en",
) -> str:
    active = [(name, prob) for name, prob, is_pos in predictions if is_pos]
    if not active:
        top = max(predictions, key=lambda x: x[1])
        active = [(top[0], top[1])]

    if language == "ar":
        label_dict = LABEL_FULL_AR
    else:
        label_dict = LABEL_FULL
    named = [(label_dict.get(name, name), round(prob, 2)) for name, prob in active]

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
    language: str = "en",
) -> str:
    if not api_key or not model_id:
        raise EnvironmentError("OpenAI API key or model ID not configured.")

    client = OpenAI(api_key=api_key)

    stream = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": create_system_prompt(language=language)},
            {"role": "user",   "content": create_user_prompt(predictions, meta, language=language)},
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