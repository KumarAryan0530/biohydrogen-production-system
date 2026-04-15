import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"Warning: Failed to configure Gemini API: {e}")
        model = None
else:
    model = None


def is_ai_enabled():
    return model is not None


def generate_executive_summary(run_data):
    """Generates an executive summary based on simulation metrics."""
    if not is_ai_enabled():
        return "AI Insights are disabled. Please configure your GEMINI_API_KEY in the .env file."
        
    prompt = f"""
    You are an expert biotechnologist and chemical engineer specializing in dark fermentation and biohydrogen production.
    I have just run an anaerobic digestion simulation (pyADM1) with Model Predictive Control. 
    
    Here is the summary data from the run:
    Optimal pH: {run_data.get('optimal_pH', run_data.get('optimal_ph'))}
    Optimal Temperature: {run_data.get('optimal_temperature', run_data.get('optimal_temp'))} °C
    Total Hydrogen Yield: {run_data.get('hydrogen_yield', run_data.get('optimal_h2_yield'))} cubic meters
    MPC Improvement over uncontrolled: {run_data.get('improvement_pct', run_data.get('mpc_improvement'))} %
    Levelized Cost of Hydrogen (LCOH): ${run_data.get('lcoh')} / kg
    
    Write a concise, professional "Executive Insight Report" in a single paragraph of exactly 3 to 4 sentences. 
    Include an interpretation of the yield and cost, the biological realism (like pH suitability), and one actionable next step.
    CRITICAL: Provide ONLY plain text. Do not use ANY markdown formatting, no bullet points, no bold tags like **, and absolutely no HTML tags like <br>.
    Do not include introductory conversational text like "Here is your report". 
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI insight: {str(e)}"


def diagnose_error(error_message, parameters):
    """Diagnoses ODE crashes or simulation logic errors."""
    if not is_ai_enabled():
        return f"Simulation Failed: {error_message}"
        
    prompt = f"""
    You are an expert in SciPy solvers and Anaerobic Digestion Modeling (ADM1).
    My python simulation crashed with this exact error:
    {error_message}
    
    The parameters the user selected were:
    Baseline pH: {parameters.get('baseline_ph')}
    Baseline Temp: {parameters.get('baseline_temp', 35)} C
    Simulation Days: {parameters.get('baseline_days', 10)}
    
    Write a 1-2 sentence human-friendly diagnosis explaining why this mathematical or biological failure occurred 
    and exactly what parameter the user should change to fix it. Keep it very concise.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Diagnosis failed. Original Error: {error_message}"


def chat_with_run(run_data, chat_history, new_message):
    """Handles chatbot interactions regarding a specific simulation run."""
    if not is_ai_enabled():
        return "AI features are disabled."
        
    system_prompt = f"""
    You are the "Biohydrogen Assistant". The user is viewing a detailed dashboard of a dark fermentation simulation run.
    The primary data for this run is:
    Optimal pH: {run_data.get('optimal_pH', run_data.get('optimal_ph'))}
    Optimal Temp: {run_data.get('optimal_temperature', run_data.get('optimal_temp'))} C
    H2 Yield: {run_data.get('hydrogen_yield', run_data.get('optimal_h2_yield'))} m3
    LCOH: ${run_data.get('lcoh')}/kg
    
    Answer their question biologically and financially referencing the numbers above. Be extremely concise (2-3 sentences max). 
    CRITICAL: Provide ONLY plain text. Do not use ANY markdown formatting, no asterisks **, no underscores, no special formatting whatsoever.
    """
    
    messages = [{"role": "user", "parts": [system_prompt]}]
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        messages.append({"role": role, "parts": [msg["content"]]})
        
    messages.append({"role": "user", "parts": [new_message]})
    
    try:
        response = model.generate_content(messages)
        return response.text
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

def compare_runs_ai(runs_data):
    """Generates a comparative summary for multiple simulation runs."""
    if not is_ai_enabled():
        return "AI features are currently disabled. Please provide a GEMINI_API_KEY."

    if len(runs_data) < 2:
        return "Not enough data to compare."

    # Format the data for the prompt
    runs_text = ""
    for run in runs_data:
        runs_text += f"\nRun: {run.get('run_name', run.get('run_id'))}\n"
        runs_text += f"  - pH: {run.get('optimal_ph')}, Temp: {run.get('optimal_temp')} C\n"
        runs_text += f"  - H2 Yield: {run.get('h2_yield')} \n"
        runs_text += f"  - MPC Improvement: {run.get('mpc_improvement')}% \n"
        runs_text += f"  - LCOH: ${run.get('lcoh')}/kg\n"

    prompt = f"""
As a bioprocess engineering AI, briefly compare the following biohydrogen simulation runs.
Which one performed the best economically (lowest LCOH) and biologically (highest H2 yield/MPC)?
Explain why briefly based on the parameters (pH, Temp).

Data:
{runs_text}

CRITICAL: Provide ONLY plain text. Do not use ANY markdown formatting, no asterisks **, no underscores, no special formatting whatsoever. Keep it under 4 sentences.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

