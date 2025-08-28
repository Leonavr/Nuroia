# app/analysis/llm_analyzer.py
import logging
import os

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

from app.core.session import NeurofeedbackSession

log = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    Analyzes a completed NeurofeedbackSession using the Google Gemini API.
    Uses an advanced, empathetic persona for more human-like responses.
    """

    def __init__(self):
        """Initializes the Gemini model using an API key from environment variables."""
        load_dotenv()
        self.model = None
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_engine = os.getenv("GEMINI_ENGINE")
        try:
            if not self.api_key:
                log.critical("GEMINI_API_KEY not found in environment variables.")
                return

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.api_engine)
            log.info(f"{self.api_engine} model initialized successfully.")
        except Exception as e:
            log.critical(f"Failed to initialize Gemini model: {e}", exc_info=True)

    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """Calls the Gemini API using the new structured prompt approach."""
        if not self.model:
            return "Analysis Error: The AI model is not loaded."

        log.info(f"--- Sending request to {self.api_engine} ---")
        try:
            # The API handles instructions and context within a single prompt.
            full_prompt = f"{system_prompt}\n\n<user_request>\n{user_prompt}\n</user_request>"

            response = self.model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            log.error(f"Error during Gemini model inference: {e}", exc_info=True)
            return "Analysis Error: Could not get a response from the AI model."

    def _summarize_session_data(self, df: pd.DataFrame, session: NeurofeedbackSession) -> str:
        """
        MODIFIED: Analyzes the session DataFrame and returns a richer string summary
        of key metrics, now context-aware based on the session type.
        """
        if df.empty or len(df) < 20:
            return "No detailed data available."

        summary = []
        session_type = session.session_type

        if session_type == "meditation":
            abr_mean = df['proc_meditation'].mean()
            alpha_rel_mean = df['Rel_Alpha'].mean()
            meditation_std = df['meditation'].std()

            summary.append(f"- Average Alpha/Beta Ratio: {abr_mean:.2f} (higher is better for relaxation).")
            summary.append(f"- Average Relative Alpha Power: {alpha_rel_mean:.2f} (higher indicates a relaxed state).")
            summary.append(
                f"- Relaxation Stability (Std Dev): {meditation_std:.2f} (lower means more stable relaxation).")

            top_10_percent = df[df['meditation'] >= df['meditation'].quantile(0.9)]
            summary.append(f"- ABR during best relaxation moments: {top_10_percent['proc_meditation'].mean():.2f}")

        else:
            tbr_mean = df['Theta_Beta_Ratio'].mean()
            beta_rel_mean = df['Rel_Beta'].mean()
            attention_std = df['attention'].std()
            drops = (df['attention'].diff() < -30).sum()

            summary.append(f"- Average Theta/Beta Ratio (TBR): {tbr_mean:.2f} (lower is better for focus).")
            summary.append(f"- Average Relative Beta Power: {beta_rel_mean:.2f} (higher indicates active thinking).")
            summary.append(f"- Attention Stability (Std Dev): {attention_std:.2f} (lower means more stable focus).")
            summary.append(f"- Number of Major Focus Drops: {drops}")

            top_10_percent = df[df['attention'] >= df['attention'].quantile(0.9)]
            bottom_10_percent = df[df['attention'] <= df['attention'].quantile(0.1)]
            summary.append(f"- TBR during best focus moments: {top_10_percent['Theta_Beta_Ratio'].mean():.2f}")
            summary.append(f"- TBR during moments of distraction: {bottom_10_percent['Theta_Beta_Ratio'].mean():.2f}")

        return "\n".join(summary)

    def analyze_check_in(self, metrics: dict, user_name: str, device_name: str) -> str:
        """
        MODIFIED: Now accepts a device_name and checks if valence was calculated
        to provide a more informative prompt to the LLM.
        """
        system_prompt = (
            "<role>"
            "You are 'Nuroia', a personal neurofeedback coach. Your tone is supportive and insightful."
            "</role>\n"
            "<instructions>"
            "Your task is to analyze the user's daily 'Mind State Check-in' and provide a short insight.\n"
            "1.  Interpret the key metrics provided. **When the 'Emotional Valence Score' is available, you MUST incorporate its meaning into your analysis.** A low score (under 40) can suggest tension or a withdrawn state, while a high score (over 60) suggests a positive, engaged state.\n"
            "2.  Connect the metrics. For example, high focus but low relaxation and low valence suggests tense, forced concentration.\n"
            "3.  Provide one small, actionable piece of advice for the day based on your analysis.\n"
            "4.  **If the 'Emotional Valence Score' is 'Not Available'**, briefly and neutrally mention that the user's current device does not support this metric.\n"
            "5.  Keep the entire response to 3-5 sentences.\n"
            "6.  Do not give medical advice.\n"
            "</instructions>"
        )

        # --- NEW: Conditionally build the prompt based on the flag ---
        valence_calculated = metrics.get('valence_calculated', False)
        if valence_calculated:
            valence_prompt_line = f"- Emotional Valence Score: {metrics.get('valence', 'N/A')} (higher is associated with positive states)"
        else:
            valence_prompt_line = f"- Emotional Valence Score: Not Available (The '{device_name}' device does not have the required frontal sensors for this metric)."

        user_prompt = f"""
           My name is {user_name}. Here are the results of my 60-second mind state check-in. The scores are from 0-100.

           **My Mind State Snapshot:**
           - Relaxation Score: {metrics.get('relaxation', 'N/A')}
           - Focus Score: {metrics.get('focus', 'N/A')}
           {valence_prompt_line}

           Based on my data, provide a short, supportive insight for my day.
           """
        return self._call_llm_api(system_prompt, user_prompt)

    def analyze_session(self, session: NeurofeedbackSession) -> str:
        """
            Creates a detailed prompt, including full CSV data, and returns a
            neurophysiologically-informed summary from the LLM.
        """
        if not session.data_log:
            return "Not enough data for analysis."

        # --- IMPROVED & STRUCTURED SYSTEM PROMPT FOR GEMINI ---
        system_prompt = (
            "<role>"
            "You are 'Nuroia', a personal neurofeedback coach and AI neurophysiologist. Your tone is empathetic, supportive, and insightful. You speak directly to the user in a friendly, personal tone."
            "</role>\n"
            "<instructions>"
            "Your goal is to analyze the user's session and provide a concise, inspiring summary. Adhere to these principles:\n"
            "1.  **Be Flexible in Your Communication:** Start the conversation in various ways. Sometimes with a greeting ('Hi, {name}!'), sometimes with a direct observation ('Looks like today was an interesting session.'), and sometimes with an encouraging comment. Avoid repetitive openings.\n"
            "2.  **Analyze, Don't Just List:** Do not simply state facts. Find connections between the user's subjective feelings and the objective data. Here are some examples of analytical patterns to look for:\n"
            "    - **High Effort, Low Feeling:** If 'Time in Improvement Zone' is high but the user rated their focus low, they might be trying too hard. Suggest a more relaxed approach.\n"
            "    - **Volatility:** If 'Attention Stability' is high (meaning high variability), but 'TBR' is low (good), it could indicate short, intense bursts of focus alternating with distractions. Frame this as a normal part of training.\n"
            "    - **Performance Ceiling:** If 'Peak Log Raw Attention' significantly exceeds the calibrated range maximum, it means the user surpassed expectations. This is a great sign of progress.\n"
            "    - **Difficulty and Progress:** If the 'Difficulty' increased significantly during the session, it means the user consistently met the challenge.\n"
            "3.  **Value the Effort:** Always emphasize the importance of the practice itself and of consistency, not just high scores.\n"
            "4.  **Explain Neurophysiology Simply:** Explain complex terms (like TBR) very simply. For example: 'Your TBR score improved, which means your brain was in a more focused and less drowsy state.'\n"
            "</instructions>\n"
            "<output_rules>"
            "1.  **Language:** Respond exclusively in English.\n"
            "2.  **Length:** Keep the summary concise (approximately 5-8 sentences).\n"
            "3.  **Format:** Provide only the direct response to the user without any meta-commentary or self-reflection.\n"
            "</output_rules>"
        )

        # FIXED: Call the summary function once and store the result
        data_summary_str = "No detailed data available for summary."
        try:
            # You can decide if you want to use the live data_log or the saved CSV
            # Using data_log is faster if the CSV is just for historical purposes
            df = pd.DataFrame(session.data_log)
            if not df.empty:
                data_summary_str = self._summarize_session_data(df, session)

        except Exception as e:
            logging.error(f"Error creating data summary for LLM: {e}")
            data_summary_str = "Error processing session data."

        user_prompt = f"""
        My name is {session.user_name}. Please analyze my neurofeedback session data below.
        
        **My Subjective Feedback:**
        - How I felt after the session (1-5, 5=best): {session.user_feeling}
        - My self-rated focus (1-5, 5=best): {session.user_focus_rating}
        - My personal notes: "{session.user_notes if session.user_notes else 'None'}"
        
        **My Objective Session Metrics:**
        - Session Type: {session.session_type.capitalize()}
        - Time in 'Consistency Zone' (Maintaining State): {session.time_in_consistency_zone:.1f}%
        - Time in 'Improvement Zone' (Pushing Limits): {session.time_in_success_zone:.1f}%
        - Longest Sustained Period: {session.longest_streak:.1f} seconds
        - Difficulty Adjustments: {session.threshold_changes} times
        - Final Difficulty Threshold: {session.final_threshold}
        
        **Key Neurophysiological Summary:**
        {data_summary_str}
        
        Based on the instructions provided, provide your empathetic analysis.
        """
        return self._call_llm_api(system_prompt, user_prompt)