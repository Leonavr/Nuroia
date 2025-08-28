# app/core/session.py
import logging
import os
import pandas as pd
from datetime import datetime

log = logging.getLogger(__name__)

class NeurofeedbackSession:
    """
    Represents a single neurofeedback training session.
    This class holds all session-related data, including user info,
    timestamps, calculated metrics, and the final score.
    """

    def __init__(self, user_name: str, session_type: str, duration_minutes: int, exercise_type: str):
        self.user_name: str = user_name
        self.session_type: str = session_type  # 'attention' or 'meditation'
        self.exercise_type: str = exercise_type
        self.start_time: datetime = datetime.now()
        self.duration_seconds: int = duration_minutes * 60
        self.data_log: list = []
        self.score: int = 0
        self.baseline: dict = {}  # Populated after the calibration phase
        # Initialize with default "not calculated" values
        self.time_to_first_success: float | None = None
        self.longest_streak: float = 0.0
        self.threshold_crossings: int = 0
        self.threshold_changes: int = 0
        self.minute_by_minute_performance: dict = {}

        self.calibrated_log_ranges: dict | None = None

        self.user_feeling: int | None = None  # Scale 1-5
        self.user_focus_rating: int | None = None  # Scale 1-5
        self.user_notes: str = ""  # Optional text notes
        self.final_threshold = 0
        self.time_in_success_zone = 0.0
        self.baseline_threshold = 0
        self.time_in_consistency_zone = 0.0
        self.report_text = ""

    def add_data_point(self, processed_data: dict, threshold: int):
        """
        Appends a complete data point dictionary to the session's data log.
        """
        data_to_append = processed_data.copy()
        data_to_append['timestamp'] = datetime.now()
        data_to_append['threshold'] = threshold
        self.data_log.append(data_to_append)

    def save_to_csv(self):
        """
        Saves the complete session data log to a CSV file with a structured column order.
        """
        # Create the directory and define the filename
        os.makedirs("sessions", exist_ok=True)
        filename = f"sessions/{self.user_name}_{self.session_type}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_filename = filename  # Store filename in the object

        if not self.data_log:
            log.warning("Data log is empty. Nothing to save to %s.", filename)
            # Create an empty file so the program doesn't crash looking for it
            with open(filename, 'w') as f:
                f.write('')
            return filename

        # Create a DataFrame from our list of data dictionaries
        df = pd.DataFrame(self.data_log)

        # --- NEW LOGIC TO ORDER COLUMNS ---
        # Define the desired order of columns for better readability
        core_columns = ['timestamp', 'attention', 'meditation', 'threshold',
                        'raw_attention', 'raw_meditation']
        band_columns = ['Rel_Alpha', 'Rel_Beta', 'Rel_Theta', 'Rel_Delta',
                        'Rel_Gamma', 'Theta_Beta_Ratio']

        # Get all columns that actually exist in the DataFrame
        existing_columns = df.columns.tolist()

        # Build the final ordered list of columns to ensure no data is lost
        final_ordered_columns = [col for col in core_columns if col in existing_columns]
        final_ordered_columns += [col for col in band_columns if col in existing_columns]
        final_ordered_columns += [col for col in existing_columns if col not in final_ordered_columns]

        # Reorder the DataFrame according to our list
        df = df[final_ordered_columns]
        # --- END OF NEW LOGIC ---

        # Save the complete, reordered DataFrame to CSV
        df.to_csv(filename, index=False)
        log.info(f"Session data saved to: {filename}")
        return filename