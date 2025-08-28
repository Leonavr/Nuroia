# app/ui/main_window.py
import logging
import threading
from datetime import datetime

import ttkbootstrap as ttk
from tkinter import messagebox

from brainflow import BoardIds
from ttkbootstrap import WARNING, SECONDARY, DANGER, SUCCESS

from app.analysis.llm_analyzer import LLMAnalyzer
from app.core.profiles import load_profiles, save_profiles
from app.core.session import NeurofeedbackSession
# --- MODIFIED IMPORTS ---
# The old device classes are removed.
from app.devices.brainflow_device import BrainflowDevice
from app.devices.neurobit_device import NeurobitDevice
# The old eeg_processor is replaced with the new one.
from app.processing.eeg_processor import calculate_snapshot_metrics, calculate_average_tbr
from app.ui.frames import (
    WelcomeFrame, InstructionsFrame, CalibrationFrame, TrainingFrame, ReportFrame, HistoryFrame, PostSessionSurveyFrame,
    CheckInFrame, RecommendationFrame, TBRCalibrationIntroFrame, TBRCalibrationStepFrame
)

log = logging.getLogger(__name__)

class NuroiaApp(ttk.Window):
    """
    The main application class. Acts as a controller for the GUI.
    Manages frames, device connection, and session state.
    """

    def __init__(self):
        super().__init__(themename="superhero")
        self.title("Nuroia")
        self.geometry("950x800")

        # --- State Management ---
        self.device = None
        self.profiles = load_profiles()
        self.current_user = None
        self.session = None
        self.tbr_calibration_data = {}  # NEW: To temporarily store calibration data

        self.llm_analyzer = LLMAnalyzer()

        # --- Device Management ---
        self.available_devices = {
            "Neurobit Optima": NeurobitDevice,
            "Muse 2 / S": lambda: BrainflowDevice(BoardIds.MUSE_S_BOARD.value),
            "Synthetic": lambda: BrainflowDevice(BoardIds.SYNTHETIC_BOARD.value)
        }

        # --- Frame Management ---
        container = ttk.Frame(self, padding=10)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        # --- NEW: Added TBR Calibration frames ---
        for F in (WelcomeFrame, InstructionsFrame, CalibrationFrame, TrainingFrame, ReportFrame,
                  HistoryFrame, PostSessionSurveyFrame, CheckInFrame, RecommendationFrame,
                  TBRCalibrationIntroFrame, TBRCalibrationStepFrame):
            frame_name = F.__name__
            frame = F(container, self)
            self.frames[frame_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame('WelcomeFrame')

    def show_frame(self, frame_name: str, **kwargs):
        """Brings the specified frame to the front."""
        frame = self.frames[frame_name]
        if hasattr(frame, "on_show"):
            frame.on_show(**kwargs)
        frame.tkraise()

    def prepare_training_session(self, user_name: str, device_name: str, session_type: str, duration_str: str,
                                 exercise_type: str):
        """
        MODIFIED: Simplified device check.
        """
        if not user_name or not device_name:
            messagebox.showerror("Error", "Please select a profile and device.")
            return

        if not (self.device and self.device.is_connected()):
            messagebox.showerror("Device Error",
                                 f"{device_name} is not connected. Please select it from the list and wait for connection.")
            return

        self.current_user = user_name
        if user_name not in self.profiles:
            self.profiles[user_name] = {"created_at": str(datetime.now()), "sessions": []}
            save_profiles(self.profiles)
            self.frames['WelcomeFrame'].update_profile_list()

        self.session = NeurofeedbackSession(user_name, session_type, int(duration_str.split()[0]), exercise_type)
        self.show_frame('InstructionsFrame')

    def start_mind_check_in(self):
        """
        MODIFIED: Now includes logic to create a new profile if one doesn't exist,
        preventing a KeyError during calibration.
        """
        welcome_frame = self.frames['WelcomeFrame']
        user_name = welcome_frame.profile_combo.get()
        device_name = welcome_frame.device_combo.get()

        if not user_name or not device_name:
            messagebox.showerror("Error", "Please select a profile and a device.")
            return

        if not (self.device and self.device.is_connected()):
            messagebox.showerror("Device Error",
                                 f"{device_name} is not connected. Please select it from the list and wait for connection.")
            return
        log.info("--- Check-in button clicked. Debugging device state: ---")
        log.info(f"Device object is: {self.device}")
        if self.device:
            log.info(f"Device is_connected() returns: {self.device.is_connected()}")
        self.current_user = user_name

        if user_name not in self.profiles:
            log.info(f"Creating new profile: {user_name}")
            self.profiles[user_name] = {"created_at": str(datetime.now()), "sessions": []}
            save_profiles(self.profiles)
            # Also, update the combobox to reflect the new profile
            welcome_frame.update_profile_list()
            welcome_frame.profile_combo.set(user_name)

        user_profile = self.profiles.get(user_name, {})

        # The rest of the logic remains the same
        if 'tbr_range' not in user_profile:
            log.info("TBR range not found for user. Initiating one-time TBR calibration.")
            self.show_frame('TBRCalibrationIntroFrame')
        else:
            log.info("TBR range found in profile. Starting daily check-in.")
            self.show_frame('CheckInFrame')

    def start_tbr_calibration(self):
        """
        Starts the first step of the two-part TBR calibration process.
        Ensures a device is connected before starting.
        """
        device_name = self.frames['WelcomeFrame'].device_combo.get()
        log.info("Starting TBR calibration process...")
        self._connect_device_and_show_frame(device_name, 'TBRCalibrationStepFrame', step='focus')

    def process_tbr_calibration_step(self, step_name: str, collected_data: list[dict]):
        """
        Processes data from a calibration step. If it's the first step,
        it launches the second. If it's the second, it finalizes the process.
        """
        self.tbr_calibration_data[step_name] = collected_data
        log.info(f"Finished TBR calibration step '{step_name}' with {len(collected_data)} data points.")

        if step_name == 'focus':
            # This was the first step, now launch the second
            log.info("Starting 'relax' step of TBR calibration...")
            self.show_frame('TBRCalibrationStepFrame', step='relax')
        elif step_name == 'relax':
            # This was the second and final step, now process and save
            self._finalize_tbr_calibration()

    def _finalize_tbr_calibration(self):
        """
        MODIFIED: Now forces the user to recalibrate if the results are not ideal,
        instead of creating a confusing provisional range.
        """
        log.info("Finalizing TBR calibration...")
        focus_data = self.tbr_calibration_data.get('focus', [])
        relax_data = self.tbr_calibration_data.get('relax', [])

        if not focus_data or not relax_data:
            messagebox.showerror("Calibration Error", "Could not collect sufficient data.")
            return self.show_frame('WelcomeFrame')

        tbr_focused_avg = calculate_average_tbr(focus_data)
        tbr_relax_avg = calculate_average_tbr(relax_data)

        if tbr_focused_avg is None or tbr_relax_avg is None:
            messagebox.showerror("Calculation Error", "Failed to calculate averages from calibration data.")
            return self.show_frame('WelcomeFrame')

        # --- NEW, STRICTER LOGIC ---
        # We now require the focus TBR to be clearly lower than the relax TBR.
        if tbr_focused_avg >= tbr_relax_avg:
            log.warning(
                f"Calibration failed: Focus TBR ({tbr_focused_avg:.2f}) was not lower than Relax TBR ({tbr_relax_avg:.2f}).")

            # Show a clear message and guide the user to try again.
            messagebox.showinfo("Calibration Unsuccessful",
                                "Unfortunately, we could not detect a clear difference between your focus and relax states, likely due to muscle tension or distraction.\n\n"
                                "Please try the 2-minute calibration again, making sure to relax your face and jaw muscles.")

            # Reset and send the user back to the start of the calibration process.
            self.tbr_calibration_data = {}
            return self.show_frame('TBRCalibrationIntroFrame')

        # --- This code now only runs on successful calibration ---
        # Calculate a dynamic buffer (e.g., 10% of the measured range)
        dynamic_buffer = (tbr_relax_avg - tbr_focused_avg) * 0.10
        range_high = tbr_relax_avg + dynamic_buffer
        range_low = max(0, tbr_focused_avg - dynamic_buffer)

        tbr_range = [round(range_high, 2), round(range_low, 2)]

        self.profiles[self.current_user]['tbr_range'] = tbr_range
        save_profiles(self.profiles)
        log.info(f"SUCCESS: Saved new TBR range for user {self.current_user}: {tbr_range}")

        self.tbr_calibration_data = {}
        # Proceed to the check-in only after a successful calibration
        self.show_frame('CheckInFrame')

    def _connect_device_and_show_frame(self, device_name: str, frame_name: str, **kwargs):
        """
        MODIFIED: Helper function to ensure a device is connected, then show a specific frame.
        This version avoids the TypeError by simplifying the device check.
        """
        # The user has already selected and connected the device via the proactive thread.
        # We just need to ensure it's still connected.
        if self.device and self.device.is_connected():
            # If the connected device's name matches what's expected, we are good to go.
            if self.device.device_name == device_name:
                log.info(f"Device '{device_name}' is ready. Showing frame '{frame_name}'.")
                self.show_frame(frame_name, **kwargs)
            else:
                # This can happen if the user changes the dropdown after connecting.
                messagebox.showerror("Device Mismatch",
                                     f"A different device ({self.device.device_name}) is currently connected. Please disconnect it first or select it from the list.")
        else:
            # If no device is connected, prompt the user to connect it first.
            messagebox.showerror("Connection Error",
                                 f"Device '{device_name}' is not connected. Please select it from the dropdown on the main screen and wait for the connection to establish.")

    def force_tbr_recalibration(self):
        """
        NEW: A dedicated method to allow the user to manually trigger
        the TBR calibration process from the main menu.
        """
        welcome_frame = self.frames['WelcomeFrame']
        user_name = welcome_frame.profile_combo.get()
        device_name = welcome_frame.device_combo.get()

        if not user_name:
            messagebox.showinfo("Information", "Please select a profile before starting calibration.")
            return

        if not (self.device and self.device.is_connected()):
            messagebox.showerror("Device Error",
                                 f"{device_name} is not connected. Please select it from the list and wait for connection.")
            return

        self.current_user = user_name
        log.info(f"User '{user_name}' is manually starting TBR recalibration.")
        self.show_frame('TBRCalibrationIntroFrame')

    def process_check_in_results(self, snapshot_data: list[dict]):
        log.info(f"Received {len(snapshot_data)} data points for check-in analysis. Starting analysis thread.")
        self.show_frame('RecommendationFrame',
                        recommended_session_args={},
                        llm_insight="Analyzing your mind state...")
        thread = threading.Thread(target=self._process_check_in_thread, args=(snapshot_data,), daemon=True)
        thread.start()

    def _process_check_in_thread(self, snapshot_data: list[dict]):
        """
        MODIFIED: This background thread now passes the user's profile to the
        snapshot calculation function.
        """
        user_profile = self.profiles.get(self.current_user, {})
        metrics = calculate_snapshot_metrics(snapshot_data, user_profile)

        if not metrics:
            self.after(0, messagebox.showerror, "Analysis Error", "Could not analyze the collected data.")
            self.after(0, self.show_frame, 'WelcomeFrame')
            return

        if metrics['focus'] < 55:  # Lowered threshold slightly
            rec_args = {'session_type': 'attention', 'exercise_type': 'Focus (Rocket Game)'}
        elif metrics['relaxation'] < 50:
            rec_args = {'session_type': 'meditation', 'exercise_type': 'Zen Garden'}
        else:
            rec_args = {'session_type': 'attention', 'exercise_type': 'Focus (Sunrise)'}

        device_name = self.device.__class__.__name__ if self.device else "Unknown Device"
        llm_insight = self.llm_analyzer.analyze_check_in(metrics, self.current_user, device_name)

        self.after(0, lambda: self.show_frame('RecommendationFrame',
                                              recommended_session_args=rec_args,
                                              llm_insight=llm_insight))

    def start_calibration(self):
        """
        Shows the CalibrationFrame, which will now calculate the baseline statistics
        (mean, std) as per the document's recommendation[cite: 161, 218].
        """
        self.show_frame('CalibrationFrame')

    def start_training(self, baseline_stats: dict, initial_buffer=None):
        """
        MODIFIED: This is now called by CalibrationFrame after it finishes.
        The baseline_stats (mean, std) are stored in the session for Z-score calculation.
        """
        self.session.baseline_stats = baseline_stats
        log.info(f"Starting training with baseline stats: {baseline_stats}")
        self.show_frame('TrainingFrame', session=self.session, initial_buffer=initial_buffer)

    def process_finished_session(self, session: NeurofeedbackSession):
        """
        Saves session data object (in the new dictionary format) and transitions to the survey.
        """
        # This method first saves the session data to a CSV file.
        # The save_to_csv method now also stores the filename in session.csv_filename.
        session.save_to_csv()

        if session.user_name not in self.profiles:
            self.profiles[session.user_name] = {"created_at": str(datetime.now()), "sessions": []}

        session_entry = {
            "file": session.csv_filename,
            "report_text": "",
            "final_threshold": session.final_threshold
        }
        self.profiles[session.user_name]["sessions"].append(session_entry)
        save_profiles(self.profiles)
        self.show_frame('PostSessionSurveyFrame', session=session)

    def on_closing(self):
        """Handles the application window closing event."""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            if self.device:
                self.device.disconnect()
            self.destroy()

    def on_device_select(self, device_name):
        """
        Handles device selection from the dropdown. Connects in a background thread.
        This replaces the old, complex logic.
        """
        if not device_name:
            return

        # Check if we are already connected to the selected device
        if self.device and self.device.device_name == device_name and self.device.is_connected():
            log.info(f"Device {device_name} is already connected. No action needed.")
            return

        welcome_frame = self.frames['WelcomeFrame']
        welcome_frame.device_status_label.config(text=f"Status: Connecting to {device_name}...", bootstyle=WARNING)
        welcome_frame.disconnect_button.pack_forget()

        thread = threading.Thread(target=self._connect_device_thread, args=(device_name,), daemon=True)
        thread.start()

    def _connect_device_thread(self, device_name: str):
        """
        MODIFIED: Now instantiates the correct device class from the dictionary.
        """
        welcome_frame = self.frames['WelcomeFrame']
        try:
            if self.device:
                self.device.disconnect()

            # Look up the correct class or lambda function to create the device instance
            device_constructor = self.available_devices.get(device_name)
            if not device_constructor:
                raise ValueError(f"No constructor found for device: {device_name}")

            self.device = device_constructor()
            self.device.device_name = device_name  # Store the name for the UI

            if self.device.connect():
                log.info(f"Proactively connected to {device_name}.")
            else:
                log.error(f"Proactive connection to {device_name} failed.")
                self.device = None
        except Exception as e:
            log.critical(f"Error during proactive connection: {e}", exc_info=True)
            self.device = None
        finally:
            self.after(0, welcome_frame.sync_ui_with_device_state)

    def disconnect_device(self):
        """
        MODIFIED: Now calls the centralized UI sync method.
        """
        if self.device:
            self.device.disconnect()
            self.device = None
            log.info("Device disconnected by user.")

            welcome_frame = self.frames['WelcomeFrame']
            # We don't need to manually update the UI here anymore.
            # We just call the sync method which knows what to do.
            welcome_frame.sync_ui_with_device_state()