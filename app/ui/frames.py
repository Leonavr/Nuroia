# app/ui/frames.py
import os
import threading
import tkinter as tk
from collections import deque
from tkinter import messagebox

import pygame
import ttkbootstrap as ttk
from matplotlib.figure import Figure
from ttkbootstrap.scrolled import ScrolledText, ScrolledFrame

from ttkbootstrap.constants import *
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import beepy as bp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import calendar
import math
import logging

from app.analysis.llm_analyzer import LLMAnalyzer
from app.core.profiles import save_profiles
from app.core.session import NeurofeedbackSession
from app.utils import resource_path

log = logging.getLogger(__name__)

# Import from other modules in the project
from app.processing.eeg_processor import (
    process_brainflow_eeg, is_signal_plausible # MODIFIED IMPORT
)
# This avoids circular import issues with type hinting
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.ui.main_window import NuroiaApp


class WelcomeFrame(ttk.Frame):
    """The first screen of the application, for user and session setup."""

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller

        # --- WIDGETS ---
        ttk.Label(self, text="Nuroia", font=("Helvetica", 28, "bold"), bootstyle=PRIMARY).pack(pady=(20, 10))
        ttk.Label(self, text="Your Personal Neuro-Enhancement Companion", font=("Helvetica", 14),
                  bootstyle=SECONDARY).pack()

        ttk.Button(self, text="🚀 Daily Mind State Check-in & Recommendation", bootstyle=SUCCESS,
                   command=self.controller.start_mind_check_in).pack(pady=20, ipady=10)

        settings_frame = ttk.Frame(self, padding=(0, 20, 0, 10))

        ttk.Label(settings_frame, text="Select or Create Profile:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.profile_combo = ttk.Combobox(settings_frame, width=28, bootstyle=PRIMARY)
        self.profile_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(settings_frame, text="Select Device:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.device_combo = ttk.Combobox(settings_frame, width=28, state="readonly", bootstyle=PRIMARY)
        self.device_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_selected_event)

        # This frame will hold the status label and disconnect button
        status_frame = ttk.Frame(settings_frame)
        self.device_status_label = ttk.Label(status_frame, text="Status: Disconnected", bootstyle=SECONDARY)
        self.device_status_label.pack(side="left", padx=5)

        # MODIFIED: Changed bootstyle to a string format for better compatibility.
        self.disconnect_button = ttk.Button(status_frame, text="Disconnect", bootstyle="danger-outline",
                                            command=self.controller.disconnect_device)
        # The button is initially hidden and will be shown by sync_ui_with_device_state()

        status_frame.grid(row=1, column=2, sticky='w', padx=5, pady=5)

        ttk.Label(settings_frame, text="Session Duration:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.duration_combo = ttk.Combobox(settings_frame, values=["1 minute", "5 minutes", "8 minutes"], width=28,
                                           state="readonly", bootstyle=PRIMARY)
        self.duration_combo.current(0)
        self.duration_combo.grid(row=2, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(settings_frame, text="Focus Exercise:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.exercise_combo = ttk.Combobox(
            settings_frame,
            values=["Focus (Sunrise)", "Focus (Rocket Game)", "Focus (Circle)", "Audio Only (Eyes Closed)"],
            width=28,
            state="readonly",
            bootstyle=PRIMARY
        )
        self.exercise_combo.current(0)
        self.exercise_combo.grid(row=3, column=1, sticky='w', padx=5, pady=5)

        settings_frame.pack(pady=20)

        action_frame = ttk.Frame(self)
        ttk.Button(action_frame, text="Focus Training", bootstyle="success-outline", command=self.start_focus).pack(
            side="left", padx=10, ipady=10, ipadx=10)
        ttk.Button(action_frame, text="Relaxation Training", bootstyle="info-outline", command=self.start_relax).pack(
            side="left", padx=10, ipady=10, ipadx=10)
        action_frame.pack()
        ttk.Button(action_frame, text="Recalibrate Range", bootstyle="warning-outline",
                   command=self.controller.force_tbr_recalibration).pack(side="left", padx=10, ipady=10, ipadx=10)
        ttk.Button(self, text="History & Progress", bootstyle=SECONDARY, command=self.show_history).pack(pady=30,
                                                                                                         ipady=5)

    def on_show(self, **kwargs):
        """Called when the frame is brought to the front."""
        self.update_profile_list()
        self.update_device_list()
        self.sync_ui_with_device_state()  # Ensure UI is correct on show

    def update_profile_list(self):
        """Fetches the latest profile list and updates the combobox."""
        profiles = list(self.controller.profiles.keys())
        self.profile_combo['values'] = profiles
        if profiles and not self.profile_combo.get():
            self.profile_combo.current(0)

    def update_device_list(self):
        """
        MODIFIED: Correctly gets device names (keys) from the controller's
        available_devices dictionary. This fixes the AttributeError.
        """
        # self.controller.available_devices is now a dictionary like {"Muse 2 / S": Class}
        # We need its keys for the combobox values.
        devices = list(self.controller.available_devices.keys())
        self.device_combo['values'] = devices

        # Set a default selection if nothing is selected yet
        if devices and not self.device_combo.get():
            self.device_combo.current(0)

    def sync_ui_with_device_state(self):
        """
        NEW: A centralized method to update the UI based on the controller's device state.
        This is now the single source of truth for the connection UI.
        """
        device = self.controller.device
        if device and device.is_connected():
            # State: Connected
            device_name = device.device_name
            self.device_combo.set(device_name)
            self.device_status_label.config(text=f"Status: Connected", bootstyle=SUCCESS)
            self.disconnect_button.pack(side="left", padx=5)
        else:
            # State: Disconnected
            self.device_status_label.config(text="Status: Disconnected", bootstyle=SECONDARY)
            self.disconnect_button.pack_forget()

    def on_device_selected_event(self, event):
        """Called when a user selects a device from the combobox."""
        selected_device = self.device_combo.get()
        self.controller.on_device_select(selected_device)

    # --- Other methods (start_focus, start_relax, show_history) remain unchanged ---
    def start_focus(self):
        """Prepares and starts a focus training session."""
        self.controller.prepare_training_session(
            user_name=self.profile_combo.get(),
            device_name=self.device_combo.get(),
            session_type="attention",
            duration_str=self.duration_combo.get(),
            exercise_type=self.exercise_combo.get()
        )

    def start_relax(self):
        """Prepares and starts a relaxation training session."""
        self.controller.prepare_training_session(
            user_name=self.profile_combo.get(),
            device_name=self.device_combo.get(),
            session_type="meditation",
            duration_str=self.duration_combo.get(),
            exercise_type="Zen Garden"
        )

    def show_history(self):
        """Switches to the HistoryFrame for the selected user."""
        user = self.profile_combo.get()
        if not user:
            messagebox.showinfo("Information", "Please select a profile to view history.")
            return
        self.controller.show_frame('HistoryFrame', username=user)


class InstructionsFrame(ttk.Frame):
    """A simple frame displaying instructions before a session."""

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="Session Preparation", font=("Helvetica", 24, "bold"), bootstyle=PRIMARY).pack(pady=20)

        instructions_text = (
            "1. **Prepare Device:** Ensure your EEG device is turned on and charged.\n\n"
            "2. **Wear Correctly:** Place the sensors on your head according to the device's manual. "
            "Ensure a good contact with the skin.\n\n"
            "3. **Get Comfortable:** Sit in a comfortable chair. Try to avoid active movements "
            "or tensing your facial and jaw muscles during the session.\n\n"
            "4. **Relax:** Take a few deep breaths before you begin.\n\n"
            "When you are ready, click the button below to start a short calibration."
        )
        ttk.Label(self, text=instructions_text, justify="left", font=("Helvetica", 12)).pack(pady=20, padx=40, fill='x')

        ttk.Button(self, text="Start Calibration", bootstyle=SUCCESS, command=self.controller.start_calibration).pack(
            pady=20, ipady=10)


class CalibrationFrame(ttk.Frame):
    """Frame for calibrating the user's baseline EEG levels."""
    CALIBRATION_SECONDS = 60

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="Calibration", font=("Helvetica", 24, "bold"), bootstyle=INFO).pack(pady=20)

        self.signal_wait_label = ttk.Label(self, text="", font=("Helvetica", 14, "bold"), bootstyle=INFO)
        self.signal_wait_label.pack(pady=5)

        self.signal_warning_label = ttk.Label(self, text="", font=("Helvetica", 12, "bold"), bootstyle=DANGER)
        self.signal_warning_label.pack(pady=5)

        self.status_label = ttk.Label(self, text="Please remain calm. Measuring your baseline activity...",
                                      font=("Helvetica", 14))
        self.status_label.pack(pady=10)

        self.progress = ttk.Progressbar(self, length=400, mode='determinate', bootstyle="info-striped")
        self.progress.pack(pady=20)

        self.timer_label = ttk.Label(self, text="", font=("Helvetica", 16))
        self.timer_label.pack()

    def on_show(self, **kwargs):
        """Initializes and starts the baseline collection process."""
        self.session = self.controller.session
        self.calibration_raw_data = []
        self.progress['value'] = 0
        self.timer_label.config(text=f"{self.CALIBRATION_SECONDS} s")
        self.is_warning_active = False

        log.info("Attempting to start stream for baseline calibration...")
        if self.controller.device.start_stream():
            self.signal_wait_label.config(text="Waiting for a stable signal...")
            self.status_label.config(text="Please check the electrode contact.")
            self.wait_start_time = datetime.now()
            self.after(500, self.wait_for_signal)
        else:
            self.status_label.config(text="Startup Error. Check the device.", bootstyle=DANGER)
            messagebox.showerror("Error",
                                 "Failed to start measurement. Check electrode connection and restart the application.")
            self.controller.show_frame('WelcomeFrame')

    def wait_for_signal(self):
        """Checks for a stable signal before starting the timer and collection."""
        if (datetime.now() - self.wait_start_time).total_seconds() > 20:
            messagebox.showerror("Signal Error",
                                 "Could not get a stable signal from the device in 20 seconds. Please check the electrode connections and try again.")
            self.controller.device.stop_stream()
            self.controller.show_frame('WelcomeFrame')
            return

        if self.controller.device.check_signal_quality():
            log.info("Signal is stable. Starting calibration timer.")
            self.signal_wait_label.config(text="")
            self.status_label.config(text="Please remain calm. Measuring your baseline activity...")

            self.start_time = datetime.now()

            self.update_calibration()
        else:
            self.after(500, self.wait_for_signal)

    def update_calibration(self):
        """
        FINAL CORRECTED VERSION: Ensures the timer and progress bar
        perfectly sync with the signal status.
        """
        is_signal_good = self.controller.device.check_signal_quality()

        if not is_signal_good:
            if not self.is_warning_active:
                log.warning("Signal lost. Pausing calibration.")
                self.signal_warning_label.config(text="POOR SIGNAL QUALITY - CHECK CONTACT", bootstyle=DANGER)
                self.is_warning_active = True

            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.start_time = datetime.now() - timedelta(seconds=elapsed)

        else:
            if self.is_warning_active:
                log.info("Signal restored. Resuming calibration.")
                self.signal_warning_label.config(text="")
                self.is_warning_active = False

            raw_data = self.controller.device.get_data()
            if raw_data is not None and raw_data.shape[1] > 0:
                self.calibration_raw_data.append(raw_data)

            elapsed = (datetime.now() - self.start_time).total_seconds()
            progress_value = (elapsed / self.CALIBRATION_SECONDS) * 100
            self.progress['value'] = progress_value
            remaining = max(0, self.CALIBRATION_SECONDS - elapsed)
            self.timer_label.config(text=f"{int(remaining)} s")

            if remaining <= 0:
                self.finish_calibration()
                return

        self.after(200, self.update_calibration)

    def finish_calibration(self):
        """
        Processes the collected data to calculate baseline statistics (mean, std).
        This implements the "normalization relative to an individual baseline" strategy[cite: 161].
        """
        self.controller.device.stop_stream()  # Stop stream after collection

        if not self.calibration_raw_data:
            messagebox.showerror("Calibration Error", "No valid data was collected.")
            return self.controller.show_frame('WelcomeFrame')
        full_data = np.concatenate(self.calibration_raw_data, axis=1)
        device = self.controller.device
        primary_idx = device.get_primary_eeg_channel_index()
        eeg_channels = device.get_eeg_channels()
        sampling_rate = device.get_sampling_rate()
        window_size = sampling_rate * 2
        step_size = sampling_rate // 4
        all_powers = []
        num_samples = full_data.shape[1]
        for i in range(0, num_samples - window_size, step_size):
            window_chunk = full_data[:, i:i + window_size]
            indices_to_process = {'fp1': primary_idx}
            powers = process_brainflow_eeg(window_chunk, eeg_channels, sampling_rate, indices_to_process)
            if powers:
                all_powers.append(powers)
        if len(all_powers) < 20:
            messagebox.showerror("Calibration Error", "Not enough clean data to establish a baseline.")
            return self.controller.show_frame('WelcomeFrame')
        df = pd.DataFrame(all_powers)
        baseline_stats = {
            'attention': {'mean': df['proc_attention'].mean(), 'std': df['proc_attention'].std()},
            'meditation': {'mean': df['proc_meditation'].mean(), 'std': df['proc_meditation'].std()}
        }
        samples_per_update = sampling_rate * TrainingFrame.PROCESSING_WINDOW_SECONDS
        log.info(f"Baseline calibration complete. Stats: {baseline_stats}")
        self.status_label.config(text="Calibration complete! Starting training...")
        self.after(1500, lambda: self.controller.start_training(baseline_stats, initial_buffer=full_data[:, -samples_per_update:]))


class TBRCalibrationIntroFrame(ttk.Frame):
    """Explains the two-step TBR calibration process to the user."""

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="Personal  Calibration", font=("Helvetica", 24, "bold"), bootstyle=PRIMARY).pack(pady=20)

        instructions_text = (
            "To personalize your experience, we need to measure your unique brain patterns for focus and distraction.\n\n"
            "This is a one-time, 2-minute process with two steps:\n\n"
            "1.  **Focus Task (60s):** You will be asked to perform a simple mental task.\n"
            "2.  **Relax Task (60s):** You will be asked to relax and let your mind wander.\n\n"
            "This will help Nuroia understand you better. Please ensure you are in a quiet environment."
        )
        ttk.Label(self, text=instructions_text, justify="left", font=("Helvetica", 12), wraplength=700).pack(pady=20,
                                                                                                             padx=40,
                                                                                                             fill='x')

        ttk.Button(self, text="Start Focus Calibration", bootstyle=SUCCESS,
                   command=self.controller.start_tbr_calibration).pack(
            pady=20, ipady=10)
        ttk.Button(self, text="Back to Main Menu", bootstyle=(SECONDARY, OUTLINE),
                   command=lambda: controller.show_frame('WelcomeFrame')).pack(pady=10)


class TBRCalibrationStepFrame(ttk.Frame):
    """
    A generic frame for a timed data collection task, like TBR calibration.
    MODIFIED: Updated to use the BrainFlow data acquisition model and wait for stable signal.
    """
    CALIBRATION_STEP_SECONDS = 60

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller

        self.title_label = ttk.Label(self, text="", font=("Helvetica", 24, "bold"), bootstyle=INFO)
        self.title_label.pack(pady=20)

        self.signal_wait_label = ttk.Label(self, text="", font=("Helvetica", 14, "bold"), bootstyle=INFO)
        self.signal_wait_label.pack(pady=5)

        self.status_label = ttk.Label(self, text="", font=("Helvetica", 14), justify="center")
        self.status_label.pack(pady=10)

        self.progress = ttk.Progressbar(self, length=400, mode='determinate', bootstyle="info-striped")
        self.progress.pack(pady=20)

        self.timer_label = ttk.Label(self, text="", font=("Helvetica", 16))
        self.timer_label.pack()

        ttk.Button(self, text="Back to Main Menu", bootstyle=(SECONDARY, OUTLINE),
                   command=lambda: controller.show_frame('WelcomeFrame')).pack(pady=20, side="bottom")

    def on_show(self, step: str):
        """Initializes the frame for a specific calibration step ('focus' or 'relax')."""
        self.step_name = step
        self.raw_data_buffer = np.array([[] for _ in range(300)])
        self.collected_data = []
        self.progress['value'] = 0
        self.timer_label.config(text=f"{self.CALIBRATION_STEP_SECONDS} s")

        if step == 'focus':
            self.title_label.config(text="Step 1: Focus Task")
            self.status_label.config(
                text="Please concentrate for 60 seconds.\nFor example, count backwards from 1000 by 7s in your head.")
        else:
            self.title_label.config(text="Step 2: Relax Task")
            self.status_label.config(
                text="Please relax for 60 seconds.\nLet your mind wander freely, look around the room.")

        if not (self.controller.device and self.controller.device.is_running):
            self.controller.device.start_stream()

        self.signal_wait_label.config(text="Waiting for a stable signal...")
        self.wait_start_time = datetime.now()
        self.after(500, self.wait_for_signal)

    def wait_for_signal(self):
        """Checks for a stable signal before starting the timer."""
        if (datetime.now() - self.wait_start_time).total_seconds() > 20:
            messagebox.showerror("Signal Error", "Could not get a stable signal. Please check connections.")
            self.controller.device.stop_stream()
            self.controller.show_frame('WelcomeFrame')
            return

        if self.controller.device.check_signal_quality():
            log.info(f"✅ Signal stable for TBR step '{self.step_name}'. Starting.")
            self.signal_wait_label.config(text="")
            self.start_time = datetime.now()
            self.update_measurement()
        else:
            self.after(500, self.wait_for_signal)

    def update_measurement(self):
        """
        Gathers data for 60 seconds using the BrainFlow pull method and processes it in windows.
        """
        device = self.controller.device
        primary_idx = device.get_primary_eeg_channel_index()
        sampling_rate = device.get_sampling_rate()
        new_data = device.get_data()
        if new_data is not None and new_data.shape[1] > 0:
            if self.raw_data_buffer.shape[1] == 0:
                self.raw_data_buffer = new_data
            else:
                self.raw_data_buffer = np.concatenate((self.raw_data_buffer, new_data), axis=1)
        window_size = 2 * sampling_rate
        step_size = sampling_rate // 4
        while self.raw_data_buffer.shape[1] >= window_size:
            window_chunk = self.raw_data_buffer[:, :window_size]
            indices_to_process = {'fp1': primary_idx}
            powers = process_brainflow_eeg(window_chunk, device.get_eeg_channels(), sampling_rate, indices_to_process)
            if powers:
                bands_data = powers.get('bands', {})
                self.collected_data.append(bands_data)
            self.raw_data_buffer = self.raw_data_buffer[:, step_size:]
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress_value = (elapsed / self.CALIBRATION_STEP_SECONDS) * 100
        self.progress['value'] = progress_value
        remaining = max(0, self.CALIBRATION_STEP_SECONDS - elapsed)
        self.timer_label.config(text=f"{int(remaining)} s")
        if remaining <= 0:
            self.finish_step()
        else:
            self.after(100, self.update_measurement)

    def finish_step(self):
        """Passes the collected data for this step to the controller."""
        if self.controller.device:
            self.controller.device.stop_stream()
        self.controller.process_tbr_calibration_step(self.step_name, self.collected_data)


class TrainingFrame(ttk.Frame):
    """
    The main frame for the neurofeedback training session and visualization.
    FINAL VERSION: This class is completely rewritten to use the BrainFlow backend
    and the recommended Z-score normalization for feedback.
    """
    # The new threshold is a Z-score. A value of 1.0 means the goal is to be
    # 1 standard deviation *above* your own personal baseline average.
    # This provides a dynamic, personalized, and scientifically-grounded target.
    INITIAL_THRESHOLD_Z_SCORE = 1.0
    MIN_THRESHOLD_Z_SCORE, MAX_THRESHOLD_Z_SCORE = 0.5, 2.5  # Min/max Z-score targets

    # This constant determines how much historical data (in seconds) to use
    # for the processing window in each feedback loop update. 2 seconds is standard.
    PROCESSING_WINDOW_SECONDS = 4

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller

        self.audio_initialized = False
        try:
            pygame.mixer.init()
            self.audio_initialized = True
            log.info("Pygame mixer initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize pygame mixer: {e}")
            messagebox.showerror("Audio Error",
                                 "Could not initialize the audio system. Audio feedback will be unavailable.")
        # --- UI Setup ---
        self.info_frame = ttk.Frame(self)
        self.info_frame.pack(pady=10, fill='x', padx=20)
        self.title_label = ttk.Label(self.info_frame, text="", font=("Helvetica", 18, "bold"))
        self.title_label.pack(side='left', expand=True, fill='x')
        self.score_label = ttk.Label(self.info_frame, text="Score: 0", font=("Helvetica", 18, "bold"),
                                     bootstyle=SUCCESS)
        self.score_label.pack(side='right')

        self.signal_warning_label = ttk.Label(self, text="", font=("Helvetica", 12, "bold"), bootstyle=DANGER)
        self.signal_warning_label.pack(pady=5)

        style = ttk.Style()
        bg_color = style.lookup('TFrame', 'background')
        self.feedback_canvas = tk.Canvas(self, width=600, height=350, bg=bg_color, highlightthickness=0)
        self.feedback_canvas.pack(pady=10)

        self.status_label = ttk.Label(self, text="Get ready...", font=("Helvetica", 14), bootstyle=SECONDARY)
        self.status_label.pack()

        progress_frame = ttk.Frame(self)
        self.timer_label = ttk.Label(progress_frame, text="", font=("Helvetica", 12))
        self.timer_label.pack(side='left', padx=10)
        self.difficulty_label = ttk.Label(progress_frame, text="", font=("Helvetica", 12), bootstyle=INFO)
        self.difficulty_label.pack(side='left', padx=10)
        progress_frame.pack(pady=10)

        # Animation state variables
        self.pulse_angle = 0
        self.current_radius = 20.0
        self.stars = []

    def on_show(self, session: NeurofeedbackSession, initial_buffer=None):
        """Initializes the state for a new training session with baseline stats."""
        if initial_buffer is not None and initial_buffer.shape[1] > 0:
            self.raw_data_buffer = initial_buffer
        else:
            self.raw_data_buffer = np.array([[] for _ in range(300)])
        self.session = session
        self.update_job = None
        self.last_processed_powers = {}
        self.TIME_TO_INCREASE_DIFFICULTY = 3
        self.success_streak_start_time = None
        # The core feedback is now driven by a Z-score.
        self.smoothed_z_score = 0.0
        self.threshold = self.INITIAL_THRESHOLD_Z_SCORE

        # --- Prepare explanatory string for the LLM report ---
        # This preserves the context for the post-session analysis.
        self.session.threshold_calculation_details = (
            f"Your training started with a target Z-score of {self.threshold:.2f}. "
            f"This means the goal was to maintain your brain activity state at least {self.threshold:.2f} "
            f"standard deviations above your personal baseline, which was measured during the initial calibration."
        )
        log.info(f"New session starting. Reason: {self.session.threshold_calculation_details}")

        # --- Reset all session metrics ---
        self.score = 0
        self.difficulty_points = 0
        self.is_warning_active = False
        self.time_to_first_success_achieved = False
        self.current_streak_start_time = None
        self.longest_streak_so_far = 0.0
        self.threshold_crossing_count = 0
        self.was_successful_last_frame = False
        self.threshold_change_count = 0
        self.minute_performance_log = {}

        # Reset stars for rocket game
        self.stars = []
        for _ in range(50):
            self.stars.append((np.random.randint(0, 600), np.random.randint(0, 350)))

        # --- UI and Audio Setup ---
        exercise_type = getattr(session, 'exercise_type', 'Focus (Sunrise)')
        if exercise_type == "Audio Only (Eyes Closed)":
            self.feedback_canvas.pack_forget()
            self.status_label.config(text="Close your eyes and focus on the sound...")
            if self.audio_initialized:
                try:
                    pygame.mixer.music.load(resource_path("assets/melody.mp3"))
                    pygame.mixer.music.set_volume(0.2)
                    pygame.mixer.music.play(loops=-1)
                except Exception as e:
                    log.error(f"Could not load or play melody.mp3: {e}")
        else:
            self.feedback_canvas.pack()
            status_text = "Maintain a high level of concentration!" if self.session.session_type == "attention" else "Relax and observe..."
            self.status_label.config(text=status_text)

        mode_text = "Focus" if session.session_type == "attention" else "Relaxation"
        self.title_label.config(text=f"{mode_text} Training")
        self.score_label.config(text="Score: 0")

        # --- Start the Stream ---
        self.session.start_time = datetime.now()
        if not self.controller.device.is_running:
            self.controller.device.start_stream()

        self.update_job = self.after(50, self.update_feedback)

    def update_metrics_on_the_fly(self, is_success):
        """Calculates live metrics during the session."""
        if is_success and not self.time_to_first_success_achieved:
            self.session.time_to_first_success = (datetime.now() - self.session.start_time).total_seconds()
            self.time_to_first_success_achieved = True

        if is_success:
            if self.current_streak_start_time is None: self.current_streak_start_time = datetime.now()
        else:
            if self.current_streak_start_time is not None:
                streak_duration = (datetime.now() - self.current_streak_start_time).total_seconds()
                if streak_duration > self.longest_streak_so_far:
                    self.longest_streak_so_far = streak_duration
                self.current_streak_start_time = None

        if is_success and not self.was_successful_last_frame:
            self.threshold_crossing_count += 1
        self.was_successful_last_frame = is_success

        elapsed_seconds = (datetime.now() - self.session.start_time).total_seconds()
        current_minute = int(elapsed_seconds // 60)

        if current_minute not in self.minute_performance_log:
            self.minute_performance_log[current_minute] = []
        self.minute_performance_log[current_minute].append(self.smoothed_z_score)

    def update_feedback(self):
        """The main feedback loop using BrainFlow and Z-score normalization."""
        device = self.controller.device
        primary_idx = device.get_primary_eeg_channel_index()
        if not (device and device.is_connected()):
            self.stop_session()
            return

        sampling_rate = device.get_sampling_rate()
        samples_per_update = sampling_rate * self.PROCESSING_WINDOW_SECONDS


        new_data = device.get_data()
        if new_data is not None and new_data.shape[1] > 0:
            if self.raw_data_buffer.shape[1] == 0:
                self.raw_data_buffer = new_data
            else:
                self.raw_data_buffer = np.concatenate((self.raw_data_buffer, new_data), axis=1)

        if self.raw_data_buffer.shape[1] < samples_per_update:
            self.update_job = self.after(50, self.update_feedback)
            return

        data_to_process = self.raw_data_buffer[:, -samples_per_update:]

        max_buffer_samples = sampling_rate * 5
        if self.raw_data_buffer.shape[1] > max_buffer_samples:
            self.raw_data_buffer = self.raw_data_buffer[:, -max_buffer_samples:]

        indices_to_process = {'fp1': primary_idx}
        powers = process_brainflow_eeg(
            data_to_process,
            device.get_eeg_channels(),
            sampling_rate,
            indices_to_process
        )

        if not powers:
            self.update_job = self.after(50, self.update_feedback)
            return

        self.last_processed_powers = powers

        session_type = self.session.session_type

        metric_to_normalize = 'proc_attention' if session_type == 'attention' else 'proc_meditation'
        current_value = powers.get(metric_to_normalize, 0.0)

        baseline = self.session.baseline_stats[session_type]
        mean, std = baseline['mean'], baseline['std']
        MIN_RELIABLE_STD = 1e-4

        if std < MIN_RELIABLE_STD:
            current_z_score = 0.0
        else:
            current_z_score = (current_value - mean) / std

        smoothing_factor = 0.05
        self.smoothed_z_score = (smoothing_factor * current_z_score) + ((1 - smoothing_factor) * self.smoothed_z_score)
        is_success = self.smoothed_z_score > self.threshold

        ui_value = np.interp(self.smoothed_z_score, [-1.5, 2.5], [0, 100])
        ui_value = np.clip(ui_value, 0, 100)
        ui_threshold = np.interp(self.threshold, [-1.5, 2.5], [0, 100])

        self.update_metrics_on_the_fly(is_success)

        data_to_log = {**powers, **powers.get('bands', {}), 'z_score': self.smoothed_z_score, session_type: ui_value}
        self.session.add_data_point(data_to_log, ui_threshold)

        if is_success: self.score += 10
        self.update_difficulty(is_success)

        exercise_type = getattr(self.session, 'exercise_type', 'Focus (Sunrise)')
        if exercise_type != "Audio Only (Eyes Closed)":
            self.feedback_canvas.delete("all")
            if exercise_type == "Focus (Sunrise)":
                self.draw_focus_sunrise(ui_value, is_success)
            elif exercise_type == "Focus (Rocket Game)":
                self.draw_rocket_game(ui_value, is_success)
            elif exercise_type == "Focus (Circle)":
                target_radius = 20 + (ui_value / 100) * 120
                self.current_radius += (target_radius - self.current_radius) * 0.1
                self.draw_focus_circle(self.current_radius, is_success, ui_threshold)
            elif exercise_type == "Zen Garden":
                self.draw_zen_garden(ui_value, is_success)

        self.score_label.config(text=f"Score: {self.score}")
        self.difficulty_label.config(text=f"Difficulty (Z-Target): {self.threshold:.2f}")

        elapsed_time = (datetime.now() - self.session.start_time).total_seconds()
        remaining_time = max(0, self.session.duration_seconds - elapsed_time)
        self.timer_label.config(text=f"Time Left: {int(remaining_time)} s")
        if remaining_time <= 0:
            self.stop_session()
            return

        self.update_job = self.after(50, self.update_feedback)

    def draw_focus_sunrise(self, attention_level, is_success):
        """Draws the 'Focus Sunrise' exercise visualization."""
        canvas = self.feedback_canvas
        width, height = 600, 350

        def lerp_color(c1, c2, t):
            return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

        t = np.clip(attention_level / 100.0, 0, 1)
        night_sky, day_sky = (15, 23, 42), (135, 206, 235)
        sky_color_rgb = lerp_color(night_sky, day_sky, t)
        sky_color_hex = f'#{sky_color_rgb[0]:02x}{sky_color_rgb[1]:02x}{sky_color_rgb[2]:02x}'
        canvas.create_rectangle(0, 0, width, height, fill=sky_color_hex, outline="")

        canvas.create_arc(width * -0.5, height * 0.85, width * 1.5, height * 1.5,
                          start=0, extent=180, fill="#1A2E05", outline="", style=tk.CHORD)

        sun_y = height * 0.9 - (height * 0.8 * t)
        sun_x = width * 0.2 + (width * 0.6 * t)

        self.pulse_angle += (0.1 + t * 0.3)
        sun_radius = 40 * (1.0 + math.sin(self.pulse_angle) * 0.05 * t)

        glow_color_rgb = lerp_color((0, 0, 0), (255, 165, 0), t)
        glow_color_hex = f'#{glow_color_rgb[0]:02x}{glow_color_rgb[1]:02x}{glow_color_rgb[2]:02x}'
        canvas.create_oval(sun_x - sun_radius * 1.5, sun_y - sun_radius * 1.5,
                           sun_x + sun_radius * 1.5, sun_y + sun_radius * 1.5,
                           fill=glow_color_hex, outline="")

        canvas.create_oval(sun_x - sun_radius, sun_y - sun_radius,
                           sun_x + sun_radius, sun_y + sun_radius,
                           fill="#FFD700", outline="")

    def draw_rocket_game(self, attention_level, is_success):
        """Draws the Rocket Game visualization."""
        canvas = self.feedback_canvas
        width, height = 600, 350

        star_speed = 1 + (attention_level / 100) * 8
        for i, (x, y) in enumerate(self.stars):
            new_y = y + star_speed
            if new_y > height:
                new_y, x = 0, np.random.randint(0, width)
            self.stars[i] = (x, new_y)
            size = 1 if i % 3 == 0 else 2
            canvas.create_oval(x, new_y, x + size, new_y + size, fill="white", outline="")

        rocket_y = np.clip(height - (attention_level / 100) * height, 30, height - 30)
        rocket_x = width / 2

        canvas.create_polygon(rocket_x, rocket_y - 30, rocket_x - 15, rocket_y + 15, rocket_x + 15, rocket_y + 15,
                              fill="#E0E0E0", outline="black")
        canvas.create_oval(rocket_x - 7, rocket_y - 15, rocket_x + 7, rocket_y, fill="#87CEEB")

        flame_length = 5 + (attention_level / 100) * 40
        flame_color = "#FF4500" if is_success else "#FFD700"
        flicker = flame_length + np.random.randint(-5, 5)
        canvas.create_polygon(rocket_x, rocket_y + 15, rocket_x - 10, rocket_y + 15 + flicker, rocket_x + 10,
                              rocket_y + 15 + flicker, fill=flame_color)

    def draw_focus_circle(self, radius, is_success, ui_threshold):
        """Draws the expanding/contracting circle visualization."""
        style = self.master.master.style
        s_color, w_color, i_color = style.colors.success, style.colors.warning, style.colors.info
        color = s_color if is_success else w_color

        threshold_radius = 20 + (ui_threshold / 100) * 120
        self.feedback_canvas.create_oval(300 - threshold_radius, 175 - threshold_radius, 300 + threshold_radius,
                                         175 + threshold_radius, outline=i_color, dash=(4, 4), width=2)

        self.feedback_canvas.create_oval(300 - radius, 175 - radius, 300 + radius, 175 + radius, fill=color,
                                         outline=s_color, width=2)

    def draw_zen_garden(self, meditation_level, is_success):
        """Draws the Zen Garden visualization."""
        canvas, width, height = self.feedback_canvas, 600, 350

        def lerp_color(c1, c2, t):
            return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

        calm_c, disturb_c = (135, 206, 250), (70, 130, 180)
        t_color = meditation_level / 100.0
        water_rgb = lerp_color(disturb_c, calm_c, t_color)
        water_hex = f'#{water_rgb[0]:02x}{water_rgb[1]:02x}{water_rgb[2]:02x}'
        canvas.create_rectangle(0, 0, width, height, fill=water_hex, outline="")

        bloom_factor = np.clip((meditation_level - 10) / 70, 0, 1)
        center_x, center_y = width / 2, height / 2

        closed_p, open_p = (221, 160, 221), (255, 255, 255)
        petal_rgb = lerp_color(closed_p, open_p, bloom_factor)
        petal_hex = f'#{petal_rgb[0]:02x}{petal_rgb[1]:02x}{petal_rgb[2]:02x}'

        for i in range(8):
            angle = i * (360 / 8) + 45
            p_len = 30 + 50 * bloom_factor
            p_width = 10 + 20 * bloom_factor
            rad_angle = math.radians(angle)
            p_end_x = center_x + math.cos(rad_angle) * p_len
            p_end_y = center_y + math.sin(rad_angle) * p_len
            c1_x = center_x + math.cos(rad_angle - 0.5) * p_width
            c1_y = center_y + math.sin(rad_angle - 0.5) * p_width
            c2_x = center_x + math.cos(rad_angle + 0.5) * p_width
            c2_y = center_y + math.sin(rad_angle + 0.5) * p_width
            canvas.create_polygon(center_x, center_y, c1_x, c1_y, p_end_x, p_end_y, c2_x, c2_y,
                                  fill=petal_hex, outline="#F0F8FF", smooth=True)

        if is_success:
            glow_factor = (meditation_level - self.threshold) / (100 - self.threshold) if (
                                                                                                      100 - self.threshold) > 0 else 1
            glow_radius = int(5 + 15 * glow_factor)
            canvas.create_oval(center_x - glow_radius, center_y - glow_radius,
                               center_x + glow_radius, center_y + glow_radius,
                               fill="#FFFFE0", outline="")

    def stop_session(self):
        """Stops the session, finalizes metrics, and transitions."""
        if self.update_job:
            self.after_cancel(self.update_job)
            self.update_job = None

        if self.audio_initialized:
            pygame.mixer.music.stop()

        if self.current_streak_start_time is not None:
            streak_duration = (datetime.now() - self.current_streak_start_time).total_seconds()
            if streak_duration > self.longest_streak_so_far:
                self.longest_streak_so_far = streak_duration

        self.session.final_threshold = self.threshold
        self.session.score = self.score
        self.session.longest_streak = self.longest_streak_so_far
        self.session.threshold_crossings = self.threshold_crossing_count
        self.session.threshold_changes = self.threshold_change_count

        for minute, values in self.minute_performance_log.items():
            self.session.minute_by_minute_performance[minute] = np.mean(values)

        self.status_label.config(text="Session complete! Saving results...")
        if self.controller.device:
            self.controller.device.stop_stream()

        if self.session.data_log:
            df = pd.DataFrame(self.session.data_log)
            self.session.time_in_success_zone = (df[self.session.session_type] > df['threshold']).mean() * 100
        else:
            self.session.time_in_success_zone = 0.0

        self.after(1000, lambda: self.controller.process_finished_session(self.session))

    def update_difficulty(self, is_success):
        """
        MODIFIED: Adjusts the Z-score threshold with a time-based buffer.
        Difficulty now increases only after maintaining success for a set duration,
        but still decreases quickly to help the user recover.
        """
        if is_success:
            if self.success_streak_start_time is None:
                self.success_streak_start_time = datetime.now()

            streak_duration = (datetime.now() - self.success_streak_start_time).total_seconds()
            if streak_duration >= self.TIME_TO_INCREASE_DIFFICULTY:
                self.threshold = min(self.MAX_THRESHOLD_Z_SCORE, self.threshold + 0.05)
                self.threshold_change_count += 1
                log.info(
                    f"Difficulty INCREASED to Z-Target > {self.threshold:.2f} after {streak_duration:.1f}s of success.")
                self.success_streak_start_time = None

        else:
            self.success_streak_start_time = None

            self.difficulty_points -= 1
            if self.difficulty_points <= -10:
                self.threshold = max(self.MIN_THRESHOLD_Z_SCORE, self.threshold - 0.05)
                self.threshold_change_count += 1
                self.difficulty_points = 0
                log.info(f"Difficulty DECREASED to Z-Target > {self.threshold:.2f}")


class CheckInFrame(ttk.Frame):
    """
    Frame for the 60-second mind state snapshot.
    MODIFIED: Updated to wait for a stable signal before starting.
    """
    CHECK_IN_SECONDS = 60

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="Mind State Check-in", font=("Helvetica", 24, "bold"), bootstyle=SUCCESS).pack(pady=20)

        self.signal_wait_label = ttk.Label(self, text="", font=("Helvetica", 14, "bold"), bootstyle=INFO)
        self.signal_wait_label.pack(pady=5)
        self.signal_warning_label = ttk.Label(self, text="", font=("Helvetica", 12, "bold"), bootstyle=DANGER)
        self.signal_warning_label.pack(pady=5)
        self.status_label = ttk.Label(self,
                                      text="Please relax with your eyes open for 60 seconds.\nWe are measuring your current brain state.",
                                      font=("Helvetica", 14), justify="center")
        self.status_label.pack(pady=10)

        self.progress = ttk.Progressbar(self, length=400, mode='determinate', bootstyle="success-striped")
        self.progress.pack(pady=20)

        self.timer_label = ttk.Label(self, text="", font=("Helvetica", 16))
        self.timer_label.pack()

    def on_show(self, **kwargs):
        """Initializes and starts the check-in process."""
        self.raw_data_buffer = np.array([[] for _ in range(300)])
        self.snapshot_data = []
        self.progress['value'] = 0
        self.timer_label.config(text=f"{self.CHECK_IN_SECONDS} s")
        self.is_warning_active = False

        if not (self.controller.device and self.controller.device.is_running):
            self.controller.device.start_stream()

        self.signal_wait_label.config(text="Waiting for a stable signal...")
        self.wait_start_time = datetime.now()
        self.after(500, self.wait_for_signal)

    def wait_for_signal(self):
        """Checks for a stable signal before starting the timer."""
        if (datetime.now() - self.wait_start_time).total_seconds() > 20:
            messagebox.showerror("Signal Error", "Could not get a stable signal. Please check connections.")
            self.controller.device.stop_stream()
            self.controller.show_frame('WelcomeFrame')
            return

        if self.controller.device.check_signal_quality():
            log.info("Signal stable for Check-in. Starting.")
            self.signal_wait_label.config(text="")
            self.start_time = datetime.now()
            self.update_check_in()
        else:
            self.after(500, self.wait_for_signal)

    def update_check_in(self):
        """Gathers data with a CORRECTED immediate pause/resume for the timer UI."""
        is_signal_good = self.controller.device.check_signal_quality()

        if not is_signal_good:
            if not self.is_warning_active:
                log.warning("Signal lost. Pausing Check-in.")
                self.signal_warning_label.config(text="POOR SIGNAL QUALITY - CHECK CONTACT")
                self.is_warning_active = True
        else:
            if self.is_warning_active:
                log.info("✅ Signal restored. Resuming Check-in.")
                self.signal_warning_label.config(text="")
                self.is_warning_active = False

            device = self.controller.device
            frontal_indices = device.get_frontal_channel_indices()
            sampling_rate = device.get_sampling_rate()
            new_data = device.get_data()
            if new_data is not None and new_data.shape[1] > 0:
                if self.raw_data_buffer.shape[1] == 0:
                    self.raw_data_buffer = new_data
                else:
                    self.raw_data_buffer = np.concatenate((self.raw_data_buffer, new_data), axis=1)
            window_size = 2 * sampling_rate
            step_size = sampling_rate // 4
            while self.raw_data_buffer.shape[1] >= window_size:
                window_chunk = self.raw_data_buffer[:, :window_size]
                powers = process_brainflow_eeg(window_chunk, device.get_eeg_channels(), sampling_rate, frontal_indices)
                if powers:
                    bands_data = powers.get('bands', {})
                    data_point = {'raw_meditation': powers.get('raw_meditation', 0.0),
                                  'Rel_Beta': bands_data.get('Rel_Beta', 0.0),
                                  'Theta_Beta_Ratio': bands_data.get('Theta_Beta_Ratio', 0.0),
                                  'Alpha_Fp1': powers.get('Alpha_Fp1', 0.0), 'Alpha_Fp2': powers.get('Alpha_Fp2', 0.0)}
                    self.snapshot_data.append(data_point)
                self.raw_data_buffer = self.raw_data_buffer[:, step_size:]

        elapsed = (datetime.now() - self.start_time).total_seconds()

        if self.is_warning_active:
            self.start_time = datetime.now() - timedelta(seconds=elapsed)
        else:
            progress_value = (elapsed / self.CHECK_IN_SECONDS) * 100
            self.progress['value'] = progress_value
            remaining = max(0, self.CHECK_IN_SECONDS - elapsed)
            self.timer_label.config(text=f"{int(remaining)} s")

            if remaining <= 0:
                self.finish_check_in()
                return

        self.after(200, self.update_check_in)

    def finish_check_in(self):
        """Finishes the check-in and passes the data to the controller for analysis."""
        self.status_label.config(text="Analysis in progress...")
        if self.controller.device:
            self.controller.device.stop_stream()
        if len(self.snapshot_data) < 20:
            messagebox.showwarning("Check-in Warning",
                                   "Could not collect enough quality data for an accurate analysis.")
            self.controller.show_frame('WelcomeFrame')
            return
        self.controller.process_check_in_results(self.snapshot_data)


class RecommendationFrame(ttk.Frame):
    """Displays the session recommendation and AI Coach insight."""

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller
        self.recommended_session_args = {}

        ttk.Label(self, text="Your Daily Recommendation", font=("Helvetica", 24, "bold"), bootstyle=PRIMARY).pack(
            pady=20)

        # --- In-App Session Recommendation ---
        rec_frame = ttk.Labelframe(self, text="Recommended Session for Today", bootstyle=INFO, padding=15)
        rec_frame.pack(pady=10, padx=20, fill="x")

        self.recommendation_label = ttk.Label(rec_frame, text="Calculating...", font=("Helvetica", 14, "italic"),
                                              justify="center")
        self.recommendation_label.pack(pady=10)

        self.start_button = ttk.Button(rec_frame, text="Start Recommended Session", bootstyle=SUCCESS,
                                       command=self.start_recommended_session)
        self.start_button.pack(pady=10, ipady=5)

        # --- LLM Lifestyle Insight ---
        llm_frame = ttk.Labelframe(self, text="Your AI Coach Insight", bootstyle=PRIMARY, padding=15)
        llm_frame.pack(pady=10, padx=20, fill="x")

        self.insight_label = ttk.Label(llm_frame, text="AI is generating your daily insight...", font=("Helvetica", 12),
                                       wraplength=750, justify="left")
        self.insight_label.pack(pady=10)

        ttk.Button(self, text="Back to Main Menu", bootstyle=(SECONDARY, OUTLINE),
                   command=lambda: controller.show_frame('WelcomeFrame')).pack(pady=20, side="bottom")

    def on_show(self, recommended_session_args: dict, llm_insight: str):
        """Receives the analysis results and updates the UI."""
        self.recommended_session_args = recommended_session_args

        # Format the recommendation text
        session_type = self.recommended_session_args.get('session_type', 'attention').capitalize()
        exercise_type = self.recommended_session_args.get('exercise_type', 'Focus (Sunrise)')
        rec_text = f"{session_type} Training\nExercise: {exercise_type}"
        self.recommendation_label.config(text=rec_text)

        # Display the LLM insight
        self.insight_label.config(text=llm_insight)

    def start_recommended_session(self):
        """Starts the session using the arguments provided by the controller."""
        if self.recommended_session_args:
            # We need to get the current profile and device from the WelcomeFrame, as they are not passed
            welcome_frame = self.controller.frames['WelcomeFrame']
            user_name = welcome_frame.profile_combo.get()
            device_name = welcome_frame.device_combo.get()

            # The duration can be a default value for recommended sessions
            duration_str = "8 minutes"

            # Update the arguments with the latest selections
            self.recommended_session_args['user_name'] = user_name
            self.recommended_session_args['device_name'] = device_name
            self.recommended_session_args['duration_str'] = duration_str

            self.controller.prepare_training_session(**self.recommended_session_args)


class ReportFrame(ttk.Frame):
    """
    Frame for displaying the post-session report, including an AI-generated summary,
    a progress graph, and numerical statistics.
    """

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller
        self.session: NeurofeedbackSession | None = None

        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=10, pady=(10, 5))
        header_frame.columnconfigure(0, weight=1)

        ttk.Label(header_frame, text="Session Report", font=("Helvetica", 24, "bold"), bootstyle=PRIMARY).grid(row=0,
                                                                                                               column=0,
                                                                                                               sticky='w')
        ttk.Button(
            header_frame,
            text="Back to Main Menu",
            bootstyle=(SECONDARY, OUTLINE),
            command=lambda: controller.show_frame('WelcomeFrame')
        ).grid(row=0, column=1, sticky='e', padx=10)

        content_frame = ScrolledFrame(self, autohide=True)
        content_frame.pack(fill='both', expand=True, padx=10, pady=5)

        llm_frame = ttk.Labelframe(content_frame, text="AI Coach Summary", bootstyle=INFO, padding=10)
        llm_frame.pack(fill='x', padx=10, pady=10)
        self.llm_summary_label = ttk.Label(llm_frame, text="Analyzing your session...",
                                           font=("Helvetica", 12, "italic"), wraplength=700, justify="left")
        self.llm_summary_label.pack(fill='x')

        self.canvas_frame = ttk.Frame(content_frame, bootstyle=SECONDARY, relief="sunken", borderwidth=2)
        self.canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.stats_label = ttk.Label(content_frame, text="", font=("Helvetica", 12), justify=tk.LEFT)
        self.stats_label.pack(pady=10, padx=10)

    def on_show(self, session: NeurofeedbackSession):
        """
        This method is called when the frame becomes visible.
        It orchestrates the report generation.
        """
        self.session = session

        # 1. Immediately display the parts that are fast (graph and stats)
        self.plot_session_graph()
        self.display_stats()

        # 2. Start the slow part (LLM analysis) in a background thread to keep the UI responsive
        self.llm_summary_label.config(text="AI is analyzing your session, this may take a moment...")
        analysis_thread = threading.Thread(target=self.run_llm_analysis, daemon=True)
        analysis_thread.start()

    def run_llm_analysis(self):
        """
        This function runs in a separate thread to call the LLM,
        then safely updates the UI and saves the summary to the user's profile.
        """
        try:
            # This is the blocking call that can take several seconds
            summary = self.controller.llm_analyzer.analyze_session(self.session)

            # Save the summary to the user's profile for the journal feature
            for session_entry in self.controller.profiles[self.session.user_name]["sessions"]:
                if session_entry.get("file") == self.session.csv_filename:
                    session_entry["report_text"] = summary
                    break

            save_profiles(self.controller.profiles)

            log.info("LLM summary saved to profile.")

            # Schedule the UI update to run on the main thread
            self.after(0, lambda: self.llm_summary_label.config(text=summary))

        except Exception as e:
            log.error(f"Error during LLM analysis thread: {e}", exc_info=True)
            self.after(0, lambda: self.llm_summary_label.config(text="Failed to generate AI summary."))

    def plot_session_graph(self):
        """Creates and displays the session progress graph."""
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        df = pd.DataFrame(self.session.data_log)
        if df.empty:
            return

        plt.style.use("dark_background")
        style = self.controller.style
        fig = plt.Figure(figsize=(7, 3.5), dpi=100, facecolor=style.colors.bg)
        ax = fig.add_subplot(111, facecolor=style.colors.inputbg)

        target_metric_name = self.session.session_type
        target_metric_label = "Attention (Focus)" if target_metric_name == "attention" else "Meditation (Relaxation)"

        ax.plot(df['timestamp'], df[target_metric_name], label=target_metric_label, color=style.colors.success,
                zorder=5)
        ax.plot(df['timestamp'], df['threshold'], label='Difficulty Threshold', color=style.colors.info, linestyle='--',
                zorder=4)
        ax.fill_between(df['timestamp'], df[target_metric_name], df['threshold'],
                        where=df[target_metric_name] >= df['threshold'],
                        facecolor=style.colors.success, alpha=0.3, interpolate=True, label='Success Zone')

        ax.set_title(f"Session Progress for {self.session.user_name}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Level (0-100)")
        ax.grid(True, linestyle='--', alpha=0.6)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_color(style.colors.fg)

        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color(style.colors.fg)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_session_progress(self, df: pd.DataFrame, session: NeurofeedbackSession):
        """
        Plots the session progress with the new "hills and valleys" visualization.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if df.empty or len(df) < 2:
            ax.text(0.5, 0.5, 'No data to display.', ha='center', va='center', color="white")
            self.canvas.draw()
            return

        session_type = session.session_type
        if session_type not in df.columns:
            ax.text(0.5, 0.5, f"Data for '{session_type}' not found.", ha='center', va='center', color="white")
            self.canvas.draw()
            return

        ax.fill_between(
            df.index,
            df[session_type],
            df['threshold'],
            where=(df[session_type] < df['threshold']),
            color='#E57373',
            alpha=0.5,
            interpolate=True,
            label='Focus Dips'
        )

        ax.fill_between(
            df.index,
            df['threshold'],
            df['threshold'] + 5,
            color='#81D4FA',
            alpha=0.4,
            label='Success Buffer'
        )

        ax.fill_between(
            df.index,
            df['threshold'],
            df[session_type],
            where=(df[session_type] > df['threshold']),
            color='#66BB6A',
            alpha=0.7,
            interpolate=True,
            label='Improvement Peaks'
        )

        ax.plot(df.index, df[session_type], label=f'{session_type.capitalize()}', color="#FFFFFF", linewidth=2)
        ax.plot(df.index, df['threshold'], label='Difficulty Threshold', linestyle='--', color="#FFB74D")

        ax.set_title(f"Session Progress for {session.user_name}", color="white")
        ax.set_xlabel("Time (seconds)", color="white")
        ax.set_ylabel("Level (0-100)", color="white")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_facecolor("#343a40")
        self.figure.patch.set_alpha(0.0)

        self.figure.tight_layout()
        self.canvas.draw()

    def display_stats(self):
        """Calculates and displays the final numerical statistics for the session."""
        df = pd.DataFrame(self.session.data_log)
        if df.empty:
            self.stats_label.config(text="No data recorded to display statistics.")
            return

        target_metric_name = self.session.session_type
        target_metric_label = "Attention" if target_metric_name == "attention" else "Meditation"

        avg_metric = df[target_metric_name].mean()
        time_in_zone = (df[target_metric_name] >= df['threshold']).sum() / len(df) * 100 if not df.empty else 0

        stats_text = (
            f"Total Score: {self.session.score}\n"
            f"Average '{target_metric_label}': {avg_metric:.1f}\n"
            f"Time in 'Success Zone': {time_in_zone:.1f}%\n\n"
            f"Longest Streak in Zone: {self.session.longest_streak:.1f} seconds\n"
            f"Entries into Zone: {self.session.threshold_crossings} times\n"
            f"Difficulty Adjustments: {self.session.threshold_changes} times\n"
        )

        if self.session.time_to_first_success is not None:
            stats_text += f"Time to First Success: {self.session.time_to_first_success:.1f} seconds"

        self.stats_label.config(text=stats_text)


class HistoryFrame(ttk.Frame):
    """
    Frame for viewing past sessions, calendar view of training days,
    a long-term progress graph, and the AI-generated journal.
    """

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller
        self.username = None
        self.displayed_date = date.today()
        self.training_dates = []

        # --- OVERALL LAYOUT ---
        main_panel = ttk.Frame(self)
        main_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        right_panel = ttk.Frame(self, width=350)
        right_panel.pack(side="right", fill="y", padx=10, pady=10, expand=False)
        right_panel.pack_propagate(False)

        # --- LEFT PANEL: LONG-TERM PROGRESS GRAPH ---
        ttk.Label(main_panel, text="Long-Term Progress (Time in Success Zone %)", font=("Helvetica", 14)).pack(pady=5)
        self.progress_canvas_frame = ttk.Frame(main_panel, bootstyle=SECONDARY, relief="sunken", borderwidth=2)
        self.progress_canvas_frame.pack(fill="both", expand=True)

        # --- RIGHT PANEL: CALENDAR, SESSION LIST, AND JOURNAL ---
        self.title_label = ttk.Label(right_panel, text="", font=("Helvetica", 18, "bold"), bootstyle=PRIMARY)
        self.title_label.pack(pady=10, anchor="w", padx=10)

        self.calendar_frame = ttk.Frame(right_panel, padding=5)
        self.calendar_frame.pack(pady=5)

        ttk.Label(right_panel, text="Session List:").pack(anchor="w", pady=(10, 2), padx=10)
        self.session_listbox = tk.Listbox(right_panel, height=8, background="#32383e", foreground="white",
                                          selectbackground=self.controller.style.colors.primary, relief="flat",
                                          borderwidth=0)
        self.session_listbox.pack(fill="x", expand=False, padx=10)
        self.session_listbox.bind("<<ListboxSelect>>", self.on_session_select)

        journal_frame = ttk.Labelframe(right_panel, text="AI Coach Journal", bootstyle=INFO, padding=10)
        journal_frame.pack(pady=10, padx=10, fill='both', expand=True)

        self.journal_text = ScrolledText(journal_frame, wrap="word", height=10, state="disabled", autohide=True)
        self.journal_text.pack(fill="both", expand=True)

        ttk.Button(right_panel, text="Back to Main Menu", bootstyle=(SECONDARY, OUTLINE),
                   command=lambda: controller.show_frame('WelcomeFrame')).pack(pady=10, side="bottom")

    def on_show(self, username: str):
        """Called when the frame is brought to the front. Loads and displays user data."""
        self.username = username
        self.training_dates = self.get_training_dates()
        streak = self.calculate_streak(self.training_dates)
        streak_text = f" (🔥 Streak: {streak} days)" if streak > 1 else ""
        self.title_label.config(text=f"History: {username}{streak_text}")

        self.displayed_date = date.today()
        self.update_calendar()
        self.update_session_list()
        self.plot_long_term_progress()
        self.update_journal_text("Select a session from the list to see the AI summary.")

    def get_filepath_from_entry(self, session_entry) -> str | None:
        """Helper function to safely get a filepath from either the old (str) or new (dict) format."""
        if isinstance(session_entry, dict):
            return session_entry.get("file")
        elif isinstance(session_entry, str):
            return session_entry  # Handle old format for backward compatibility
        return None

    def get_training_dates(self) -> List[date]:
        """Extracts unique training dates from the user's session files."""
        sessions = self.controller.profiles.get(self.username, {}).get("sessions", [])
        dates = set()
        for session_entry in sessions:
            filepath = self.get_filepath_from_entry(session_entry)
            if not filepath: continue
            try:
                date_str = os.path.basename(filepath).split('_')[2]
                dates.add(datetime.strptime(date_str, '%Y%m%d').date())
            except (IndexError, ValueError):
                continue
        return sorted(list(dates))

    def calculate_streak(self, dates: list[date]) -> int:
        """Calculates the current training streak in days."""
        if not dates:
            return 0

        today = date.today()
        streak = 0

        # Check if there was a session today or yesterday to start the streak count
        if today in dates:
            streak += 1
            check_date = today - timedelta(days=1)
        elif (today - timedelta(days=1)) in dates:
            streak += 1
            check_date = today - timedelta(days=2)
        else:
            return 0  # Streak is broken

        # Count backwards from there
        while check_date in dates:
            streak += 1
            check_date -= timedelta(days=1)
        return streak

    def update_calendar(self):
        """Renders the monthly calendar view, highlighting training days."""
        for widget in self.calendar_frame.winfo_children():
            widget.destroy()

        # Calendar navigation
        cal_nav_frame = ttk.Frame(self.calendar_frame)
        ttk.Button(cal_nav_frame, text="<", bootstyle=SECONDARY, command=lambda: self.change_month(-1)).pack(
            side="left")
        month_label = ttk.Label(cal_nav_frame, text=f"{self.displayed_date.strftime('%B %Y')}", width=20,
                                anchor="center")
        month_label.pack(side="left", expand=True)
        ttk.Button(cal_nav_frame, text=">", bootstyle=SECONDARY, command=lambda: self.change_month(1)).pack(
            side="right")
        cal_nav_frame.grid(row=0, column=0, columnspan=7, pady=(0, 5))

        # Day headers
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, day in enumerate(days):
            ttk.Label(self.calendar_frame, text=day, bootstyle=SECONDARY).grid(row=1, column=i)

        # Populate days
        cal = calendar.monthcalendar(self.displayed_date.year, self.displayed_date.month)
        for r, week in enumerate(cal):
            for c, day_num in enumerate(week):
                if day_num == 0:
                    continue
                day_date = date(self.displayed_date.year, self.displayed_date.month, day_num)

                style = "secondary"
                if day_date == date.today(): style = "info"  # Highlight today
                if day_date in self.training_dates: style = "success"  # Highlight training days

                day_label = ttk.Label(self.calendar_frame, text=str(day_num), bootstyle=style, anchor="center")
                day_label.grid(row=r + 2, column=c, sticky="nsew", ipadx=5, ipady=5)

    def change_month(self, delta: int):
        """Moves the calendar to the next or previous month."""
        year, month = self.displayed_date.year, self.displayed_date.month
        month += delta
        if month > 12:
            month, year = 1, year + 1
        if month < 1:
            month, year = 12, year - 1
        self.displayed_date = date(year, month, 1)
        self.update_calendar()

    def update_session_list(self):
        """Populates the listbox with the user's past sessions."""
        self.session_listbox.delete(0, tk.END)
        sessions = self.controller.profiles.get(self.username, {}).get("sessions", [])
        for session_entry in reversed(sessions):
            filepath = self.get_filepath_from_entry(session_entry)
            if not filepath: continue
            display_name = os.path.basename(filepath).replace(f"{self.username}_", "").replace(".csv", "")
            self.session_listbox.insert(tk.END, display_name)

    def on_session_select(self, event):
        """Displays journal entry and stats when a session is selected."""
        selected_indices = self.session_listbox.curselection()
        if not selected_indices: return

        sessions = self.controller.profiles.get(self.username, {}).get("sessions", [])
        selected_session_entry = list(reversed(sessions))[selected_indices[0]]

        # Get AI-generated report text from the profile data
        report_text = "No AI summary available for this session."
        if isinstance(selected_session_entry, dict):
            report_text = selected_session_entry.get("report_text", report_text)

        # Get numerical stats from the CSV file
        stats_text = ""
        filepath = self.get_filepath_from_entry(selected_session_entry)
        if filepath:
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    session_type = df.columns[1]  # 'attention' or 'meditation'
                    avg_metric = df[session_type].mean()
                    time_in_zone = (df[session_type] >= df['threshold']).sum() / len(df) * 100
                    stats_text = (f"--- Session Stats ---\n"
                                  f"Average Metric: {avg_metric:.1f}\n"
                                  f"Time in Zone: {time_in_zone:.1f}%\n"
                                  f"-----------------------\n\n")
            except Exception as e:
                stats_text = f"--- Could not load stats: {e} ---\n\n"

        self.update_journal_text(stats_text + report_text)

    def plot_long_term_progress(self):
        """Plots Time in Success Zone (%) instead of score for better comparison."""
        for widget in self.progress_canvas_frame.winfo_children(): widget.destroy()

        sessions = self.controller.profiles.get(self.username, {}).get("sessions", [])
        if len(sessions) < 2:
            ttk.Label(self.progress_canvas_frame, text="At least 2 sessions are needed for a progress graph.").pack(
                pady=50)
            return

        progress_data = []
        for session_entry in sessions:
            filepath = self.get_filepath_from_entry(session_entry)
            if not filepath: continue
            try:
                df = pd.read_csv(filepath)
                if df.empty or df.shape[1] < 2:
                    log.warning(f"Skipping empty or malformed session file for graph: {filepath}")
                    continue

                session_type = df.columns[1]
                time_in_zone = (df[session_type] >= df['threshold']).sum() / len(df) * 100

                parts = os.path.basename(filepath).split('_')
                session_date = datetime.strptime(parts[2], '%Y%m%d')
                progress_data.append({"date": session_date, "metric": time_in_zone, "type": session_type})
            except Exception as e:
                log.warning(f"Could not process session file for graph {filepath}: {e}")
                continue

        if not progress_data: return
        df_progress = pd.DataFrame(progress_data).sort_values(by="date")
        df_avg_progress = df_progress.groupby([df_progress['date'].dt.date, 'type']).metric.mean().reset_index()

        df_focus = df_avg_progress[df_avg_progress['type'] == 'attention']
        df_relax = df_avg_progress[df_avg_progress['type'] == 'meditation']

        plt.style.use("dark_background")
        style = self.controller.style
        fig = plt.Figure(figsize=(5, 4), dpi=100, facecolor=style.colors.bg)
        ax = fig.add_subplot(111, facecolor=style.colors.inputbg)

        if not df_focus.empty:
            ax.plot(df_focus['date'], df_focus['metric'], marker='o', linestyle='-', color=style.colors.success,
                    label="Focus")
        if not df_relax.empty:
            ax.plot(df_relax['date'], df_relax['metric'], marker='x', linestyle='--', color=style.colors.info,
                    label="Relaxation")

        if not df_focus.empty or not df_relax.empty:
            legend = ax.legend()
            plt.setp(legend.get_texts(), color=style.colors.fg)

        ax.set_ylabel("Time in Success Zone (%)")  # Updated Y-axis label
        ax.set_ylim(0, 100)  # Set Y-axis from 0 to 100 for percentage
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', colors=style.colors.fg, rotation=30)
        ax.tick_params(axis='y', colors=style.colors.fg)
        ax.yaxis.label.set_color(style.colors.fg)
        ax.xaxis.label.set_color(style.colors.fg)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.progress_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_journal_text(self, text: str):
        """Safely updates the content of the ScrolledText widget."""
        # To change the state, we need to configure the internal .text widget
        self.journal_text.text.config(state="normal")

        # .delete and .insert are correctly forwarded, so they can be called directly
        self.journal_text.delete('1.0', tk.END)
        self.journal_text.insert('1.0', text)

        # And we disable the internal .text widget again
        self.journal_text.text.config(state="disabled")


class PostSessionSurveyFrame(ttk.Frame):
    """A frame that appears after a session to collect subjective user feedback."""

    def __init__(self, parent, controller: 'NuroiaApp'):
        super().__init__(parent)
        self.controller = controller
        self.session = None

        ttk.Label(self, text="Session Feedback", font=("Helvetica", 24, "bold"), bootstyle=PRIMARY).pack(pady=20)
        ttk.Label(self, text="Your subjective feelings are a key part of progress.", font=("Helvetica", 12),
                  bootstyle=SECONDARY).pack(pady=(0, 20))

        # --- Feeling Scale ---
        feeling_frame = ttk.Frame(self)
        ttk.Label(feeling_frame, text="How do you feel now?", font=("Helvetica", 14)).pack()
        self.feeling_scale = ttk.Scale(feeling_frame, from_=1, to=5, length=300)
        self.feeling_scale.set(3)  # Default to neutral
        self.feeling_scale.pack(pady=5)
        labels_frame1 = ttk.Frame(feeling_frame)
        ttk.Label(labels_frame1, text="Stressed/Tired", bootstyle=DANGER).pack(side="left")
        ttk.Label(labels_frame1, text="Relaxed/Energized", bootstyle=SUCCESS).pack(side="right")
        labels_frame1.pack(fill="x", expand=True, padx=45)
        feeling_frame.pack(pady=10)

        # --- Focus Scale ---
        focus_frame = ttk.Frame(self)
        ttk.Label(focus_frame, text="How focused were you during the session?", font=("Helvetica", 14)).pack()
        self.focus_scale = ttk.Scale(focus_frame, from_=1, to=5, length=300)
        self.focus_scale.set(3)  # Default to neutral
        self.focus_scale.pack(pady=5)
        labels_frame2 = ttk.Frame(focus_frame)
        ttk.Label(labels_frame2, text="Very Distracted", bootstyle=DANGER).pack(side="left")
        ttk.Label(labels_frame2, text="In the Zone", bootstyle=SUCCESS).pack(side="right")
        labels_frame2.pack(fill="x", expand=True, padx=50)
        focus_frame.pack(pady=10)

        # --- Notes ---
        notes_frame = ttk.Frame(self)
        ttk.Label(notes_frame, text="Any notes? (e.g., drank coffee, was sleepy)", font=("Helvetica", 14)).pack()
        self.notes_entry = ttk.Entry(notes_frame, width=50)
        self.notes_entry.pack(pady=5)
        notes_frame.pack(pady=10)

        # --- Submit Button ---
        ttk.Button(self, text="Continue to Report", bootstyle=SUCCESS, command=self.submit_feedback).pack(pady=30,
                                                                                                          ipady=10)

    def on_show(self, session):
        """Called when the frame is shown. Resets the widgets."""
        self.session = session
        self.feeling_scale.set(3)
        self.focus_scale.set(3)
        self.notes_entry.delete(0, tk.END)

    def submit_feedback(self):
        """Saves the feedback to the session object and proceeds to the report."""
        if self.session:
            self.session.user_feeling = int(self.feeling_scale.get())
            self.session.user_focus_rating = int(self.focus_scale.get())
            self.session.user_notes = self.notes_entry.get()

        # Now, show the report frame with the enriched session object
        self.controller.show_frame('ReportFrame', session=self.session)