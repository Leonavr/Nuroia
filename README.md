# Nuroia

Nuroia is a personal neurofeedback companion designed to help users improve focus and relaxation. The application interfaces with multiple consumer and professional EEG devices to provide real-time, AI-enhanced feedback on the user's mental state.

## Features

-   **Multi-Device Compatibility:** Natively supports both the professional **Neurobit Optima+** (via official SDK) and the popular consumer **Muse 2 / Muse S** headbands (via direct Bluetooth LE connection). The architecture is extensible for adding more devices.
-   **Advanced Real-time EEG Processing:** Utilizes the scientific-grade `yasa` library to calculate real-time brainwave band powers (Delta, Theta, Alpha, Beta, Gamma). These are used to derive key metrics:
    -   **Focus:** Based on the Theta/Beta Ratio (TBR).
    -   **Relaxation:** Based on relative Alpha power.
    -   **Emotional Valence:** Capability for Frontal Alpha Asymmetry analysis.
-   **Personalized Calibration & Training:**
    -   **Dynamic Difficulty:** The challenge threshold in training exercises automatically adapts to your performance.
    -   **One-Time Focus Calibration:** A unique two-step process measures your personal range of focus and distraction (TBR range) for more accurate daily assessments.
    -   **Daily "Mind State Check-in":** A 60-second daily snapshot of your brain activity to provide personalized recommendations.
-   **AI-Powered Insights (Google Gemini):**
    -   **AI Coach:** Provides detailed, empathetic analysis and summaries after each training session.
    -   **Daily Lifestyle Tips:** Generates a short, actionable insight for your day based on your Mind State Check-in.
-   **Engaging Feedback & Progress Tracking:**
    -   Multiple visual exercises: "Focus Sunrise," "Rocket Game," and "Zen Garden."
    -   Auditory feedback mode for eyes-closed training.
    -   Saves all sessions with comprehensive reports, graphs, and statistics to track long-term progress.

## Project Structure

The project is organized into a modular structure for maintainability and scalability.

nuroia/
│
├── app/                  # Main application source code
│   ├── analysis/         # AI analysis logic (LLM communication)
│   ├── core/             # Core logic (sessions, profiles)
│   ├── devices/          # Device communication and abstraction
│   ├── processing/       # EEG signal processing logic
│   └── ui/               # GUI components (main window, frames)
│
├── assets/               # Audio and image files
├── sessions/             # Saved user session data (CSVs)
│
├── .env                  # Environment variables (for API keys)
├── main.py               # Main entry point to run the application
├── NeurobitDrv64.dll     # Required Neurobit device driver library
├── profiles.json         # User profiles
├── requirements.txt      # Python package dependencies
└── README.md             # This documentation file


## Installation and Setup

### Prerequisites

-   Python 3.9+
-   An active internet connection for AI analysis.
-   For Muse: A computer with Bluetooth support.
-   For Neurobit Optima: [FTDI D2XX Drivers](https://ftdichip.com/drivers/d2xx-drivers/) installed.

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Leonavr/Neuro](https://github.com/Leonavr/Neuro)
    cd Neuro
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up AI Coach API Key:**
    -   Create a file named `.env` in the root directory of the project.
    -   Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    -   Add the following line to the `.env` file:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```

5.  **Hardware-Specific Setup:**
    -   **For Neurobit Optima:** Ensure `NeurobitDrv64.dll` is placed in the root directory.
    -   **For Muse 2 / S:** Ensure your device is turned on and paired with your computer via Bluetooth.

## Core Modules Explained
### Device Abstraction (app/devices/)
The application is designed to be hardware-agnostic.

base_device.py: Defines the BaseDevice interface with methods like connect(), start_stream(), and is_connected().

neurobit_device.py: Implementation for the Neurobit Optima.

muse_device.py: Implementation for Muse devices using the bleak Bluetooth LE library.

To add a new device:

Create a new_device.py in app/devices/.

Create a class that inherits from BaseDevice and implement its methods.

Register the new device in the self.available_devices dictionary in app/ui/main_window.py.

### Signal Processing (app/processing/)
The eeg_processor.py module is the analytical core of Nuroia. It has been refactored to use the BrainFlow library's DataFilter methods for all real-time calculations, ensuring a consistent and scientific-grade pipeline for all compatible devices.

Core Neurofeedback Principles
The application's feedback is grounded in established neuroscientific principles to ensure efficacy and accuracy.

Personalized Z-Score Normalization: Instead of relying on generic thresholds, Nuroia uses Z-Score normalization. After a baseline calibration, the application calculates your personal mean and standard deviation for specific brain states. During training, your real-time performance is converted into a Z-score, which represents how many standard deviations you are above or below your own average. This makes the feedback loop inherently personalized, adaptive, and scientifically grounded.

Advanced Metric Derivation:

Relaxation (Alpha/Beta Ratio): To provide a robust measure of relaxation, the app uses the ratio of Alpha power (8-12 Hz) to Beta power (13-30 Hz). A higher ratio indicates a state of calm wakefulness. This metric is superior to using Alpha power alone, as it is more resilient to contamination from muscle tension artifacts (EMG), which can otherwise create a "paradox" where tension is rewarded.

Focus (Theta/Beta Ratio - TBR): The application employs the widely validated TBR as its core metric for focus and attention. A lower ratio, indicating suppressed Theta activity and heightened Beta activity, is correlated with stronger concentration.

Emotional Valence (Frontal Alpha Asymmetry - FAA): During the "Mind State Check-in," the app calculates FAA by comparing Alpha power between the left (Fp1/AF7) and right (Fp2/AF8) prefrontal cortex. This metric is linked to emotional states, where greater relative left-hemispheric activity is often associated with positive, "approach-oriented" emotions, while greater right-hemispheric activity is linked to "withdrawal-oriented" states.

Real-time Data Pipeline
The signal processing follows a standardized pipeline for each segment of EEG data:

Data Acquisition: Raw EEG data is collected in overlapping time-series windows (typically 4 seconds).

Signal Cleaning:

A Band-Stop (Notch) Filter removes electrical grid noise (50/60 Hz).

A Band-Pass Filter (1-50 Hz) isolates the primary EEG frequencies, removing DC offset and high-frequency noise.

Spectral Analysis: The power spectral density (PSD) of the cleaned signal is calculated using Welch's method, which provides a stable estimate of signal power at different frequencies.

Band Power Calculation: From the PSD, the absolute power (in µV²/Hz) is calculated for each of the core neurophysiological frequency bands (e.g., Delta, Theta, Alpha, Beta).

Metric Computation: The final, user-facing metrics (TBR, Alpha/Beta Ratio, FAA, and Z-scores) are derived from these calculated band powers.

### AI Analysis (app/analysis/)
The llm_analyzer.py module handles all communication with the Google Gemini API.

It takes structured session data (both objective metrics and subjective user feedback) as input.

It uses carefully crafted prompts to instruct the LLM to act as an empathetic neuro-coach.

It generates both detailed post-session reports and concise daily insights.

## How to Run

Execute the `main.py` script from the root directory:

```bash
python main.py