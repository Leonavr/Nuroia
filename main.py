# main.py
import logging
import sys
from datetime import datetime

from app.ui.main_window import NuroiaApp

log_filename = f"nuroia_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("Application starting up...")

if __name__ == "__main__":
    """
    Main entry point for the Nuroia application.
    Initializes and runs the main application window.
    """
    log = logging.getLogger(__name__)
    log.info("Starting Nuroia application...")

    app = NuroiaApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

    log.info("Nuroia application finished.")