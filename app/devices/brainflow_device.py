# app/devices/brainflow_device.py
import logging
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError, BoardIds
from app.devices.base_device import BaseDevice

log = logging.getLogger(__name__)


class BrainflowDevice(BaseDevice):
    """
    A device class for hardware supported by the BrainFlow library.
    """
    PRIMARY_CHANNEL_MAP = {
        BoardIds.MUSE_2_BOARD.value: 1,  # For Muse 2/S, AF7 is the channel with index 1
        BoardIds.MUSE_S_BOARD.value: 1,
    }

    FRONTAL_CHANNEL_MAP = {
        BoardIds.MUSE_2_BOARD.value: {'fp1': 1, 'fp2': 2},  # AF7 is index 1, AF8 is index 2
        BoardIds.MUSE_S_BOARD.value: {'fp1': 1, 'fp2': 2},
    }

    def __init__(self, board_id):
        super().__init__()
        self.board_id = board_id
        params = BrainFlowInputParams()
        self.board = BoardShim(self.board_id, params)
        self.is_running = False
        # The device_name will be set by the controller after instantiation
        self.device_name = "BrainFlow Device"

    def connect(self) -> bool:
        if self.board.is_prepared():
            return True
        try:
            self.board.prepare_session()
            return True
        except BrainFlowError as e:
            log.error(f"BrainFlow error preparing session: {e}", exc_info=True)
            return False

    def start_stream(self) -> bool:
        if not self.is_connected(): return False
        if self.is_running: return True
        try:
            self.board.start_stream()
            self.is_running = True
            return True
        except BrainFlowError as e:
            log.error(f"BrainFlow error starting stream: {e}", exc_info=True)
            return False

    def get_data(self) -> np.ndarray | None:
        if not self.is_running: return None
        try:
            # Get all data and clear the internal buffer
            return self.board.get_board_data()
        except BrainFlowError as e:
            log.error(f"BrainFlow error getting data: {e}", exc_info=True)
            return None

    def stop_stream(self):
        if not self.is_running: return
        try:
            self.board.stop_stream()
            self.is_running = False
        except BrainFlowError as e:
            log.error(f"BrainFlow error stopping stream: {e}", exc_info=True)

    def disconnect(self):
        if self.is_running:
            self.stop_stream()
        if self.is_connected():
            self.board.release_session()

    def is_connected(self) -> bool:
        return self.board.is_prepared()

    def get_sampling_rate(self) -> int:
        return BoardShim.get_sampling_rate(self.board_id)

    def get_eeg_channels(self) -> list:
        return BoardShim.get_eeg_channels(self.board_id)

    def check_signal_quality(self) -> bool:
        return self.is_running

    def get_primary_eeg_channel_index(self) -> int:
        """Returns the index of the primary EEG channel for this board."""
        # This method now correctly references the PRIMARY_CHANNEL_MAP
        return self.PRIMARY_CHANNEL_MAP.get(self.board_id, 0)

    def get_frontal_channel_indices(self) -> dict | None:
        """Returns a dictionary with indices for left (fp1) and right (fp2) frontal channels."""
        return self.FRONTAL_CHANNEL_MAP.get(self.board_id)