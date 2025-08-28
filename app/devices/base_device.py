# app/devices/base_device.py
import queue
from abc import ABC, abstractmethod

class BaseDevice(ABC):
    """
    Abstract Base Class for all neurofeedback devices.
    Defines the common interface that all device wrappers must implement.
    """
    def __init__(self):
        self.is_running = False
        self.data_queue = queue.Queue()

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish a connection to the device.
        Returns True on success, False on failure.
        """
        pass

    @abstractmethod
    def start_stream(self) -> bool:
        """
        Start the data stream from the device.
        Returns True on success, False on failure.
        """
        pass

    @abstractmethod
    def check_signal_quality(self) -> bool:
        """
        Checks the quality of the signal or electrode contact.
        This is a mandatory method for all devices.
        Returns True if the signal is good, False otherwise.
        """
        pass

    @abstractmethod
    def stop_stream(self):
        """
        Stop the data stream.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Disconnect from the device and release all resources.
        """
        pass

    @abstractmethod
    def is_connected(self):
        pass
