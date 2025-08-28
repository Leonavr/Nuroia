# app/devices/neurobit_device.py

import ctypes
import os
import threading
import time
import numpy as np
import logging
from tkinter import messagebox

from app.devices.base_device import BaseDevice

log = logging.getLogger(__name__)

# --- CTYPES DEFINITIONS FOR NEUROBIT API ---
DLL_NAME = "NeurobitDrv64.dll"
try:
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_path, '..', '..'))
    dll_path = os.path.join(project_root, DLL_NAME)
    if os.path.exists(dll_path):
        os.add_dll_directory(project_root)
        lib = ctypes.WinDLL(dll_path)
    else:
        lib = ctypes.WinDLL(DLL_NAME)
    log.info(f"Library '{DLL_NAME}' loaded successfully.")
except Exception as e:
    log.critical(f"Failed to load {DLL_NAME}. Application cannot run.", exc_info=True)
    messagebox.showerror("Library Load Error", f"Failed to load {DLL_NAME}.\nError: {e}")
    lib = None

ND_PAR_CH_EN = 0x81
ND_PAR_CH_PROF = 0x86
ND_PROF_EXG = 1
ND_DISCOVERY_PROGRESS = 1
ND_DISCOVERY_SUCCESS = 3
ND_MEASURE_NORMAL = 0
ND_IND_SIGNAL = 3
ND_SIG_OK = 0
ND_SIG_LACK = 1


class NDVAL(ctypes.Union):
    _fields_ = [("i", ctypes.c_int), ("b", ctypes.c_bool), ("f", ctypes.c_float), ("t", ctypes.c_char_p)]


class NDSETVAL(ctypes.Union):
    _fields_ = [("opt", ctypes.c_uint), ("val", NDVAL)]


class NdAvailDevData(ctypes.Structure):
    _fields_ = [("status", ctypes.c_ushort), ("devModel", ctypes.c_char * 33), ("devSN", ctypes.c_char * 13),
                ("transcId", ctypes.c_char * 20)]


class NdPackChan(ctypes.Structure):
    _fields_ = [("num", ctypes.c_ushort), ("samps", ctypes.POINTER(ctypes.c_int32)), ("sig_st", ctypes.c_ushort),
                ("mask", ctypes.c_ushort)]


TProcSamples = ctypes.CFUNCTYPE(None, ctypes.c_ushort, ctypes.c_ushort, ctypes.c_ushort, ctypes.POINTER(NdPackChan))

class NeurobitDevice(BaseDevice):
    SAMPLING_RATE = 250

    def __init__(self):
        super().__init__()
        if lib is None: raise RuntimeError("Neurobit DLL is not loaded.")
        self.context_handle = -1;
        self.engine_thread = None
        self.stop_engine_event = threading.Event()
        self.num_channels = 0
        self.device_name = "Neurobit Optima"
        self._c_callback = TProcSamples(self._eeg_data_callback)
        lib.NdSetCallbacks.argtypes = [TProcSamples, ctypes.c_void_p, ctypes.c_void_p]
        lib.NdSetCallbacks(self._c_callback, None, None)
        log.info("Neurobit callbacks configured.")

    def _eeg_data_callback(self, dc, phase, sum_st, chans):
        try:
            if not chans or self.num_channels == 0: return
            data_to_send = {}
            for i in range(self.num_channels):
                channel = chans[i]
                if channel and channel.num > 0 and channel.samps:
                    samples = np.ctypeslib.as_array(channel.samps, shape=(channel.num,))
                    data_to_send[f'CH{i}'] = samples.copy()
            if data_to_send and len(data_to_send) == self.num_channels:
                self.data_queue.put(data_to_send)
        except Exception:
            log.error("Error in Neurobit callback", exc_info=True)

    def connect(self) -> bool:
        log.info("Searching for Neurobit device...")
        dev_data = NdAvailDevData()
        lib.NdFindFirstAvailDev.argtypes = [ctypes.c_char_p, ctypes.c_ushort, ctypes.POINTER(NdAvailDevData)]
        lib.NdFindFirstAvailDev.restype = ctypes.c_int
        result = lib.NdFindFirstAvailDev(None, 0xffff, ctypes.byref(dev_data))
        if result != 0: log.error(f"Failed to start Neurobit discovery. Code: {result}"); return False
        log.info("... discovery in progress ...")
        while dev_data.status == ND_DISCOVERY_PROGRESS: lib.NdProtocolEngine(); time.sleep(0.1)
        if dev_data.status != ND_DISCOVERY_SUCCESS: log.error("Neurobit device not found."); return False
        self.specific_device_model = dev_data.devModel
        device_model_str = self.specific_device_model.decode('utf-8')

        log.info(f"Device found: {device_model_str}")
        if "4" in device_model_str:
            self.num_channels = 4
        else:
            self.num_channels = 2
        log.info(f"Detected {self.num_channels}-channel Neurobit device.")
        lib.NdOpenDevContext.argtypes = [ctypes.c_char_p];
        lib.NdOpenDevContext.restype = ctypes.c_int
        self.context_handle = lib.NdOpenDevContext(self.specific_device_model)
        if self.context_handle >= 0:
            log.info(f"Context opened. ID: {self.context_handle}")
            if not self.configure_channels(): self.disconnect(); return False
            log.info("Applying settings...");
            [lib.NdProtocolEngine() for _ in range(10)];
            time.sleep(0.05)
            log.info("Settings applied.")
            return True
        else:
            log.error(f"Failed to open context. Code: {self.context_handle}")
            self.context_handle = -1;
            return False

    def start_stream(self) -> bool:
        """
        Starts the data measurement stream.
        FINAL VERSION: Proactively performs a soft-reset using the specific device
        model ID that was captured during the initial connection. This ensures
        a robust and reliable state transition between measurement sessions.
        """
        if self.is_running:
            log.warning("Stream is already running.")
            return True

        if self.is_connected():
            log.info("Performing proactive soft-reset before starting stream...")

            if not self.specific_device_model:
                log.error("Cannot perform soft-reset: specific device model was not stored.")
                return False

            self.disconnect()
            time.sleep(0.5)

            self.context_handle = lib.NdOpenDevContext(self.specific_device_model)

            if self.context_handle < 0:
                log.error(f"Soft-reset failed: Could not reconnect to the device. Code: {self.context_handle}")
                self.context_handle = -1
                return False

            log.info(f"Soft-reset re-opened context. ID: {self.context_handle}")
            if not self.configure_channels():
                self.disconnect()
                return False
            log.info("Applying settings after soft-reset...");
            [lib.NdProtocolEngine() for _ in range(10)];
            time.sleep(0.05)

        elif not self.connect():
            log.error("Cannot start stream: failed to connect to device.")
            return False

        lib.NdStartMeasurement.argtypes = [ctypes.c_ushort, ctypes.c_ushort]
        lib.NdStartMeasurement.restype = ctypes.c_int
        result = lib.NdStartMeasurement(self.context_handle, ND_MEASURE_NORMAL)

        if result != 0:
            log.error(f"Failed to start measurement even after soft-reset. Code: {result}")
            self.disconnect()
            return False

        self.stop_engine_event.clear()
        self.engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self.engine_thread.start()
        self.is_running = True
        log.info("Measurement started successfully.")
        return True

    def get_data(self) -> np.ndarray | None:
        """
        MODIFIED: Pulls all available data from the queue and returns it as a single
        NumPy array, dynamically handling the number of channels and ensuring the
        data type is float64 for compatibility with BrainFlow filters.
        """
        if self.data_queue.empty():
            return None

        data_chunks = []
        while not self.data_queue.empty():
            data_chunks.append(self.data_queue.get_nowait())

        if not data_chunks:
            return None

        all_channel_data = []
        for i in range(self.num_channels):
            channel_key = f'CH{i}'
            if any(channel_key in d for d in data_chunks):
                concatenated_channel = np.concatenate([d[channel_key] for d in data_chunks if channel_key in d])
                all_channel_data.append(concatenated_channel)

        if not all_channel_data:
            return None

        return np.vstack(all_channel_data).astype(np.float64)

    def get_sampling_rate(self) -> int:
        return self.SAMPLING_RATE

    def get_eeg_channels(self) -> list:
        return list(range(self.num_channels))

    def get_primary_eeg_channel_index(self) -> int:
        return 0

    def get_frontal_channel_indices(self) -> dict | None:
        if self.num_channels == 2:
            log.info("2-channel Neurobit device detected. Assuming CH0=Fp1 and CH1=Fp2 for Valence Score.")
            return {'fp1': 0, 'fp2': 1}
        else:
            # log.warning(f"Valence Score (FAA) is disabled for {self.num_channels}-channel Neurobit device.")
            return None

    def is_connected(self) -> bool:
        return self.context_handle >= 0

    def check_signal_quality(self) -> bool:
        if not self.is_running: return False
        lib.NdGetUserInd.argtypes = [ctypes.c_ushort, ctypes.c_int, ctypes.c_ushort]
        lib.NdGetUserInd.restype = ctypes.c_int
        for i in range(self.num_channels):
            if lib.NdGetUserInd(self.context_handle, ND_IND_SIGNAL, i) != ND_SIG_OK:
                log.warning(f"Bad signal detected on channel {i}.")
                return False
        return True

    def stop_stream(self):
        if not self.is_running: return
        log.info("Stopping data stream...")
        self.stop_engine_event.set()
        if self.engine_thread: self.engine_thread.join(timeout=1)
        lib.NdStopMeasurement.argtypes = [ctypes.c_ushort]
        lib.NdStopMeasurement.restype = ctypes.c_int
        lib.NdStopMeasurement(self.context_handle)
        log.info("Measurement stopped.")
        self.is_running = False

    def disconnect(self):
        if self.is_running: self.stop_stream()
        if self.context_handle >= 0:
            lib.NdCloseDevContext.argtypes = [ctypes.c_ushort]
            lib.NdCloseDevContext(self.context_handle)
            log.info("Context closed, resources released.")
            self.context_handle = -1

    def _run_engine(self):
        log.info("Starting NdProtocolEngine in a background thread...")
        while not self.stop_engine_event.is_set():
            lib.NdProtocolEngine()
            time.sleep(0.001)
        log.info("NdProtocolEngine background thread stopped.")

    def configure_channels(self) -> bool:
        log.info("Configuring channels...")
        lib.NdSetParam.argtypes = [ctypes.c_ushort, ctypes.c_ushort, ctypes.POINTER(NDSETVAL)]
        lib.NdSetParam.restype = ctypes.c_int
        for i in range(self.num_channels):
            val_prof = NDSETVAL();
            val_prof.opt = ND_PROF_EXG
            if lib.NdSetParam(ND_PAR_CH_PROF, i, ctypes.byref(val_prof)) != 0: log.warning(
                f"Could not set profile for channel {i}.")
            val_en = NDSETVAL();
            val_en.val.b = True
            if lib.NdSetParam(ND_PAR_CH_EN, i, ctypes.byref(val_en)) != 0: log.error(
                f"ailed to enable channel {i}."); return False
        log.info("Channels configured successfully.")
        return True