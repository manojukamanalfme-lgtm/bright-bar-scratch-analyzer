import os
#print("RUNNING FILE:", os.path.abspath(__file__))
#print("CWD:", os.getcwd())
import os
import json
import time
import logging
import threading
import numpy as np
import cv2
import csv
import signal
import hashlib
import uuid
import psutil
import atexit
from datetime import datetime, timedelta
from collections import deque
from sort import Sort
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pathlib import Path
import zipfile
import io
from flask import send_from_directory

from gi.repository import Gst, GLib
from queue import Queue
ssr_queue = Queue()
ssr_lock = threading.Lock()
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst
Gst.init(None)        

# ===== SSR Diagnostics =====
ssr_pulse_count = 0
ssr_last_pulse_time = None
        
class GstCamera:
    def __init__(self, width=1280, height=720):
        self.frame = None
        self.lock = threading.Lock()
        self.is_running = False
        self.pipeline = None
        self.appsink = None
        self.mainloop = None
        self.mainloop_thread = None
        self.frame_available = threading.Event()

        try:
            self.pipeline_str = (
                "pylonsrc! "
                "queue max-size-buffers=2 leaky=downstream ! "
                f"video/x-raw,format=BGR,width={width},height={height} ! "
                "videoconvert ! "
                "appsink name=appsink "
                "emit-signals=true sync=false max-buffers=1 drop=true"
            )
            self.pipeline = Gst.parse_launch(self.pipeline_str)
            self.appsink = self.pipeline.get_by_name("appsink")
            if self.appsink is None:
                raise RuntimeError("appsink not found in GStreamer pipeline")
            self.appsink.connect("new-sample", self.on_new_sample)
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start GStreamer pipeline")
            self.is_running = True
            self.mainloop = GLib.MainLoop()
            self.mainloop_thread = threading.Thread(
                target=self._run_mainloop,
                daemon=False,
                name="GstMainLoop"
            )
            self.mainloop_thread.start()
            logging.info("GStreamer camera initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize GStreamer camera: {e}")
            self.cleanup()
            raise

    def _run_mainloop(self):
        try:
            if self.mainloop:
                self.mainloop.run()
        except Exception as e:
            logging.error(f"Mainloop error: {e}")
        finally:
            logging.info("Mainloop thread exiting")

    def on_new_sample(self, sink):
        if not self.is_running:
            return Gst.FlowReturn.EOS
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
            expected_size = width * height * 3
            if frame.size != expected_size:
                return Gst.FlowReturn.OK
            frame = frame.reshape((height, width, 3))
            with self.lock:
                self.frame = frame.copy()
                self.frame_available.set()
        finally:
            buffer.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def read(self):
        if not self.is_running:
            return False, None
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def wait_for_first_frame(self, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_running:
                raise RuntimeError("Camera stopped during initialization")
            with self.lock:
                if self.frame is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError("Camera timeout: no frames received")

    def is_healthy(self, timeout=2.0):
        if not self.is_running:
            return False
        self.frame_available.clear()
        return self.frame_available.wait(timeout=timeout)

    def cleanup(self):
        if not self.is_running:
            return
        logging.info("Starting GStreamer camera cleanup...")
        self.is_running = False
        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                ret = self.pipeline.get_state(timeout=5 * Gst.SECOND)
        except Exception as e:
            logging.error(f"Error stopping pipeline: {e}")
        try:
            if self.mainloop and self.mainloop.is_running():
                self.mainloop.quit()
        except Exception as e:
            logging.error(f"Error quitting mainloop: {e}")
        try:
            if self.mainloop_thread and self.mainloop_thread.is_alive():
                self.mainloop_thread.join(timeout=3.0)
        except Exception as e:
            logging.error(f"Error joining mainloop thread: {e}")
        self.appsink = None
        self.pipeline = None
        self.mainloop = None
        logging.info("GStreamer camera cleanup completed")

    def stop(self):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass

# ===== DEBUG MODE =====
DEBUG_MODE = False  

# ===== VIDEO MODE CONFIGURATION =====
VIDEO_MODE = False
VIDEO_PATH = r"D:\KAP\KAP_AI\ManojSir_Task\Rod\ROD_PRD_V4\Videos\rod.mp4"
VIDEO_LOOP = False

# ===== GPIO Configuration =====
try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
    logging.info("Jetson.GPIO imported successfully - GPIO ENABLED")
except (ImportError, RuntimeError) as e:
    print(f"GPIO not available - using mock mode: {e}")
    GPIO_AVAILABLE = False
    
    class MockGPIO:
        BOARD = "BOARD"
        OUT = "OUT"
        IN = "IN"
        HIGH = 1
        LOW = 0
        PUD_DOWN = "PUD_DOWN"
        @staticmethod
        def setmode(mode): print("[MOCK GPIO] setmode called")
        @staticmethod
        def setup(pin, mode, **kwargs): print(f"[MOCK GPIO] setup pin {pin} mode {mode}")
        @staticmethod
        def output(pin, state): print(f"[MOCK GPIO] output pin {pin} state {state}")
        @staticmethod
        def input(pin): return True
        @staticmethod
        def cleanup(): print("[MOCK GPIO] cleanup called")
    
    GPIO = MockGPIO()

# ===== Flask Setup =====
app = Flask(__name__, static_folder='assets', static_url_path='/static')
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

# ===== Global Variables =====
stop_event = threading.Event()
speed_display_event = threading.Event()
total_count = 0
rod_status = {}
is_running = True
sleep_mode = False
last_conveyor_state = True
cleanup_done = False
gpio_initialized = False

inactivity_timer = None
INACTIVITY_TIMEOUT = 900

ZONE_HALF_WIDTH_PIXELS = 50

SPEED_ZERO_TIMEOUT = 2.0
last_speed_update_time = 0.0

rod_speed_mm_sec = 0.0
latest_frame = None
frame_lock = threading.Lock()
conveyor_active = True

heat_number = None
_current_heat_number = None
_current_log_date = None
pause_detection = False
heat_start_time = None
prev_heat_number = None
prev_heat_total_count = 0
heat_metadata = {}
_current_heat_metadata = None

_logged_heats = {}
_logged_heats_lock = threading.Lock()

hourly_net_count = 0
hour_start_time = None
hourly_log_path = None
heat_start_date_str = None

frame_count = 0
captured_frames_dir = None

video_fps = 30.0
video_total_frames = 0

ui_lock = threading.Lock()

ui_state = {
    "heat_uuid": None,
    "heat_number": None,
    "heat_start_time": None,
    "checksum": None,
    "metadata": {},
    "metadata_locked": False,
    "speed_enabled": False
}

video_processing_started = threading.Event()

camera_instance = None

camera_health_check_interval = 5.0
last_camera_health_check = 0

speed_history = deque(maxlen=30)
speed_lock = threading.Lock()

# ===== Load Config =====
CONFIG_PATH = r"./config.json"
if not os.path.exists(CONFIG_PATH):
    default_config = {
        "output_dir": "./output",
        "engine_path": "./Models/best.pt",
        "confidence_threshold": 0.25,
        "class_id": [0],
        "class_names": ["rod-lcgJ"],
        "ssr_pin": 7,
        "led": {
            "green_pin": 33,
            "yellow_pin": 29,
            "red_pin": 31,
            "conveyor_pin": 15
        },
        "system": {"inactivity_timeout": 900},
        "logging": {
            "summary_enabled": True,
            "summary_file": "summary/heat_summary.csv",
            "retention_days": 90,
            "hourly_dir": "hourly"
        },
        "video": {
            "default_path": "D:\\project\\rod.mp4",
            "loop": True,
            "speed": 1.0,
            "auto_pause_on_inactivity": False
        },
        "calibration": {"mm_per_pixel": 0.496875}
    }
    with open(CONFIG_PATH, 'w') as f:
        json.dump(default_config, f, indent=2)
    print(f"Created default config at {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    config = json.load(f)

MM_PER_PIXEL = config.get("calibration", {}).get("mm_per_pixel")
if MM_PER_PIXEL is None:
    print("ERROR: calibration.mm_per_pixel missing in config.json")
    exit(1)

if "video" in config:
    if "default_path" in config["video"]:
        VIDEO_PATH = config["video"]["default_path"]
    if "loop" in config["video"]:
        VIDEO_LOOP = config["video"]["loop"]

OUTPUT_DIR = os.path.abspath(config["output_dir"])
log_dir     = os.path.join(OUTPUT_DIR, "logs")
summary_dir = os.path.join(OUTPUT_DIR, "summary")
hourly_dir  = os.path.join(OUTPUT_DIR, "hourly")
frames_dir  = os.path.join(OUTPUT_DIR, "frames")
DATA_DIR    = os.path.join(OUTPUT_DIR, "runtime")

for d in (log_dir, summary_dir, hourly_dir, frames_dir, DATA_DIR):
    os.makedirs(d, exist_ok=True)

ACTIVE_HEAT_FILE = os.path.join(DATA_DIR, "active_heat.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# =========================
# UTILITIES
# =========================
def generate_heat_uuid():
    return str(uuid.uuid4())

def compute_checksum(meta: dict):
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()

def persist_active_heat():
    global ui_state
    tmp = ACTIVE_HEAT_FILE + ".tmp"
    try:
        with ui_lock:
            if not ui_state.get("metadata_locked"):
                for filepath in [ACTIVE_HEAT_FILE, tmp]:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                logging.info("Heat state cleared (no active heat)")
                return
            state_to_save = ui_state.copy()
        with open(tmp, "w") as f:
            json.dump(state_to_save, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ACTIVE_HEAT_FILE)
        logging.info(f"Heat state persisted: {state_to_save.get('heat_number')}")
    except Exception as e:
        logging.error(f"Failed to persist heat state: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass

def restore_active_heat():
    global _current_heat_number, _current_heat_metadata, heat_start_time, heat_number, video_processing_started
    if not os.path.exists(ACTIVE_HEAT_FILE):
        logging.info("No active heat file found - WAITING for heat to be set")
        return
    try:
        with open(ACTIVE_HEAT_FILE) as f:
            state = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read active heat file: {e}")
        return
    try:
        if compute_checksum(state.get("metadata", {})) != state.get("checksum"):
            logging.error("Active heat checksum mismatch - restore skipped")
            return
    except Exception as e:
        logging.error(f"Active heat validation error: {e}")
        return
    if not state.get("metadata_locked"):
        logging.info("Previous heat was not locked - starting fresh")
        return
    with ui_lock:
        ui_state.update(state)
    try:
        _current_heat_number  = state["heat_number"]
        heat_number           = state["heat_number"]
        _current_heat_metadata = state["metadata"]
        heat_start_time       = datetime.fromisoformat(state["heat_start_time"])
        if state.get("speed_enabled"):
            speed_display_event.set()
        video_processing_started.set()
    except Exception as e:
        logging.error(f"Active heat state malformed: {e}")
        return
    setup_logger(new_heat=_current_heat_number)
    logging.info(f"Restored active heat {_current_heat_number} from {state['heat_start_time']}")
    print(f"Heat {_current_heat_number} restored successfully")

# =========================
# LOGGING FUNCTIONS
# =========================
def _summary_window_date(now=None):
    if now is None:
        now = datetime.now()
    six_am = now.replace(hour=6, minute=0, second=0, microsecond=0)
    window_date = now.date() if now >= six_am else (now - timedelta(days=1)).date()
    return window_date.strftime("%Y-%m-%d")

def _build_summary_log_path():
    day_str = _summary_window_date()
    return os.path.join(summary_dir, f"heat_summary_{day_str}.csv")

def _hourly_filename(heat_num, start_date_str):
    return os.path.join(hourly_dir, f"{heat_num}_{start_date_str}.csv")

def _init_hourly_log_for_heat(heat_num):
    global hourly_log_path, hour_start_time, hourly_net_count, heat_start_date_str, captured_frames_dir
    heat_start_date_str = datetime.now().strftime("%Y-%m-%d")
    hourly_log_path = _hourly_filename(heat_num, heat_start_date_str)
    captured_frames_dir = os.path.join(frames_dir, f"{heat_num}_{heat_start_date_str}")
    os.makedirs(captured_frames_dir, exist_ok=True)
    tmp = hourly_log_path + ".tmp"
    try:
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start_time', 'end_time', 'bars_in_hour', 'total_bars'])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, hourly_log_path)
        logging.info(f"Created hourly log: {hourly_log_path}")
    except Exception as e:
        logging.error(f"Failed to create hourly log: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass
    hour_start_time  = datetime.now().replace(minute=0, second=0, microsecond=0)
    hourly_net_count = 0

def _append_hour_row(end_time, rods_in_hour):
    if hourly_log_path is None or hour_start_time is None:
        return
    tmp = hourly_log_path + ".tmp"
    try:
        content = open(hourly_log_path, 'r').read() if os.path.exists(hourly_log_path) else ""
        with open(tmp, 'w', newline='') as f:
            f.write(content)
            writer = csv.writer(f)
            writer.writerow([hour_start_time.isoformat(), end_time.isoformat(), rods_in_hour, total_count])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, hourly_log_path)
    except Exception as e:
        logging.error(f"Failed to append hour row: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass

def _roll_hours_if_needed(now=None):
    global hour_start_time, hourly_net_count
    if hourly_log_path is None or hour_start_time is None:
        return
    if now is None:
        now = datetime.now()
    next_boundary = hour_start_time + timedelta(hours=1)
    first = True
    while now >= next_boundary:
        rods_to_write = hourly_net_count if first else 0
        _append_hour_row(next_boundary, rods_to_write)
        hour_start_time = next_boundary
        next_boundary   = hour_start_time + timedelta(hours=1)
        if first:
            hourly_net_count = 0
            first = False

def _finalize_hourly_log():
    global hourly_log_path, hour_start_time, hourly_net_count, total_count
    if hourly_log_path is None:
        return
    now = datetime.now()
    if hour_start_time is not None:
        _append_hour_row(now, hourly_net_count)
    tmp = hourly_log_path + ".tmp"
    try:
        content = open(hourly_log_path, 'r').read() if os.path.exists(hourly_log_path) else ""
        with open(tmp, 'w', newline='') as f:
            f.write(content)
            writer = csv.writer(f)
            writer.writerow(["TOTAL_BARS", "", "", total_count])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, hourly_log_path)
        logging.info(f"Finalized hourly log with total: {total_count}")
    except Exception as e:
        logging.error(f"Failed to finalize hourly log: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass
    hourly_log_path = None

def setup_logger(new_heat: str = None):
    global _current_heat_number, _current_log_date, total_count, heat_start_time, _current_heat_metadata
    global prev_heat_number, prev_heat_total_count

    if heat_number is None and new_heat is None:
        return

    window_date = _summary_window_date()
    heat_to_use = new_heat if new_heat is not None else heat_number

    rotate_for_heat_change = (_current_heat_number is None) or (heat_to_use != _current_heat_number)
    rotate_for_date_change = (_current_log_date is None) or (window_date != _current_log_date)

    if not rotate_for_heat_change and not rotate_for_date_change:
        return

    if rotate_for_heat_change and _current_heat_number and heat_start_time:
        heat_key = f"{_current_heat_number}_{heat_start_time.isoformat()}"
        with _logged_heats_lock:
            if heat_key not in _logged_heats:
                log_heat_summary(_current_heat_number, heat_start_time, datetime.now(), total_count)
                _logged_heats[heat_key] = {
                    'start_time': heat_start_time.isoformat(),
                    'logged': True,
                    'count': total_count
                }
                logging.info(f"Heat {_current_heat_number} logged from setup_logger (count={total_count})")
            else:
                logging.info(f"Heat {_current_heat_number} already logged, skipping duplicate in setup_logger")
        _finalize_hourly_log()
        prev_heat_number      = _current_heat_number
        prev_heat_total_count = total_count

    if rotate_for_heat_change:
        total_count     = 0
        heat_start_time = datetime.now()

    _current_heat_number = heat_to_use
    _current_log_date    = window_date
    os.makedirs(summary_dir, exist_ok=True)

    if rotate_for_heat_change:
        _init_hourly_log_for_heat(_current_heat_number)


# =====================================================================
# ✅ UPDATED: log_heat_summary — weight-based model
#    REMOVED: bloom_length_plan
#    ADDED:   bloom_weight_kg, density_kg_m3, weight_per_bar_kg
#    KEPT:    actual_bloom_length (still the cutting bar length)
# =====================================================================
def log_heat_summary(heat_num, start_time, end_time, rod_count, metadata=None):
    global _current_heat_metadata
    summary_path   = _build_summary_log_path()
    os.makedirs(summary_dir, exist_ok=True)
    start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
    end_time_str   = end_time.isoformat()   if isinstance(end_time,   datetime) else end_time

    meta = metadata if metadata is not None else _current_heat_metadata if _current_heat_metadata is not None else {}

    grade                  = meta.get('grade', '')
    customer               = meta.get('customer', '')
    bloom_size_a           = meta.get('bloom_size_a', '')
    bloom_size_b           = meta.get('bloom_size_b', '')
    bloom_size             = meta.get('bloom_size', f"{bloom_size_a}x{bloom_size_b}" if bloom_size_a else '')
    # ✅ NEW weight-based fields (replaces bloom_length_plan)
    bloom_weight_kg        = meta.get('bloom_weight_kg', '')
    density_kg_m3          = meta.get('density_kg_m3', '')
    weight_per_bar_kg      = meta.get('weight_per_bar_kg', '')
    # ✅ KEPT: actual cutting bar length
    actual_bloom_length    = meta.get('actual_bloom_length', '')
    customer_supply_length = meta.get('customer_supply_length', '')
    short_blooms           = meta.get('short_blooms', '')
    profile                = meta.get('profile', '')
    rolling_size           = meta.get('rolling_size', '')
    theoretical_bars       = meta.get('theoretical_bars', '')
    theoretical_bars_round = meta.get('theoretical_bars_round', '')

    file_exists = os.path.exists(summary_path)
    tmp = summary_path + ".tmp"
    try:
        content = open(summary_path, 'r').read() if file_exists else ""
        with open(tmp, 'w', newline='') as f:
            if not file_exists:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                # ✅ UPDATED header — weight-based columns
                writer.writerow([
                    'heat_number', 'start_time', 'end_time',
                    'grade', 'customer',
                    'bloom_size_width', 'bloom_size_height', 'bloom_size',
                    'bloom_weight_kg', 'density_kg_m3', 'weight_per_bar_kg',
                    'actual_bloom_length',
                    'customer_supply_length', 'short_blooms',
                    'profile', 'rolling_size',
                    'theoretical_bars', 'theoretical_bars_round',
                    'total_bar_count'
                ])
            else:
                f.write(content)
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # ✅ UPDATED data row — weight-based values
            writer.writerow([
                heat_num, start_time_str, end_time_str,
                grade, customer,
                bloom_size_a, bloom_size_b, bloom_size,
                bloom_weight_kg, density_kg_m3, weight_per_bar_kg,
                actual_bloom_length,
                customer_supply_length, short_blooms,
                profile, rolling_size,
                theoretical_bars, theoretical_bars_round,
                rod_count
            ])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, summary_path)
    except Exception as e:
        logging.error(f"Failed to write heat summary: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass

# =========================
# SLEEP MODE & CONVEYOR
# =========================
def enter_sleep_mode():
    global sleep_mode
    if not sleep_mode:
        sleep_mode = True
        set_led_state(green=False, yellow=True, red=False)
        logging.info("Entered sleep mode - no conveyor activity")
        socketio.emit('system_status', {'mode': 'sleep', 'message': 'Sleep mode active'})

def exit_sleep_mode():
    global sleep_mode, last_camera_health_check
    if sleep_mode:
        sleep_mode = False
        last_camera_health_check = time.time()
        set_led_state(green=True, yellow=False, red=False)
        logging.info("Resumed active detection from sleep mode")
        socketio.emit('system_status', {'mode': 'active', 'message': 'Detection active'})

def reset_inactivity_timer():
    global inactivity_timer
    if inactivity_timer:
        inactivity_timer.cancel()
    inactivity_timer = threading.Timer(INACTIVITY_TIMEOUT, enter_sleep_mode)
    inactivity_timer.daemon = True
    inactivity_timer.start()

# =========================
# GPIO / HARDWARE
# =========================
def init_gpio():
    global gpio_initialized
    if not GPIO_AVAILABLE:
        logging.info("GPIO mock mode - hardware features disabled")
        gpio_initialized = False
        return
    try:
        GPIO.setmode(GPIO.BOARD)
        ssr_pin = config.get("ssr_pin", 7)
        GPIO.setup(ssr_pin, GPIO.OUT)
        GPIO.output(ssr_pin, GPIO.LOW)
        green_pin  = config["led"]["green_pin"]
        yellow_pin = config["led"]["yellow_pin"]
        red_pin    = config["led"]["red_pin"]
        GPIO.setup(green_pin,  GPIO.OUT)
        GPIO.setup(yellow_pin, GPIO.OUT)
        GPIO.setup(red_pin,    GPIO.OUT)
        conveyor_pin = config["led"]["conveyor_pin"]
        GPIO.setup(conveyor_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        gpio_initialized = True
        logging.info("GPIO initialized successfully")
    except Exception as e:
        logging.error(f"GPIO initialization failed: {e}")
        gpio_initialized = False

def set_led_state(green=False, yellow=False, red=False):
    if not gpio_initialized:
        return
    try:
        GPIO.output(config["led"]["green_pin"],  GPIO.HIGH if green  else GPIO.LOW)
        GPIO.output(config["led"]["yellow_pin"], GPIO.HIGH if yellow else GPIO.LOW)
        GPIO.output(config["led"]["red_pin"],    GPIO.LOW  if red    else GPIO.HIGH)
    except Exception as e:
        logging.error(f"LED control error: {e}")

def ssr_worker():
    global ssr_pulse_count, ssr_last_pulse_time
    ssr_pin        = config.get("ssr_pin", 7)
    pulse_duration = 0.1
    while True:
        direction = ssr_queue.get()
        if direction is None:
            break
        with ssr_lock:
            GPIO.output(ssr_pin, GPIO.HIGH)
            t_high_start = time.monotonic()
            time.sleep(pulse_duration)
            GPIO.output(ssr_pin, GPIO.LOW)
            t_high_end = time.monotonic()
            time.sleep(0.02)
        actual_pulse_ms   = (t_high_end - t_high_start) * 1000
        ssr_pulse_count  += 1
        ssr_last_pulse_time = datetime.now()
        print(
            f"[SSR] #{ssr_pulse_count:05d} | "
            f"dir={direction:<9} | "
            f"HIGH={actual_pulse_ms:6.2f} ms | "
            f"time={ssr_last_pulse_time.strftime('%H:%M:%S.%f')[:-3]}"
        )
        logging.info(f"[SSR] Pulse #{ssr_pulse_count} | dir={direction} | high_ms={actual_pulse_ms:.2f}")
        ssr_queue.task_done()

def check_conveyor_status():
    if not gpio_initialized:
        return True
    try:
        return GPIO.input(config["led"]["conveyor_pin"])
    except Exception as e:
        logging.error(f"Conveyor sensor error: {e}")
        return True

# ===== YOLO & DETECTION =====
BASE_DIR    = Path(__file__).resolve().parent
engine_cfg  = config.get("engine_path", r".\Models\best.pt")
engine_path = Path(engine_cfg)
ENGINE_PATH = engine_path if engine_path.is_absolute() else (BASE_DIR / engine_path).resolve()

print("CONFIG engine_path:", engine_cfg)
print("ENGINE_PATH resolved to:", ENGINE_PATH)

if not ENGINE_PATH.is_file():
    raise FileNotFoundError(f"Model file not found at {ENGINE_PATH}")

model   = YOLO(ENGINE_PATH)
tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.2)

logging.info(f"Model loaded: {ENGINE_PATH}")
logging.info(f"Confidence threshold: {config['confidence_threshold']}")
logging.info(f"Class IDs: {config['class_id']}")

# ===== Frame Processing =====
def process_frame(image, frame_count):
    global total_count, hourly_net_count, rod_speed_mm_sec, last_speed_update_time, latest_frame
    
    draw = image.copy()
    height, width, _ = image.shape

    roi_w  = int(width  * 0.50)
    roi_h  = int(height * 0.95)
    roi_x0 = (width  - roi_w) // 2
    roi_y0 = (height - roi_h) // 2
    roi_x1 = roi_x0 + roi_w
    roi_y1 = roi_y0 + roi_h

    roi_view = image[roi_y0:roi_y1, roi_x0:roi_x1]
    output   = model(roi_view, verbose=False)
    detections  = []
    total_boxes = len(output[0].boxes) if output[0].boxes is not None else 0

    for box in output[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        if DEBUG_MODE:
            bx1, by1, bx2, by2 = x1+roi_x0, y1+roi_y0, x2+roi_x0, y2+roi_y0
            if conf > config["confidence_threshold"] and cls_id in config["class_id"]:
                cv2.rectangle(draw, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(draw, (bx1, by1 - label_size[1] - 8),
                              (bx1 + label_size[0] + 4, by1), (0, 255, 0), -1)
                cv2.putText(draw, label, (bx1 + 2, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(draw, (bx1, by1), (bx2, by2), (0, 0, 255), 1)
        if conf > config["confidence_threshold"] and cls_id in config["class_id"]:
            detections.append([x1+roi_x0, y1+roi_y0, x2+roi_x0, y2+roi_y0, conf])

    tracked = np.empty((0, 5)) if not detections else tracker.update(np.array(detections))

    roi_xmin, roi_xmax = int(width * 0.05), int(width * 0.95)
    roi_ymin, roi_ymax = int(height * 0),   int(height * 1)
    vertical_x   = (roi_xmin + roi_xmax) // 2
    zone_start_x = vertical_x - ZONE_HALF_WIDTH_PIXELS
    zone_end_x   = vertical_x + ZONE_HALF_WIDTH_PIXELS

    current_time  = time.time()
    speed_samples = []

    for obj in tracked:
        obj_id = int(obj[4])
        x1, y1, x2, y2 = map(int, obj[:4])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if roi_xmin <= cx <= roi_xmax and roi_ymin <= cy <= roi_ymax:
            if obj_id not in rod_status:
                is_reentry_from_right = cx > zone_end_x
                rod_status[obj_id] = {
                    "prev_cx": cx,
                    "prev_time": current_time,
                    "prev_cx_for_speed": cx,
                    "speed_samples": deque(maxlen=5),
                    "counted": is_reentry_from_right,
                    "armed": False,
                    "last_zone": None
                }

            rod_status[obj_id]["prev_cx"] = cx

            if   cx < zone_start_x: zone = "LEFT"
            elif cx > zone_end_x:   zone = "RIGHT"
            else:                   zone = "ZONE"

            state     = rod_status[obj_id]
            last_zone = state["last_zone"]
            counted   = state["counted"]

            if not counted:
                if last_zone == "ZONE" and zone == "RIGHT":
                    total_count     += 1
                    hourly_net_count += 1
                    state["counted"] = True
                    state["armed"]   = False
                    logging.info(f"[COUNT+] Bar {obj_id} exited zone RIGHT → TOTAL={total_count}")
                    socketio.emit('count_update', {'total': total_count})
                    ssr_queue.put("increment")

            if counted and zone == "ZONE":
                state["armed"] = True

            if counted and state["armed"]:
                if last_zone == "ZONE" and zone == "RIGHT":
                    state["armed"] = False
                if last_zone == "ZONE" and zone == "LEFT":
                    total_count       = max(0, total_count - 1)
                    hourly_net_count -= 1
                    state["counted"]  = False
                    state["armed"]    = False
                    logging.info(f"[COUNT-] Bar {obj_id} exited zone LEFT → TOTAL={total_count}")
                    socketio.emit('count_update', {'total': total_count})
                    ssr_queue.put("decrement")

            state["last_zone"] = zone

            if speed_display_event.is_set():
                prev_time     = rod_status[obj_id]["prev_time"]
                prev_cx_speed = rod_status[obj_id]["prev_cx_for_speed"]
                delta_t       = current_time - prev_time
                if delta_t > 0.02:
                    delta_x_pixels = abs(cx - prev_cx_speed)
                    if delta_x_pixels > 0:
                        speed_mm_sec = (delta_x_pixels * MM_PER_PIXEL) / delta_t
                        if 10 < speed_mm_sec < 20000:
                            speed_samples.append(speed_mm_sec)
                            rod_status[obj_id]["speed_samples"].append(speed_mm_sec)
                            rod_status[obj_id]["prev_time"]         = current_time
                            rod_status[obj_id]["prev_cx_for_speed"] = cx

        cv2.circle(draw, (cx, cy), 6, (0, 0, 0), -1)
        cv2.circle(draw, (cx, cy), 4, (0, 255, 255), -1)

        if DEBUG_MODE:
            id_label  = f"#{obj_id}"
            id_size   = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(draw, (cx + 8, cy - id_size[1] - 4),
                          (cx + 8 + id_size[0] + 4, cy), (0, 0, 0), -1)
            cv2.putText(draw, id_label, (cx + 10, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    if speed_display_event.is_set():
        if speed_samples:
            current_speed = sum(speed_samples) / len(speed_samples)
            with speed_lock:
                speed_history.append(current_speed)
                rod_speed_mm_sec = sum(speed_history) / len(speed_history) if speed_history else current_speed
            last_speed_update_time = current_time
        else:
            if (current_time - last_speed_update_time) > SPEED_ZERO_TIMEOUT:
                with speed_lock:
                    rod_speed_mm_sec = sum(list(speed_history)[:5]) / min(5, len(speed_history)) if speed_history else 0.0
    else:
        rod_speed_mm_sec = 0.0
        with speed_lock:
            speed_history.clear()

    cv2.line(draw, (zone_start_x, roi_ymin), (zone_start_x, roi_ymax), (0, 255, 255), 2)
    cv2.line(draw, (zone_end_x,   roi_ymin), (zone_end_x,   roi_ymax), (0, 255, 255), 2)
    cv2.rectangle(draw, (roi_x0, roi_y0), (roi_x1, roi_y1), (100, 200, 255), 1)

    def draw_text_with_bg(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale=0.6, text_color=(255, 255, 255),
                          bg_color=(0, 0, 0), thickness=1, padding=6):
        x, y = pos
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.rectangle(img,
                      (x - padding, y - text_size[1] - padding),
                      (x + text_size[0] + padding, y + padding),
                      bg_color, -1)
        cv2.rectangle(img,
                      (x - padding, y - text_size[1] - padding),
                      (x + text_size[0] + padding, y + padding),
                      (60, 60, 60), 1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        return text_size[1] + padding * 2

    panel_x, panel_y, line_height = 15, 30, 32

    draw_text_with_bg(draw, f"Frame: {frame_count}", (panel_x, panel_y),
                      font_scale=0.7, text_color=(100, 255, 100), thickness=2)
    panel_y += line_height
    draw_text_with_bg(draw, f"Total Bars: {total_count}", (panel_x, panel_y),
                      font_scale=0.9, text_color=(0, 255, 0), thickness=2)
    panel_y += line_height
    draw_text_with_bg(draw, f"Heat: {heat_number if heat_number else 'NOT SET'}", (panel_x, panel_y),
                      font_scale=0.7, text_color=(0, 255, 255), thickness=2)

    if speed_display_event.is_set():
        panel_y += line_height
        speed_color = (255, 200, 0) if rod_speed_mm_sec > 0 else (100, 100, 100)
        draw_text_with_bg(draw, f"Speed: {rod_speed_mm_sec/1000:.2f} m/s", (panel_x, panel_y),
                          font_scale=0.7, text_color=speed_color, thickness=2)

    if prev_heat_number:
        rpx, rpy = width - 250, 30
        draw_text_with_bg(draw, f"Prev Heat: {prev_heat_number}",       (rpx, rpy),
                          font_scale=0.6, text_color=(200, 200, 255), thickness=2)
        draw_text_with_bg(draw, f"Prev Total: {prev_heat_total_count}", (rpx, rpy + 28),
                          font_scale=0.6, text_color=(200, 200, 255), thickness=2)

    if DEBUG_MODE:
        debug_panel_x = 15
        debug_panel_y = height - 170
        overlay = draw.copy()
        cv2.rectangle(overlay, (debug_panel_x - 10, debug_panel_y - 35),
                      (350, height - 10), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, draw, 0.3, 0, draw)
        draw_text_with_bg(draw, f"Confidence: {config['confidence_threshold']}",
                          (debug_panel_x, debug_panel_y),
                          font_scale=0.5, text_color=(255, 255, 100), thickness=2)
        debug_panel_y += 25
        draw_text_with_bg(draw, f"Raw Detections: {total_boxes}",
                          (debug_panel_x, debug_panel_y),
                          font_scale=0.5, text_color=(255, 255, 100), thickness=2)
        debug_panel_y += 25
        draw_text_with_bg(draw, f"Filtered: {len(detections)}",
                          (debug_panel_x, debug_panel_y),
                          font_scale=0.5, text_color=(0, 255, 100), thickness=2)
        debug_panel_y += 25
        draw_text_with_bg(draw, f"Tracked: {len(tracked)}",
                          (debug_panel_x, debug_panel_y),
                          font_scale=0.5, text_color=(0, 255, 255), thickness=2)
        debug_panel_y += 25
        with speed_lock:
            hist_len = len(speed_history)
        draw_text_with_bg(draw, f"Speed Samples: {len(speed_samples)} current, {hist_len} history",
                          (debug_panel_x, debug_panel_y),
                          font_scale=0.5, text_color=(255, 200, 0), thickness=2)

    if sleep_mode:
        draw_text_with_bg(draw, "SLEEP MODE", (width - 260, height - 25),
                          font_scale=0.6, text_color=(255, 255, 0), thickness=2)

    if VIDEO_MODE:
        draw_text_with_bg(draw, "VIDEO" + (" | DEBUG" if DEBUG_MODE else ""), (15, height - 15),
                          font_scale=0.5, text_color=(255, 255, 0), thickness=2)

    source_text = "Source: Line Camera 01"
    ts = cv2.getTextSize(source_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    draw_text_with_bg(draw, source_text, ((width - ts[0]) // 2, height - 15),
                      font_scale=0.5, text_color=(180, 180, 180), thickness=2)

    with frame_lock:
        latest_frame = draw.copy()
    return draw

def camera_loop(cap):
    global frame_count, conveyor_active, video_fps, _current_log_date
    global last_camera_health_check, camera_instance

    logging.info("Camera loop waiting for heat to be set...")
    video_processing_started.wait()
    logging.info("Heat set - video processing starting")

    frame_delay = (1.0 / video_fps) if (VIDEO_MODE and video_fps > 0) else 0.03
    consecutive_read_failures = 0
    max_consecutive_failures  = 30

    while not stop_event.is_set():
        current_time = time.time()

        if not VIDEO_MODE and camera_instance is not None:
            if (current_time - last_camera_health_check) > camera_health_check_interval:
                if not sleep_mode:
                    if not camera_instance.is_healthy(timeout=1.0):
                        logging.warning("Camera health check failed")
                        camera_instance.read()
                last_camera_health_check = current_time

        if not VIDEO_MODE:
            new_conveyor_state = check_conveyor_status()
            if new_conveyor_state != conveyor_active:
                conveyor_active = new_conveyor_state
                if conveyor_active:
                    exit_sleep_mode()
                    reset_inactivity_timer()
            if conveyor_active:
                reset_inactivity_timer()

        ret, frame = cap.read()

        if not ret:
            consecutive_read_failures += 1
            if consecutive_read_failures >= max_consecutive_failures:
                if VIDEO_MODE:
                    if VIDEO_LOOP:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_count = 0
                        rod_status.clear()
                        with speed_lock:
                            speed_history.clear()
                        consecutive_read_failures = 0
                        continue
                    else:
                        stop_event.set()
                        break
                else:
                    time.sleep(0.5)
                    consecutive_read_failures = 0
                    continue
            else:
                time.sleep(0.01)
                continue

        consecutive_read_failures = 0

        if pause_detection or sleep_mode:
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.1)
            continue

        if _current_log_date is None:
            setup_logger()
        _roll_hours_if_needed()
        frame_count += 1
        process_frame(frame, frame_count)
        time.sleep(frame_delay)

    logging.info("Camera loop exiting normally")

def generate_frames():
    while not stop_event.is_set():
        if not video_processing_started.is_set():
            waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting_frame, "WAITING FOR HEAT NUMBER", (80, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(waiting_frame, "Please set heat in Control Panel", (100, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', waiting_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
        else:
            with frame_lock:
                if latest_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)

# =========================
# FLASK ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =====================================================================
# ✅ UPDATED: set_heat — stores weight-based fields, drops bloom_length_plan
# =====================================================================
@app.route('/set_heat', methods=['POST'])
def set_heat():
    global heat_number, heat_metadata, _current_heat_metadata, pause_detection, video_processing_started

    with ui_lock:
        if ui_state.get("metadata_locked"):
            return jsonify({
                'success': False,
                'error': 'Heat already in progress. Please complete current heat first.'
            }), 400

    data     = request.json
    new_heat = data.get('heat_number', '').strip()
    if not new_heat:
        return jsonify({'success': False, 'error': 'Heat number is required'}), 400

    pause_detection = True

    try:
        heat_number   = new_heat
        heat_metadata = {
            'heat_number':            new_heat,
            'grade':                  data.get('grade', ''),
            'customer':               data.get('customer', ''),
            'bloom_size_a':           data.get('bloom_size_a', ''),
            'bloom_size_b':           data.get('bloom_size_b', ''),
            'bloom_size':             data.get('bloom_size', ''),
            # ✅ NEW — weight-based fields (replaces bloom_length_plan)
            'bloom_weight_kg':        data.get('bloom_weight_kg', ''),
            'density_kg_m3':          data.get('density_kg_m3', 7850),
            'weight_per_bar_kg':      data.get('weight_per_bar_kg', ''),
            # ✅ KEPT — actual cutting bar length still used
            'actual_bloom_length':    data.get('actual_bloom_length', ''),
            'customer_supply_length': data.get('customer_supply_length', ''),
            'short_blooms':           data.get('short_blooms', ''),
            'profile':                data.get('profile', 'ROUND'),
            'rolling_size':           data.get('rolling_size', ''),
            'theoretical_bars':       data.get('theoretical_bars', ''),
            'theoretical_bars_round': data.get('theoretical_bars_round', ''),
        }

        _current_heat_metadata = heat_metadata.copy()
        heat_uuid = generate_heat_uuid()
        checksum  = compute_checksum(heat_metadata)

        with ui_lock:
            ui_state.update({
                "heat_uuid":       heat_uuid,
                "heat_number":     new_heat,
                "heat_start_time": datetime.now().isoformat(),
                "checksum":        checksum,
                "metadata":        heat_metadata,
                "metadata_locked": True,
                "speed_enabled":   speed_display_event.is_set()
            })

        setup_logger(new_heat=new_heat)
        persist_active_heat()
        video_processing_started.set()
        pause_detection = False

        logging.info(f"Heat {heat_number} set (UUID: {heat_uuid}) - VIDEO STARTED")
        return jsonify({'success': True, 'heat_number': heat_number, 'heat_uuid': heat_uuid})

    except Exception as e:
        pause_detection = False
        logging.error(f"Error setting heat: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/complete_heat', methods=['POST'])
def complete_heat():
    global heat_number, _current_heat_number, heat_start_time, total_count, _current_heat_metadata
    global prev_heat_number, prev_heat_total_count

    with ui_lock:
        if not ui_state.get("metadata_locked"):
            return jsonify({'success': False, 'error': 'No active heat to complete'}), 400

    try:
        data        = request.get_json() or {}
        final_count = max(0, int(data.get('final_count', total_count)))
        total_count = final_count

        if _current_heat_number and heat_start_time:
            heat_key = f"{_current_heat_number}_{heat_start_time.isoformat()}"
            with _logged_heats_lock:
                if heat_key not in _logged_heats:
                    _finalize_hourly_log()
                    log_heat_summary(
                        _current_heat_number,
                        heat_start_time,
                        datetime.now(),
                        final_count,
                        metadata=_current_heat_metadata
                    )
                    _logged_heats[heat_key] = {
                        'start_time': heat_start_time.isoformat(),
                        'logged': True,
                        'count': final_count
                    }
                    logging.info(f"Heat {_current_heat_number} logged from complete_heat (count={final_count})")
                else:
                    logging.info(f"Heat {_current_heat_number} already logged, skipping duplicate in complete_heat")

            prev_heat_number      = _current_heat_number
            prev_heat_total_count = final_count
            logging.info(f"Heat {_current_heat_number} completed with {final_count} bars")

        with ui_lock:
            completed_heat = ui_state.get("heat_number")
            ui_state.update({
                "heat_uuid":       None,
                "heat_number":     None,
                "heat_start_time": None,
                "checksum":        None,
                "metadata":        {},
                "metadata_locked": False
            })

        persist_active_heat()
        socketio.emit('heat_completed', {
            'heat_number': completed_heat,
            'total_count': final_count,
            'message': 'Heat completed. Ready for new heat.'
        })
        return jsonify({
            'success': True,
            'completed_heat': completed_heat,
            'total_bars': final_count,
            'message': 'Heat completed successfully. You can now enter a new heat.'
        })

    except Exception as e:
        logging.error(f"Error completing heat: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/ssr_status', methods=['GET'])
def ssr_status():
    return jsonify({
        "ssr_pulse_count": ssr_pulse_count,
        "last_pulse_time": ssr_last_pulse_time.isoformat() if ssr_last_pulse_time else None,
        "queue_depth": ssr_queue.qsize(),
        "gpio_available": GPIO_AVAILABLE,
        "gpio_initialized": gpio_initialized
    })

@app.route('/heat_status', methods=['GET'])
def heat_status():
    with ui_lock:
        is_locked = ui_state.get("metadata_locked", False)
        if is_locked:
            return jsonify({
                'heat_active': True,
                'heat_uuid': ui_state.get("heat_uuid"),
                'heat_number': ui_state.get("heat_number"),
                'heat_start_time': ui_state.get("heat_start_time"),
                'total_count': total_count,
                'metadata_locked': True,
                **ui_state.get("metadata", {})
            })
        else:
            return jsonify({
                'heat_active': False,
                'metadata_locked': False,
                'video_blocked': not video_processing_started.is_set()
            })

@app.route('/toggle_speed', methods=['POST'])
def toggle_speed():
    global rod_speed_mm_sec, last_speed_update_time
    data    = request.json
    enabled = data.get('enabled', False)
    if enabled:
        speed_display_event.set()
        rod_speed_mm_sec       = 0.0
        last_speed_update_time = time.time()
        with speed_lock:
            speed_history.clear()
        logging.info("Bar speed display ENABLED")
    else:
        speed_display_event.clear()
        with speed_lock:
            speed_history.clear()
        logging.info("Bar speed display DISABLED")
    with ui_lock:
        ui_state["speed_enabled"] = enabled
    persist_active_heat()
    return jsonify({'success': True, 'enabled': enabled})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    with speed_lock:
        hist_len = len(speed_history)
    return jsonify({
        'total_count': total_count,
        'heat_number': heat_number if heat_number else '-',
        'prev_heat': prev_heat_number if prev_heat_number else '-',
        'prev_total': prev_heat_total_count,
        'speed': rod_speed_mm_sec if speed_display_event.is_set() else 0,
        'speed_enabled': speed_display_event.is_set(),
        'speed_samples': hist_len,
        'sleep_mode': sleep_mode,
        'metadata_locked': ui_state.get('metadata_locked', False),
        'video_mode': VIDEO_MODE,
        'video_fps': video_fps if VIDEO_MODE else 0,
        'video_total_frames': video_total_frames if VIDEO_MODE else 0,
        'debug_mode': DEBUG_MODE,
        'video_blocked': not video_processing_started.is_set(),
        'gpio_available': GPIO_AVAILABLE,
        'gpio_initialized': gpio_initialized
    })

@app.route('/health', methods=['GET'])
def health():
    def read_temp():
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read()) / 1000, 1)
        except:
            return None
    uptime_seconds = int(time.time() - psutil.boot_time())
    return jsonify({
        "status": "healthy",
        "temperature_c": read_temp(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
        "uptime_seconds": uptime_seconds,
        "uptime_hours": round(uptime_seconds / 3600, 1),
        "gpio_available": GPIO_AVAILABLE,
        "gpio_initialized": gpio_initialized,
        "sleep_mode": sleep_mode,
        "active_heat": heat_number,
        "total_bars": total_count,
        "video_mode": VIDEO_MODE,
        "debug_mode": DEBUG_MODE,
        "video_blocked": not video_processing_started.is_set()
    })

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    global latest_frame, captured_frames_dir
    if latest_frame is None:
        return jsonify({'success': False, 'error': 'No frame available'}), 400
    if captured_frames_dir is None:
        return jsonify({'success': False, 'error': 'No active heat'}), 400
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"frame_{timestamp}_count{total_count}.jpg"
    filepath  = os.path.join(captured_frames_dir, filename)
    with frame_lock:
        cv2.imwrite(filepath, latest_frame)
    logging.info(f"Frame captured: {filename}")
    return jsonify({'success': True, 'filename': filename, 'path': filepath})

@app.route('/adjust_count', methods=['POST'])
def adjust_count():
    global total_count, hourly_net_count
    delta = int(request.json.get('delta', 0))
    total_count      = max(0, total_count + delta)
    hourly_net_count = max(0, hourly_net_count + delta)
    socketio.emit('count_update', {'total': total_count})
    logging.info(f"Manual count adjust: delta={delta}, new total={total_count}")
    return jsonify({'success': True, 'total_count': total_count})

@app.route('/set_count', methods=['POST'])
def set_count():
    global total_count, hourly_net_count
    new_count = int(request.json.get('count', 0))
    if new_count < 0:
        return jsonify({'success': False, 'error': 'Count cannot be negative'}), 400
    old_count        = total_count
    hourly_net_count = max(0, hourly_net_count + (new_count - old_count))
    total_count      = new_count
    socketio.emit('count_update', {'total': total_count})
    logging.info(f"Manual count set: {old_count} → {total_count}")
    return jsonify({'success': True, 'total_count': total_count})

@app.route('/list_files', methods=['GET'])
def list_files():
    file_type = request.args.get('type', 'all')
    files = {'summary': [], 'hourly': [], 'frames': []}
    if file_type in ['all', 'summary'] and os.path.exists(summary_dir):
        for f in sorted(os.listdir(summary_dir), reverse=True):
            if f.endswith('.csv'):
                path = os.path.join(summary_dir, f)
                files['summary'].append({
                    'name': f,
                    'size': os.path.getsize(path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
                })
    if file_type in ['all', 'hourly'] and os.path.exists(hourly_dir):
        for f in sorted(os.listdir(hourly_dir), reverse=True):
            if f.endswith('.csv'):
                path = os.path.join(hourly_dir, f)
                files['hourly'].append({
                    'name': f,
                    'size': os.path.getsize(path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
                })
    if file_type in ['all', 'frames'] and os.path.exists(frames_dir):
        for heat_dir in sorted(os.listdir(frames_dir), reverse=True):
            heat_path = os.path.join(frames_dir, heat_dir)
            if os.path.isdir(heat_path):
                frame_files = [f for f in os.listdir(heat_path) if f.endswith('.jpg')]
                files['frames'].append({
                    'heat': heat_dir,
                    'count': len(frame_files),
                    'files': frame_files[:10]
                })
    return jsonify(files)

@app.route('/download/<file_type>/<path:filename>')
def download_file(file_type, filename):
    try:
        if   file_type == 'summary': directory = summary_dir
        elif file_type == 'hourly':  directory = hourly_dir
        elif file_type == 'frames':
            parts = filename.split('/')
            if len(parts) != 2:
                return jsonify({'error': 'Invalid path format'}), 400
            directory = os.path.join(frames_dir, parts[0])
            filename  = parts[1]
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        full_path = os.path.join(directory, filename)
        if not os.path.abspath(full_path).startswith(os.path.abspath(directory)):
            return jsonify({'error': 'Invalid file path'}), 403
        if not os.path.exists(full_path):
            return jsonify({'error': 'File not found'}), 404
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_heat_package/<heat_id>')
def download_heat_package(heat_id):
    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f in os.listdir(hourly_dir):
                if f.startswith(heat_id) and f.endswith('.csv'):
                    zipf.write(os.path.join(hourly_dir, f), f"hourly/{f}")
            heat_frames_dir = os.path.join(frames_dir, heat_id)
            if os.path.exists(heat_frames_dir):
                for f in os.listdir(heat_frames_dir):
                    if f.endswith('.jpg'):
                        zipf.write(os.path.join(heat_frames_dir, f), f"frames/{f}")
            for f in os.listdir(summary_dir):
                if f.endswith('.csv'):
                    zipf.write(os.path.join(summary_dir, f), f"summary/{f}")
        memory_file.seek(0)
        return send_file(memory_file, mimetype='application/zip',
                         as_attachment=True, download_name=f'{heat_id}_data.zip')
    except Exception as e:
        logging.error(f"Error creating heat package: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_all')
def download_all():
    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f in os.listdir(summary_dir):
                if f.endswith('.csv'):
                    zipf.write(os.path.join(summary_dir, f), f"summary/{f}")
            for f in os.listdir(hourly_dir):
                if f.endswith('.csv'):
                    zipf.write(os.path.join(hourly_dir, f), f"hourly/{f}")
            for heat_dir in os.listdir(frames_dir):
                heat_path = os.path.join(frames_dir, heat_dir)
                if os.path.isdir(heat_path):
                    for f in os.listdir(heat_path):
                        if f.endswith('.jpg'):
                            zipf.write(os.path.join(heat_path, f), f"frames/{heat_dir}/{f}")
        memory_file.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(memory_file, mimetype='application/zip',
                         as_attachment=True, download_name=f'rod_detection_data_{timestamp}.zip')
    except Exception as e:
        logging.error(f"Error creating full package: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

# =========================
# SOCKETIO EVENTS
# =========================
@socketio.on('connect')
def handle_connect():
    logging.info('Client connected')
    emit('connection_response', {'data': 'Connected'})
    with speed_lock:
        hist_len = len(speed_history)
    emit('stats_update', {
        'total_count': total_count,
        'heat_number': heat_number if heat_number else '-',
        'prev_heat': prev_heat_number if prev_heat_number else '-',
        'prev_total': prev_heat_total_count,
        'speed': rod_speed_mm_sec / 1000.0 if speed_display_event.is_set() else 0,
        'speed_samples': hist_len,
        'frame_count': frame_count,
        'sleep_mode': sleep_mode,
        'video_blocked': not video_processing_started.is_set()
    })

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected')

def background_stats_emitter():
    while not stop_event.is_set():
        time.sleep(1)
        with speed_lock:
            hist_len = len(speed_history)
        socketio.emit('stats_update', {
            'total_count': total_count,
            'heat_number': heat_number if heat_number else '-',
            'prev_heat': prev_heat_number if prev_heat_number else '-',
            'prev_total': prev_heat_total_count,
            'speed': rod_speed_mm_sec / 1000.0 if speed_display_event.is_set() else 0,
            'speed_samples': hist_len,
            'frame_count': frame_count,
            'sleep_mode': sleep_mode,
            'video_blocked': not video_processing_started.is_set()
        })

# =========================
# CLEANUP
# =========================
def cleanup():
    global cleanup_done, _current_heat_number, heat_start_time, total_count, _current_heat_metadata, camera_instance
    if cleanup_done:
        return
    cleanup_done = True
    logging.info("[CLEANUP] Shutdown initiated...")
    stop_event.set()
    if inactivity_timer:
        try:
            inactivity_timer.cancel()
        except Exception as e:
            logging.error(f"[CLEANUP] Error cancelling timer: {e}")
    try:
        if _current_heat_number and heat_start_time:
            heat_key = f"{_current_heat_number}_{heat_start_time.isoformat()}"
            with _logged_heats_lock:
                if heat_key not in _logged_heats:
                    _finalize_hourly_log()
                    log_heat_summary(_current_heat_number, heat_start_time, datetime.now(), total_count,
                                     metadata=_current_heat_metadata)
                    _logged_heats[heat_key] = {
                        'start_time': heat_start_time.isoformat(),
                        'logged': True,
                        'count': total_count
                    }
                    logging.info(f"[CLEANUP] Heat {_current_heat_number} finalized with {total_count} rods")
                else:
                    logging.info(f"[CLEANUP] Heat {_current_heat_number} already logged, skipping duplicate")
    except Exception as e:
        logging.error(f"[CLEANUP] Error finalizing logs: {e}")
    if camera_instance:
        try:
            camera_instance.cleanup()
            camera_instance = None
        except Exception as e:
            logging.error(f"[CLEANUP] Camera cleanup error: {e}")
    if gpio_initialized and GPIO_AVAILABLE:
        try:
            GPIO.cleanup()
        except Exception as e:
            logging.error(f"[CLEANUP] GPIO cleanup error: {e}")
    time.sleep(0.5)
    logging.info("[CLEANUP] Application shutdown complete")
    print("\n" + "="*60 + "\nShutdown completed successfully\n" + "="*60)

def signal_handler(sig, frame):
    print("\n" + "="*60 + "\nShutdown signal received (Ctrl+C)\n" + "="*60)
    cleanup()
    time.sleep(1)
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)

# =========================
# MAIN
# =========================
def initialize_camera():
    global video_fps, video_total_frames, camera_instance
    if VIDEO_MODE:
        if not os.path.exists(VIDEO_PATH):
            raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {VIDEO_PATH}")
        video_fps          = cap.get(cv2.CAP_PROP_FPS)
        video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Video loaded: {width}x{height} @ {video_fps:.2f} FPS, {video_total_frames} frames")
        return cap
    else:
        logging.info("Initializing GStreamer camera...")
        camera_instance = GstCamera()
        camera_instance.wait_for_first_frame()
        logging.info("GStreamer camera ready")
        return camera_instance

def run_production():
    global camera_instance, last_camera_health_check
    try:
        init_gpio()
        threading.Thread(target=ssr_worker, daemon=True, name="SSRWorker").start()
        restore_active_heat()
        set_led_state(green=True)
        if not VIDEO_MODE:
            reset_inactivity_timer()
            last_camera_health_check = time.time()
        cap = initialize_camera()
        threading.Thread(target=camera_loop, args=(cap,), daemon=False, name="CameraLoop").start()
        threading.Thread(target=background_stats_emitter, daemon=True, name="StatsEmitter").start()

        print("=" * 60)
        print("BAR Counter System — Weight-Based Yield Model")
        print("=" * 60)
        print(f"Mode:    {'VIDEO FILE' if VIDEO_MODE else 'LIVE CAMERA'}")
        print(f"GPIO:    {'ENABLED' if GPIO_AVAILABLE and gpio_initialized else 'MOCK MODE'}")
        print(f"Output:  {OUTPUT_DIR}")
        print(f"Changes: bloom_length_plan REMOVED")
        print(f"         bloom_weight_kg, density_kg_m3, weight_per_bar_kg ADDED")
        if not video_processing_started.is_set():
            print("VIDEO BLOCKED — Set heat number in web interface to start")
        print("=" * 60)

        socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
    finally:
        cleanup()

if __name__ == "__main__":
    run_production()