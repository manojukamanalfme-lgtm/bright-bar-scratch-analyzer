#!/usr/bin/env python3
"""
Basler Camera Recording Application for Jetson
Features:
- GUI with start/stop recording buttons
- Date-wise folder organization
- GStreamer pipeline with pylonsrc for Basler cameras
"""

import os
import sys

# Set environment variables before importing GStreamer modules
# This ensures the packaged executable can find plugins and libraries
os.environ['GST_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'
os.environ['PYLON_ROOT'] = '/opt/pylon'
os.environ['LD_LIBRARY_PATH'] = '/opt/pylon/lib:/usr/lib/aarch64-linux-gnu'
os.environ['GST_PLUGIN_SYSTEM_PATH'] = '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'

import tkinter as tk
from tkinter import ttk, messagebox
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import datetime
import threading
import time


class BaslerRecorder:
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)
        
        # Recording state
        self.is_recording = False
        self.pipeline = None
        self.main_loop = None
        self.loop_thread = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Basler Camera Recorder")
        self.root.geometry("400x300")
        self.root.resizable(True, True)
        
        # Create GUI elements
        self.setup_gui()
        
        # Ensure proper cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_gui(self):
        """Setup the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Basler Camera Recorder", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.status_label = ttk.Label(status_frame, text="Ready to record", 
                                     font=("Arial", 10))
        self.status_label.grid(row=0, column=0)
        
        self.recording_indicator = ttk.Label(status_frame, text="●", 
                                           foreground="gray", font=("Arial", 20))
        self.recording_indicator.grid(row=0, column=1, padx=(10, 0))
        
        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        self.start_button = ttk.Button(button_frame, text="Start Recording", 
                                      command=self.start_recording, width=15)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Recording", 
                                     command=self.stop_recording, width=15, 
                                     state="disabled")
        self.stop_button.grid(row=0, column=1)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Output directory
        ttk.Label(settings_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W)
        self.output_dir = tk.StringVar(value="recordings")
        ttk.Entry(settings_frame, textvariable=self.output_dir, width=30).grid(row=0, column=1, padx=(10, 0))
        
        # Recording info
        info_frame = ttk.LabelFrame(main_frame, text="Recording Info", padding="10")
        info_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.recording_time_label = ttk.Label(info_frame, text="Recording Time: 00:00:00")
        self.recording_time_label.grid(row=0, column=0, sticky=tk.W)
        
        self.current_file_label = ttk.Label(info_frame, text="Current File: None")
        self.current_file_label.grid(row=1, column=0, sticky=tk.W)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def create_date_folder(self):
        """Create a folder based on current date"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        folder_path = os.path.join(self.output_dir.get(), today)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        return folder_path
        
    def generate_filename(self):
        """Generate filename with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H-%M-%S")
        date_folder = self.create_date_folder()
        filename = os.path.join(date_folder, f"recording_{timestamp}.mp4")
        return filename
        
    def create_pipeline(self, filename):
        """Create GStreamer pipeline for Basler camera recording"""
        # Escape the filename properly
        escaped_filename = filename.replace('\\', '\\\\')
        
        pipeline_str = (
            'pylonsrc user-set=UserSet2 ! '
            'videoconvert ! '
            'video/x-raw, format=BGR, width=1280, height=720 ! '
            'videoconvert ! '
            'video/x-raw, format=I420 ! '
            'x264enc bitrate=5000 speed-preset=medium tune=zerolatency ! '
            'video/x-h264, profile=baseline ! '
            'mp4mux faststart=true ! '
            f'filesink location="{escaped_filename}" sync=false'
        )
        
        try:
            print(f"Creating pipeline: {pipeline_str}")
            pipeline = Gst.parse_launch(pipeline_str)
            return pipeline
        except Exception as e:
            error_msg = f"Failed to create pipeline: {str(e)}"
            print(error_msg)
            messagebox.showerror("Pipeline Error", error_msg)
            return None
            
    def start_recording(self):
        """Start recording"""
        if self.is_recording:
            return
            
        try:
            # Generate filename
            filename = self.generate_filename()
            
            # Create pipeline
            self.pipeline = self.create_pipeline(filename)
            if not self.pipeline:
                return
                
            # Set up bus for message handling
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                error_msg = "Failed to start recording pipeline. Check camera connection and permissions."
                print(error_msg)
                messagebox.showerror("Error", error_msg)
                return
                
            # Update state
            self.is_recording = True
            self.start_time = time.time()
            
            # Update GUI
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="Recording...")
            self.recording_indicator.config(foreground="red")
            self.current_file_label.config(text=f"Current File: {os.path.basename(filename)}")
            
            # Start main loop in separate thread
            self.main_loop = GLib.MainLoop()
            self.loop_thread = threading.Thread(target=self.main_loop.run)
            self.loop_thread.daemon = True
            self.loop_thread.start()
            
            # Start timer update
            self.update_timer()
            
            print(f"Recording started: {filename}")
            
        except Exception as e:
            error_msg = f"Failed to start recording: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.reset_gui_state()
            
    def stop_recording(self):
        """Stop recording"""
        if not self.is_recording:
            return
            
        try:
            # Send EOS event and wait for it to be processed
            if self.pipeline:
                print("Sending EOS event...")
                self.pipeline.send_event(Gst.Event.new_eos())
                
                # Wait for EOS to be processed (important for MP4 finalization)
                bus = self.pipeline.get_bus()
                
                # Wait for EOS message with timeout
                timeout = 5 * Gst.SECOND  # 5 seconds timeout
                msg = bus.timed_pop_filtered(timeout, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                
                if msg:
                    if msg.type == Gst.MessageType.EOS:
                        print("EOS received, file should be properly finalized")
                    elif msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        print(f"Error during EOS: {err}")
                else:
                    print("Timeout waiting for EOS")
                
                # Now set pipeline to NULL state
                self.pipeline.set_state(Gst.State.NULL)
                
                # Wait for state change to complete
                ret, state, pending = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
                if ret == Gst.StateChangeReturn.SUCCESS:
                    print("Pipeline stopped successfully")
                
            # Stop main loop
            if self.main_loop:
                self.main_loop.quit()
                
            # Wait for loop thread to finish
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=2)
                
            # Update state
            self.is_recording = False
            
            # Update GUI
            self.reset_gui_state()
            
            print("Recording stopped and file finalized")
            
        except Exception as e:
            error_msg = f"Failed to stop recording: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.reset_gui_state()
            
    def reset_gui_state(self):
        """Reset GUI to initial state"""
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Ready to record")
        self.recording_indicator.config(foreground="gray")
        self.current_file_label.config(text="Current File: None")
        self.recording_time_label.config(text="Recording Time: 00:00:00")
        
    def update_timer(self):
        """Update recording timer"""
        if self.is_recording:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            time_str = f"Recording Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
            self.recording_time_label.config(text=time_str)
            
            # Schedule next update
            self.root.after(1000, self.update_timer)
            
    def on_message(self, bus, message):
        """Handle GStreamer messages"""
        t = message.type
        
        if t == Gst.MessageType.EOS:
            print("End-of-stream received - recording will be finalized")
            # Don't call stop_recording here as it would create a loop
            # The EOS is handled in stop_recording method
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            error_msg = f"GStreamer Error: {err}"
            print(f"Error: {err}, {debug}")
            self.root.after(0, lambda: messagebox.showerror("Recording Error", error_msg))
            self.root.after(0, self.force_stop_recording)
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Warning: {warn}, {debug}")
        elif t == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            if message.src == self.pipeline:
                print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
                
    def force_stop_recording(self):
        """Force stop recording without waiting for EOS (used for error cases)"""
        if not self.is_recording:
            return
            
        try:
            print("Force stopping recording due to error...")
            
            # Directly set pipeline to NULL without waiting for EOS
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                
            # Stop main loop
            if self.main_loop:
                self.main_loop.quit()
                
            # Wait for loop thread to finish
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=1)
                
            # Update state
            self.is_recording = False
            
            # Update GUI
            self.reset_gui_state()
            
            print("Recording force stopped")
            
        except Exception as e:
            print(f"Error during force stop: {e}")
            self.reset_gui_state()
            
    def on_closing(self):
        """Handle window closing"""
        if self.is_recording:
            self.stop_recording()
            
        # Wait a moment for cleanup
        time.sleep(0.5)
        self.root.destroy()
        
    def run(self):
        """Run the application"""
        self.root.mainloop()


def main():
    """Main function"""
    try:
        # Print environment info for debugging
        print("Environment Variables:")
        print(f"GST_PLUGIN_PATH: {os.environ.get('GST_PLUGIN_PATH', 'Not set')}")
        print(f"PYLON_ROOT: {os.environ.get('PYLON_ROOT', 'Not set')}")
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        
        app = BaslerRecorder()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()
