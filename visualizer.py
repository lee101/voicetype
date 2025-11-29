#!/usr/bin/python3
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf
import cairo
import struct
import os
import sys
import math
import subprocess
import threading
import queue

class WaveformVisualizer(Gtk.Window):
    def __init__(self):
        super().__init__(title="VoiceType")
        self.set_default_size(400, 120)
        self.set_keep_above(True)
        self.set_decorated(False)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_resizable(False)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)
        self.set_app_paintable(True)

        self.cancelled = False
        self.waveform_data = [0.0] * 60
        self.audio_queue = queue.Queue()

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(400, 120)
        self.drawing_area.connect("draw", self.on_draw)
        self.add(self.drawing_area)

        self.connect("key-press-event", self.on_key_press)
        self.connect("destroy", self.on_destroy)

        GLib.timeout_add(50, self.update_waveform)

        self.start_audio_monitor()

    def start_audio_monitor(self):
        def monitor():
            try:
                proc = subprocess.Popen(
                    ['parec', '--raw', '--format=s16le', '--rate=8000', '--channels=1'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                while not self.cancelled:
                    data = proc.stdout.read(256)
                    if data:
                        samples = struct.unpack(f'{len(data)//2}h', data)
                        peak = max(abs(s) for s in samples) / 32768.0 if samples else 0
                        self.audio_queue.put(peak)
                proc.terminate()
            except:
                pass

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def update_waveform(self):
        if self.cancelled:
            return False

        if not os.path.exists('/tmp/voicetype-visualizer-run'):
            self.cancelled = True
            Gtk.main_quit()
            return False

        try:
            while True:
                peak = self.audio_queue.get_nowait()
                self.waveform_data.pop(0)
                self.waveform_data.append(peak)
        except queue.Empty:
            pass

        self.drawing_area.queue_draw()
        return True

    def on_draw(self, widget, cr):
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()

        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0.1, 0.1, 0.15, 0.92)
        self.draw_rounded_rect(cr, 0, 0, width, height, 16)
        cr.fill()

        cr.set_operator(cairo.OPERATOR_OVER)
        cr.set_source_rgba(0.3, 0.6, 1.0, 0.9)
        cr.set_line_width(3)

        num_bars = 40
        bar_width = (width - 60) / num_bars
        center_y = height / 2 - 5

        for i in range(num_bars):
            idx = int(i * len(self.waveform_data) / num_bars)
            level = self.waveform_data[idx] if idx < len(self.waveform_data) else 0

            x = 30 + i * bar_width
            bar_height = max(6, level * (height - 60))

            gradient = cairo.LinearGradient(x, center_y - bar_height/2, x, center_y + bar_height/2)
            gradient.add_color_stop_rgba(0, 0.3, 0.8, 1.0, 1.0)
            gradient.add_color_stop_rgba(0.5, 0.5, 0.4, 1.0, 1.0)
            gradient.add_color_stop_rgba(1, 0.3, 0.8, 1.0, 1.0)
            cr.set_source(gradient)

            w = max(bar_width * 0.7, 4)
            cr.rectangle(x, center_y - bar_height/2, w, bar_height)
            cr.fill()

        cr.set_source_rgba(1, 1, 1, 0.9)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(12)

        text = "Recording... [Enter] Done  [ESC] Cancel"
        extents = cr.text_extents(text)
        cr.move_to((width - extents.width) / 2, height - 12)
        cr.show_text(text)

        cr.set_source_rgba(1.0, 0.3, 0.3, 1.0)
        cr.arc(20, 20, 6, 0, 2 * math.pi)
        cr.fill()

        return False

    def draw_rounded_rect(self, cr, x, y, w, h, r):
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -math.pi/2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi/2)
        cr.arc(x + r, y + h - r, r, math.pi/2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 3*math.pi/2)
        cr.close_path()

    def on_key_press(self, widget, event):
        if event.keyval == Gdk.KEY_Escape:
            self.cancelled = True
            with open('/tmp/voicetype-cancelled', 'w') as f:
                f.write('1')
            if os.path.exists('/tmp/voicetype-visualizer-run'):
                os.remove('/tmp/voicetype-visualizer-run')
            Gtk.main_quit()
            return True
        elif event.keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            if os.path.exists('/tmp/voicetype-visualizer-run'):
                os.remove('/tmp/voicetype-visualizer-run')
            Gtk.main_quit()
            return True
        return False

    def on_destroy(self, widget):
        self.cancelled = True

if __name__ == "__main__":
    with open('/tmp/voicetype-visualizer-run', 'w') as f:
        f.write('1')

    if os.path.exists('/tmp/voicetype-cancelled'):
        os.remove('/tmp/voicetype-cancelled')

    win = WaveformVisualizer()
    win.show_all()
    Gtk.main()
