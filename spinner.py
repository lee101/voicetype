#!/usr/bin/python3
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, Gdk
import cairo
import math
import os

class Spinner(Gtk.Window):
    def __init__(self):
        super().__init__(title="VoiceType")
        self.set_default_size(200, 80)
        self.set_keep_above(True)
        self.set_decorated(False)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_resizable(False)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)
        self.set_app_paintable(True)

        self.angle = 0
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(200, 80)
        self.drawing_area.connect("draw", self.on_draw)
        self.add(self.drawing_area)

        GLib.timeout_add(50, self.update)

    def update(self):
        if not os.path.exists('/tmp/voicetype-spinner-run'):
            Gtk.main_quit()
            return False
        self.angle += 0.2
        self.drawing_area.queue_draw()
        return True

    def on_draw(self, widget, cr):
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()

        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0.1, 0.1, 0.15, 0.92)
        self.draw_rounded_rect(cr, 0, 0, width, height, 12)
        cr.fill()

        cr.set_operator(cairo.OPERATOR_OVER)
        cx, cy = width / 2, height / 2 - 8
        radius = 16

        for i in range(12):
            angle = i * math.pi / 6 + self.angle
            alpha = (i + 1) / 12.0
            cr.set_source_rgba(0.3, 0.7, 1.0, alpha)
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius
            cr.arc(x, y, 3, 0, 2 * math.pi)
            cr.fill()

        cr.set_source_rgba(1, 1, 1, 0.9)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(11)
        text = "Transcribing..."
        extents = cr.text_extents(text)
        cr.move_to((width - extents.width) / 2, height - 12)
        cr.show_text(text)

        return False

    def draw_rounded_rect(self, cr, x, y, w, h, r):
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -math.pi/2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi/2)
        cr.arc(x + r, y + h - r, r, math.pi/2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 3*math.pi/2)
        cr.close_path()

if __name__ == "__main__":
    with open('/tmp/voicetype-spinner-run', 'w') as f:
        f.write('1')
    win = Spinner()
    win.show_all()
    Gtk.main()
