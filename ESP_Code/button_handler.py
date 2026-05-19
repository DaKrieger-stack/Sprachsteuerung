import time
from machine import Pin


class ButtonHandler:
    def __init__(self, pin_no, debounce_ms, active_low=True):
        pull = Pin.PULL_UP if active_low else Pin.PULL_DOWN
        self._pin = Pin(pin_no, Pin.IN, pull)
        self._debounce_ms = debounce_ms
        self._active_low = active_low
        self._enabled = True
        self._stable_state = self._read_raw()
        self._last_raw = self._stable_state
        self._last_edge_ms = time.ticks_ms()
        self._armed = True

    def _read_raw(self):
        value = self._pin.value()
        return 1 if (value == 0 if self._active_low else value == 1) else 0

    def set_enabled(self, enabled):
        self._enabled = enabled
        if not enabled:
            self._stable_state = self._read_raw()
            self._last_raw = self._stable_state
            self._armed = False
        else:
            self._armed = True

    def update(self):
        if not self._enabled:
            return False

        now = time.ticks_ms()
        raw = self._read_raw()
        if raw != self._last_raw:
            self._last_raw = raw
            self._last_edge_ms = now
            return False

        if raw != self._stable_state and time.ticks_diff(now, self._last_edge_ms) >= self._debounce_ms:
            self._stable_state = raw
            if raw == 1 and self._armed:
                self._armed = False
                return True
            if raw == 0:
                self._armed = True

        return False

