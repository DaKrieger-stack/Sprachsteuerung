from machine import Pin

import config


class OutputControl:
    def __init__(self):
        self._blinker_links = Pin(config.PIN_BLINKER_LINKS, Pin.OUT, value=0)
        self._blinker_rechts = Pin(config.PIN_BLINKER_RECHTS, Pin.OUT, value=0)
        self._innenlicht = Pin(config.PIN_INNENLICHT, Pin.OUT, value=0)
        self._licht = Pin(config.PIN_LICHT, Pin.OUT, value=0)

    def blinker_links(self):
        self._blinker_links.value(1)
        self._blinker_rechts.value(0)

    def blinker_rechts(self):
        self._blinker_rechts.value(1)
        self._blinker_links.value(0)

    def innenlicht_an(self):
        self._innenlicht.value(1)

    def innenlicht_aus(self):
        self._innenlicht.value(0)

    def licht_an(self):
        self._licht.value(1)

    def licht_aus(self):
        self._licht.value(0)

