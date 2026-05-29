from machine import CAN, Pin

import config


class OutputControl:
    def __init__(self):
        self._state = 0
        self._can = CAN(
            config.CAN_ID,
            tx=Pin(config.CAN_TX_PIN),
            rx=Pin(config.CAN_RX_PIN),
            baudrate=config.CAN_BAUDRATE,
            mode=CAN.NORMAL,
        )

    def apply(self, command_label):
        if command_label == config.COMMAND_ON:
            self.licht_an()
        elif command_label == config.COMMAND_OFF:
            self.licht_aus()

    def _send_light_state(self, payload):
        self._can.send(payload, config.CAN_ARB_ID_LIGHT)

    def licht_an(self):
        self._send_light_state(config.CAN_PAYLOAD_LIGHT_ON)
        self._state = 1

    def licht_aus(self):
        self._send_light_state(config.CAN_PAYLOAD_LIGHT_OFF)
        self._state = 0

    def state(self):
        return self._state
