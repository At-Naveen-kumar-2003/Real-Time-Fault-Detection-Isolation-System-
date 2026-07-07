# =============================================================================
#  gpio_controller.py — Buzzer driver for Raspberry Pi
#
#  Active buzzer connected to BUZZER_PIN (BCM numbering).
#  Beep pattern depends on attack/fault class:
#    Cyber attacks (class 6-10) : 3 short beeps
#    Genuine faults (class 1-5) : 1 long beep
#    Normal (class 0)           : no beep
# =============================================================================

import threading
import time

try:
    import RPi.GPIO as GPIO
    _GPIO = True
except ImportError:
    _GPIO = False
    print("[GPIO] RPi.GPIO not available — buzzer disabled (non-Pi environment)")

from config import BUZZER_PIN, GENUINE_FAULT_CLASSES, ATTACK_CLASSES


class GPIOController:

    def __init__(self):
        if _GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
                print(f"[GPIO] Buzzer ready on GPIO{BUZZER_PIN}")
            except Exception as e:
                print(f"[GPIO] Setup error: {e}")

    def alert(self, attack_class: int):
        """Trigger buzzer pattern based on attack/fault class."""
        if attack_class == 0:
            return
        if attack_class in ATTACK_CLASSES:
            # Cyber attack — 3 short beeps
            threading.Thread(
                target=self._beep,
                args=(3, 0.15, 0.1),
                daemon=True
            ).start()
        elif attack_class in GENUINE_FAULT_CLASSES:
            # Genuine fault — 1 long beep
            threading.Thread(
                target=self._beep,
                args=(1, 0.6, 0.0),
                daemon=True
            ).start()

    def _beep(self, count: int, on_time: float, off_time: float):
        if not _GPIO:
            print(f"[BUZZER] BEEP x{count} (simulated)")
            return
        try:
            for _ in range(count):
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                time.sleep(on_time)
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                if off_time > 0:
                    time.sleep(off_time)
        except Exception as e:
            print(f"[GPIO] Beep error: {e}")

    def cleanup(self):
        if _GPIO:
            try:
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass
