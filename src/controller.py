import requests
import time

from camera import ContinuousPhotoCapture
from eyes_recognition import DataProcessor
from run_model_on_crops import EyeClassifier
from pathlib import Path
from datetime import datetime, timedelta

ALERT_MSG_TEMPLATE = "Fatigue Alert!\nOur fatigue detection system has detected signs of employee fatigue. It's crucial to take immediate action to ensure the safety of the employee and those around them.\n" \
                     "- Employee: {}\n" \
                     "- Timestamp: {}\nPlease check on the employee, provide them with a break if needed, and consider reassigning tasks that require high alertness. Remember that employee safety is our top priority.\n" \
                     "Best regards,\nAlertAware"

URL = ''
DEFAULT_BASE_DIR: str = 'resources'


def turn_on_alert(employee_name, timestamp):
    alert_msg = ALERT_MSG_TEMPLATE.format(employee_name, timestamp)
    response = requests.post(
        URL,
        data={'text': alert_msg}
    )


class Controller:
    def __init__(self):
        self.camera = ContinuousPhotoCapture()
        self.data_processor = DataProcessor()
        self.detector = EyeClassifier()

    def run(self):
        employee_name = input("Please enter the employee's name: ")  # Get employee's name
        running = True  # Flag to control the loop
        detector_triggered_time = None

        while running:
            self.camera.capture_photos(photos_to_capture=3)
            directory_path: Path = Path(DEFAULT_BASE_DIR)
            self.data_processor.process_data(directory_path)

            if self.detector.is_fatigue():
                if detector_triggered_time is None:
                    detector_triggered_time = datetime.now()
                    formatted_time = detector_triggered_time.strftime("%Y-%m-%d %H:%M:%S")
                    turn_on_alert(employee_name, formatted_time)
                else:
                    current_time = datetime.now()
                    elapsed_time = current_time - detector_triggered_time
                    if elapsed_time >= timedelta(minutes=1):
                        formatted_time = detector_triggered_time.strftime("%Y-%m-%d %H:%M:%S")
                        turn_on_alert(employee_name, formatted_time)
                        detector_triggered_time = datetime.now()  # reset

            time.sleep(2)


controller = Controller()
controller.run()
