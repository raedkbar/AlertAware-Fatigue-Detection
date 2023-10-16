import time
import requests

from pathlib import Path
from datetime import datetime, timedelta
from src.camera_comp.camera import ContinuousPhotoCapture
from src.detection_comp.eyes_recognition import DataProcessor
from src.detection_comp.run_model_on_crops import EyeClassifier

# Message template for the fatigue alert
ALERT_MSG_TEMPLATE = "Fatigue Alert!\nOur fatigue detection system has detected signs of employee fatigue. It's crucial to take immediate action to ensure the safety of the employee and those around them.\n" \
                     "- Employee: {}\n" \
                     "- Timestamp: {}\nPlease check on the employee, provide them with a break if needed, and consider reassigning tasks that require high alertness. Remember that employee safety is our top priority.\n" \
                     "Best regards,\nAlertAware"

# Define the URL to send the alert message (replace with the actual URL)
URL = ''

# Default base directory for resources
DEFAULT_BASE_DIR: str = 'resources'


def turn_on_alert(employee_name, timestamp):
    """
    Send a fatigue alert message to a specified URL.

    Args:
        employee_name (str): Name of the fatigued employee.
        timestamp (str): Timestamp when fatigue was detected.
    """
    alert_msg = ALERT_MSG_TEMPLATE.format(employee_name, timestamp)
    response = requests.post(
        URL,
        data={'text': alert_msg}
    )


class Controller:
    """
    Controller class to manage the fatigue detection system.
    """

    def __init__(self):
        self.camera = ContinuousPhotoCapture()
        self.data_processor = DataProcessor()
        self.detector = EyeClassifier()

    def run(self):
        """
        Run the fatigue detection system.
        """
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
                        detector_triggered_time = datetime.now()  # Reset the detector_triggered_time

            time.sleep(2)


# Create an instance of the Controller class and run the system
controller = Controller()
controller.run()
