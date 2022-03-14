Uses OpenCV, dlib, Google Chrome API


CONTROLS:
Enable NMJM:: hold tongue out

Scrolling: (scroll speed is relative to distance from middle)
Scroll up: tongue up towards nose 
Scroll down: tongue down towards chin
Scroll right/left: tongue right/left towards cheeks

Clicking:
Right click: blink right eye
Left click: blink left eye

Holding:
Right click hold: close right eye
Left click hold: close left eye

Move cursor: look at where you want the cursor to be

Blink Suppression: if the user blinks both eyes, it does not register as a click

FILES:
nmjm.py: interfaces with Chrome API
-- outputs all desired mouse inputs

tongue.py: detects presence of a tongue, position relative to the default position
-- outputs a boolean?, the default position, the relative current position

eyes.py: detects when an eye is closed, gaze detection
-- outputs a boolean for each eye, outputs coordinates for the cursor