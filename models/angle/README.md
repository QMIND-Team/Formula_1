import camera_angle.py

create a 

```
import models.angle.camera_angle
```
```
is_track_view_model_filename = 'models/angle/IsTrackViewModel.h5'
other_view_model_filename = 'models/angle/OtherViewModel.h5'
```
```
camera_angle_classifier = CameraAngleClassifier()
```
```
camera_angle = camera_angle_classifier.getCameraAngle(image)
```