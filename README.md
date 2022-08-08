# Engine RPMs vs. Speed

Plots engine RPMs against car speed and performs k-means clustering to determine the gear number.

How to use:
1. You can collect your own source data with an OBD2 reader that supports exporting to a file.
2. Update the global constants in `engine_rpms_vs_car_speed.py` to match your source data file name and column names.
3. Install the requirements: `pip install --requirement requirements.txt`
4. Run the script: `python engine_rpms_vs_car_speed.py`

Graph of data **without** clustering:
![Without clustering](images/Engine%20RPMs%20vs.%20Car%20Speed%20without%20clustering.png)

Graph of data **with** clustering:
![With clustering](images/Engine%20RPMs%20vs.%20Car%20Speed%20with%20clustering.png)
