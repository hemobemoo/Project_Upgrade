import numpy as np

class MetadataAPIProcessor:
    def __init__(self):
        self.scaler_mean = np.array([59.62532194, 167.54743848, 71.74429069])
        self.scaler_std  = np.array([16.91161878, 7.84168442, 11.60366481])

    def process_metadata_for_api(self, age, sex, height, weight):
        # Robust defaults so the app never crashes on missing fields
        age    = float(age)    if age    is not None else 60.0
        height = float(height) if height is not None else 170.0
        weight = float(weight) if weight is not None else 70.0

        raw_vals   = np.array([age, height, weight])
        scaled_vals = (raw_vals - self.scaler_mean) / (self.scaler_std + 1e-7)

        final_meta = np.array([[
            scaled_vals[0],
            float(sex),
            scaled_vals[1],
            scaled_vals[2]
        ]], dtype=np.float32)

        return final_meta