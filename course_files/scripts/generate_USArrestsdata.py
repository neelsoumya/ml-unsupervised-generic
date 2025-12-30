"""
generate_USArrestsdata.py

This script creates a pandas DataFrame containing the USArrests dataset, which includes statistics on violent crime rates (Murder, Assault, UrbanPop, Rape) for each US state. The data is then saved as 'USArrests.csv' for use in data analysis or machine learning exercises.
"""

import pandas as pd

# Load the US Arrests data
us_arrests_data = {
    'Murder': [13.2, 10.0, 8.1, 8.8, 9.0, 7.9, 3.3, 5.9, 15.4, 17.4, 5.3, 2.6, 10.4, 7.2, 2.2, 6.0, 9.7, 15.4, 2.1, 11.3, 2.7, 16.5, 9.0, 6.0, 4.3, 12.1, 2.5, 0.8, 7.1, 11.4, 13.0, 6.8, 4.5, 12.2, 9.7, 6.6, 3.4, 14.4, 10.9, 11.1, 13.0, 6.3, 3.4, 1.4, 12.5, 9.0, 6.9, 4.2, 12.9, 5.3],
    'Assault': [236, 263, 294, 190, 276, 204, 110, 238, 335, 211, 46, 120, 249, 113, 56, 115, 109, 249, 83, 300, 72, 251, 120, 151, 96, 255, 53, 62, 178, 188, 337, 142, 169, 332, 293, 336, 86, 279, 159, 285, 337, 51, 120, 82, 257, 275, 243, 233, 337, 80],
    'UrbanPop': [58, 48, 80, 50, 91, 78, 77, 72, 80, 60, 83, 54, 62, 72, 66, 75, 82, 64, 39, 67, 70, 53, 65, 56, 51, 83, 51, 39, 71, 70, 91, 72, 74, 68, 87, 82, 45, 70, 53, 72, 80, 60, 67, 53, 95, 73, 76, 58, 90, 44],
    'Rape': [21.2, 44.5, 31.0, 19.5, 40.6, 38.7, 11.1, 15.8, 31.9, 25.8, 20.2, 14.2, 24.0, 21.0, 11.3, 18.0, 16.3, 22.2, 7.8, 27.8, 16.1, 17.5, 16.8, 15.6, 9.0, 35.1, 8.1, 11.6, 26.8, 23.0, 16.1, 21.4, 24.8, 18.0, 26.2, 20.2, 12.8, 22.5, 18.5, 25.5, 22.9, 11.6, 18.9, 8.3, 28.9, 24.9, 21.9, 17.1, 26.4, 12.8]
}

# State names
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

# Create DataFrame
df = pd.DataFrame(us_arrests_data, index=states)

# Save
df.to_csv("USArrests.csv")