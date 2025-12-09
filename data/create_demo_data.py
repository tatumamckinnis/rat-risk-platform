import pandas as pd
import numpy as np
from pathlib import Path

Path('data/processed').mkdir(parents=True, exist_ok=True)

ZIP_CODES = {
    'Manhattan': ['10001', '10002', '10003', '10011', '10012'],
    'Brooklyn': ['11201', '11211', '11215', '11222', '11225'],
    'Queens': ['11101', '11354', '11355', '11373', '11375'],
    'Bronx': ['10451', '10452', '10453', '10458', '10467'],
    'Staten Island': ['10301', '10304', '10305', '10306', '10314'],
}

dates = pd.date_range('2022-01-01', '2024-12-01', freq='MS')
records = []
np.random.seed(42)

for borough, zips in ZIP_CODES.items():
    for zip_code in zips:
        base = np.random.uniform(5, 25)
        for date in dates:
            seasonal = 1 + 0.4 * np.sin((date.month - 3) * 3.14159 / 6)
            complaints = max(0, int(base * seasonal + np.random.normal(0, 3)))
            records.append({
                'zip_code': zip_code, 'borough': borough, 'date': date,
                'complaint_count': complaints,
                'restaurant_violations_nearby': max(0, int(complaints * 0.3)),
                'building_age_mean': np.random.uniform(40, 70),
                'old_building_pct': np.random.uniform(0.3, 0.6),
            })

pd.DataFrame(records).to_csv('data/processed/master_dataset.csv', index=False)
print('Demo data created!')
