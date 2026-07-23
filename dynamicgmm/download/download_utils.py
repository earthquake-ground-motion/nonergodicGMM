"""
Download utility functions
"""
from datetime import datetime
from typing import List, Optional
import numpy as np


# YEARS = np.hstack([np.array([1969]), np.arange(1990, 2027, 1)])


def get_start_end_times(min_year: int = 1967, max_year: Optional[int] = None) -> List:
    """Get the start and end-times and folder names for download by month
    """
    now = datetime.now()
    max_year = max_year + 1 if max_year else now.year + 1

    if min_year < 1990:
       years = np.hstack([np.array([min_year]), np.arange(1990, max_year + 1, 1)])
    else:
       years = np.arange(min_year, max_year + 1, 1)
    start_end_times = []
    for i in range(0, len(years) - 1):
        start_year, end_year = years[i], years[i + 1]
        if start_year < 1990:
            start_time = "{:g}-01-01T00:00:00".format(start_year)
            end_time = "{:g}-01-01T00:00:00".format(end_year)
            folder_name = "{:g}_1989".format(start_year)
            start_end_times.append((start_time, end_time, folder_name))
        else:
            for month in range(1, 13):
                if (start_year == now.year) and (month > now.month):
                    # Date in the future
                    continue
                month_string = str(month).zfill(2)
                start_time = "{:g}-{:s}-01T00:00:00".format(start_year, month_string)
                if month == 12:
                    end_time = "{:g}-01-01T00:00:00".format(start_year + 1)
                else:
                    end_time = "{:g}-{:s}-01T00:00:00".format(start_year,
                                                              str(month + 1).zfill(2))
                folder_name = "{:g}/{:s}".format(start_year, month_string)
                start_end_times.append((start_time, end_time, folder_name))
    return start_end_times
