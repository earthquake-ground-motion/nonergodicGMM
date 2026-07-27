"""
Download utility functions
"""
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import numpy as np


# YEARS = np.hstack([np.array([1969]), np.arange(1990, 2027, 1)])


#def get_start_end_times(min_year: int = 1967, max_year: Optional[int] = None) -> List:
#    """Get the start and end-times and folder names for download by month
#    """
#    now = datetime.now()
#    max_year = max_year + 1 if max_year else now.year + 1
#
#    if min_year < 1990:
#       years = np.hstack([np.array([min_year]), np.arange(1990, max_year + 1, 1)])
#    else:
#       years = np.arange(min_year, max_year + 1, 1)
#    start_end_times = []
#    for i in range(0, len(years) - 1):
#        start_year, end_year = years[i], years[i + 1]
#        if start_year < 1990:
#            start_time = "{:g}-01-01T00:00:00".format(start_year)
#            end_time = "{:g}-01-01T00:00:00".format(end_year)
#            folder_name = "{:g}_1989".format(start_year)
#            start_end_times.append((start_time, end_time, folder_name))
#        else:
#            for month in range(1, 13):
#                if (start_year == now.year) and (month > now.month):
#                    # Date in the future
#                    continue
#                month_string = str(month).zfill(2)
#                start_time = "{:g}-{:s}-01T00:00:00".format(start_year, month_string)
#                if month == 12:
#                    end_time = "{:g}-01-01T00:00:00".format(start_year + 1)
#                else:
#                    end_time = "{:g}-{:s}-01T00:00:00".format(start_year,
#                                                              str(month + 1).zfill(2))
#                folder_name = "{:g}/{:s}".format(start_year, month_string)
#                start_end_times.append((start_time, end_time, folder_name))
#    return start_end_times


DEFAULT_START_DATE = datetime(1967, 1, 1, 0, 0, 0)


def get_start_end_times(starttime: Optional[datetime] = None,
                        endtime: Optional[datetime] = None) -> List:
    """Get the start and end-times and folder names for download by month. For years before
    1990 just one time range and output folder is provided. Otherwise the time ranges and
    outputs are monthly
    """
    now = datetime.now()
    if not starttime:
        starttime = datetime(1967, 1, 1, 0, 0, 0)  # Year of earliest event in ESM
    if not endtime:
        endtime = now.date()
    start_end_times = []
    if starttime.year < 1990:
        years = np.hstack([np.array([starttime.year]), np.arange(1990, endtime.year + 1, 1)])
    else:
        years = np.arange(starttime.year, endtime.year + 1, 1)
    for i, yr in enumerate(years):
        if yr < 1990:
            start_end_times.append((
                f"{yr}-01-01T00:00:00",
                f"{years[i + 1]:g}-01-01T00:00:00",
                str(Path(f"{starttime.year}_1989"))
                )
            )
        else:
            for month in range(1, 13):
                if (yr == now.year) and (month > now.month):
                    # Date in future, so skipping
                    continue
                start_string = f"{yr:g}-{str(month).zfill(2)}-01T00:00:00"
                if month == 12:
                    end_string = f"{str(yr + 1)}-01-01T00:00:00"
                else:
                    end_string = f"{yr}-{str(month + 1).zfill(2)}-01T00:00:00"
                folder_name = str(Path(str(yr)) / str(month).zfill(2))
                start_end_times.append((start_string, end_string, folder_name))
    return start_end_times
