__all__ = ['HOUSE_PRESSURE', 'INLET_PRESSURE', 'OUTLET_PRESSURE', 'lst_exclude_dates_dec_19',
           'lst_exclude_dates_jan_20', 'lst_exclude_dates_common', 'lst_exclude_dates_pump1',
          'lst_exclude_dates_pump2', 'lst_exclude_dates_pump3', 'lst_exclude_dates_pump4',
          'lst_exclude_dates_pump5', 'lst_exclude_dates_pump6']

# Cell
import os
import numpy as np
import pandas as pd
import plotly
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
from plotly.offline import init_notebook_mode
import plotly.io as pio

# Cell
HOUSE_PRESSURE = "HousePressure"
INLET_PRESSURE = "InletPressure"
OUTLET_PRESSURE = "OutletPressure"

lst_exclude_dates_dec_19 = [(3,12,2019),(4,12,2019),(5,12,2019),(6,12,2019),(7,12,2019),(8,12,2019),(9,12,2019),
                    (10,12,2019),(11,12,2019),(12,12,2019),(13,12,2019)]
lst_exclude_dates_jan_20 = [(13,1,2020),(14,1,2020),(15,1,2020),(16,1,2020),(17,1,2020),(18,1,2020),(19,1,2020),
                    (20,1,2020),(21,1,2020),(22,1,2020),(23,1,2020)]

lst_exclude_plant_maintenance_dates = [(30,11,2019), (18,12,2019)] 

lst_exclude_dates_common = lst_exclude_dates_dec_19 + lst_exclude_dates_jan_20 + lst_exclude_plant_maintenance_dates;
# 3rd April is prob plant maintenance
lst_exclude_dates_pump1 = lst_exclude_dates_common 
lst_exclude_dates_pump2 = lst_exclude_dates_common 
lst_exclude_dates_pump3 = lst_exclude_dates_common 
lst_exclude_dates_pump4 = lst_exclude_dates_common 
lst_exclude_dates_pump5 = lst_exclude_dates_common 
lst_exclude_dates_pump6 = lst_exclude_dates_common 




