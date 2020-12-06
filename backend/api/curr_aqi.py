import requests
import numpy as np
import pandas as pd
import time

def show_aqi(loc_list: list) -> pd.DataFrame:
    token = "4b73e0872f9eecfaf9d32580b1b378b250cbb77e"

    loc_df = pd.DataFrame({"county": loc_list})

    for i in range(loc_df.shape[0]):

        if i % 950 == 0:
            time.sleep(1)

        location = loc_df.iloc[i]["county"]
        url = f"https://api.waqi.info/feed/{location}/?token={token}"
        response = requests.get(url).json()

        # Update AQI
        try:
            loc_df.at[i, "AQI"] = response["data"]["aqi"]
        except:
            pass
        
        # Update CO
        try:
             loc_df.at[i, "CO"] = response["data"]["iaqi"]["co"]["v"]
        except:
            pass

        # Update NO2
        try:
             loc_df.at[i, "NO2"] = response["data"]["iaqi"]["no2"]["v"]
        except:
            pass

        # Update Ozone
        try:
             loc_df.at[i, "Ozone"] = response["data"]["iaqi"]["o3"]["v"]
        except:
            pass

        # Update PM10
        try:
             loc_df.at[i, "PM10"] = response["data"]["iaqi"]["pm10"]["v"]
        except:
            pass

        # Update PM2.5
        try:
             loc_df.at[i, "PM2.5"] = response["data"]["iaqi"]["pm25"]["v"]
        except:
            pass

        # Update S02
        try:
             loc_df.at[i, "SO2"] = response["data"]["iaqi"]["so2"]["v"]
        except:
            pass
    
    return loc_df
