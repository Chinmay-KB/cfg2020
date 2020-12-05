import requests

def return_aqi(location):
    token = "4b73e0872f9eecfaf9d32580b1b378b250cbb77e"
    url = f"https://api.waqi.info/feed/{location}/?token={token}"

    response = requests.get(url)

    try:
        return response.json()["data"]["aqi"]
    except TypeError:
        return None


if __name__ == "__main__":
    location = input("Enter the location:\n")
    print(return_aqi(location))
