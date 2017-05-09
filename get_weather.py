import requests
import json

# base code from https://gist.github.com/philshem/8864437#file-wunderground_historical-py

WUNDERGROUND_KEY = 'your API key here'

def get_weather(gooddate):
    urlstart = 'http://api.wunderground.com/api/'+ WUNDERGROUND_KEY +'/history_'
    urlend = '/q/Denmark/Copenhagen.json'

    url = urlstart + str(gooddate) + urlend
    data = requests.get(url).json()

    with open('data/weather/'+str(gooddate)+'.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    from datetime import date
    from dateutil.rrule import rrule, DAILY

    a = date(2014, 1, 4)
    b = date(2014, 12, 31)

    for dt in rrule(DAILY, dtstart=a, until=b):
        get_precip(dt.strftime("%Y%m%d"))
