#!/usr/bin/env python3
"""Creating a method that returns the list of ships
that can hold a given number of passengers"""

import requests

def availableShips(passengerCount):
    """Returns a list of ships that can hold a given number of passengers"""
    api_url = 'https://swapi-api.alx-tools.com/api/starships/?page=1'

    response = requests.get(api_url)
    output = []

    while response.status_code == 200:
        res = response.json()
        for ship in res['results']:
            passengers = ship['passengers'].replace(',', '')
            try:
                if int(passengers) >= passengerCount:
                    output.append(ship['name'])
            except ValueError:
                pass
        
        try:
            response = requests.get(res['next'])
        except Exception:
            break

    return output
