from geopy.geocoders import Nominatim
import math

geolocator = Nominatim(user_agent="transport_service")


def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat/2)**2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon/2)**2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def get_distance(city1, city2):

    loc1 = geolocator.geocode(city1)
    loc2 = geolocator.geocode(city2)

    if not loc1 or not loc2:
        raise Exception("City not found")

    return haversine(
        loc1.latitude,
        loc1.longitude,
        loc2.latitude,
        loc2.longitude
    )


if __name__ == "__main__":

    print(get_distance("Berlin", "Munich"))