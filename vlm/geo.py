from geopy.geocoders import Nominatim
import math
import pycountry_convert as pc

geolocator = Nominatim(user_agent="animal_transport_vlm")


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

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def alpha2_to_continent(alpha2):
    try:
        continent_code = pc.country_alpha2_to_continent_code(alpha2.upper())
        return pc.convert_continent_code_to_continent_name(continent_code)
    except Exception:
        return None


def extract_country_data(location):
    if not location:
        return None, None

    address = location.raw.get("address", {})

    country_name = address.get("country")
    country_code = address.get("country_code")  # ISO-2

    if country_code:
        country_code = country_code.upper()

    return country_name, country_code


def get_geo_info(origin: str, destination: str):

    loc1 = geolocator.geocode(origin, exactly_one=True, addressdetails=True)
    loc2 = geolocator.geocode(destination, exactly_one=True, addressdetails=True)

    if not loc1 or not loc2:
        raise ValueError("Location not found")

    distance = haversine(
        loc1.latitude,
        loc1.longitude,
        loc2.latitude,
        loc2.longitude
    )

    country1, code1 = extract_country_data(loc1)
    country2, code2 = extract_country_data(loc2)

    continent1 = alpha2_to_continent(code1) if code1 else None
    continent2 = alpha2_to_continent(code2) if code2 else None

    return {
        "distance_km": round(distance, 1),

        "origin_country": country1,
        "destination_country": country2,

        "origin_country_code": code1,
        "destination_country_code": code2,

        "origin_continent": continent1,
        "destination_continent": continent2,

        "same_country": code1 == code2 if code1 and code2 else False,
        "same_continent": continent1 == continent2 if continent1 and continent2 else False
    }