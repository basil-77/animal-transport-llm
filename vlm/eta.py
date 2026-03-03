def estimate_eta(distance_km, transport):

    result = {}

    for mode in ["car", "train", "plane", "sea"]:
        result[mode] = {
            "allowed": transport[mode]["allowed"],
            "mode": transport[mode]["mode"],
            "eta_hours": None
        }

    if transport["car"]["allowed"]:
        result["car"]["eta_hours"] = round(distance_km / 70, 1)

    if transport["train"]["allowed"]:
        result["train"]["eta_hours"] = round(distance_km / 120, 1)

    if transport["plane"]["allowed"]:
        result["plane"]["eta_hours"] = round(distance_km / 800 + 3, 1)

    if transport["sea"]["allowed"]:
        result["sea"]["eta_hours"] = round(distance_km / 35, 1)

    return result