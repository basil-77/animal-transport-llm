from vlm.mock_vlm import analyze_image
from vlm.geo import get_geo_info
from vlm.transport import determine_transport
from vlm.eta import estimate_eta


def run_pipeline(image_path, origin, destination):

    phy = analyze_image(image_path)

    geo = get_geo_info(origin, destination)

    transport = determine_transport(phy, geo)

    eta = estimate_eta(geo["distance_km"], transport)

    return {
        "physical_profile": phy,
        "geo": geo,
        "transport": transport,
        "eta_hours": eta
    }


if __name__ == "__main__":
    result = run_pipeline("dog.jpg", "Berlin", "Tokyo")
    print(result)