import sys
import json

from vlm.client import analyze_image
from vlm.geo import get_geo_info
from vlm.transport import determine_transport
from vlm.eta import estimate_eta


def run_pipeline(image_path, origin, destination):

    # VLM
    phy = analyze_image(image_path)

    # GEO
    geo = get_geo_info(origin, destination)

    # TRANSPORT RULES
    transport = determine_transport(phy, geo)

    # ETA
    transport_options = estimate_eta(
        geo["distance_km"],
        transport
    )

    result = {
        "physical_profile": phy,
        "geo": geo,
        "transport_options": transport_options
    }

    return result


def main():

    if len(sys.argv) < 4:
        print("Usage:")
        print("python main.py <image_path> <origin> <destination>")
        sys.exit(1)

    image_path = sys.argv[1]
    origin = sys.argv[2]
    destination = sys.argv[3]

    result = run_pipeline(image_path, origin, destination)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()