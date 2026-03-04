# --- Географические ограничения ---

LANDLOCKED_COUNTRIES = {
    "AF",  # :contentReference[oaicite:0]{index=0}
    "AD",  # :contentReference[oaicite:1]{index=1}
    "AM",  # :contentReference[oaicite:2]{index=2}
    "AT",  # :contentReference[oaicite:3]{index=3}
    "AZ",  # :contentReference[oaicite:4]{index=4}
    "BY",  # :contentReference[oaicite:5]{index=5}
    "BT",  # :contentReference[oaicite:6]{index=6}
    "BO",  # :contentReference[oaicite:7]{index=7}
    "BW",  # :contentReference[oaicite:8]{index=8}
    "BF",  # :contentReference[oaicite:9]{index=9}
    "BI",  # :contentReference[oaicite:10]{index=10}
    "CF",  # :contentReference[oaicite:11]{index=11}
    "TD",  # :contentReference[oaicite:12]{index=12}
    "CZ",  # :contentReference[oaicite:13]{index=13}
    "SZ",  # :contentReference[oaicite:14]{index=14}
    "ET",  # :contentReference[oaicite:15]{index=15}
    "HU",  # :contentReference[oaicite:16]{index=16}
    "KZ",  # :contentReference[oaicite:17]{index=17}
    "KG",  # :contentReference[oaicite:18]{index=18}
    "LA",  # :contentReference[oaicite:19]{index=19}
    "LS",  # :contentReference[oaicite:20]{index=20}
    "LI",  # :contentReference[oaicite:21]{index=21}
    "LU",  # :contentReference[oaicite:22]{index=22}
    "MW",  # :contentReference[oaicite:23]{index=23}
    "ML",  # :contentReference[oaicite:24]{index=24}
    "MD",  # :contentReference[oaicite:25]{index=25}
    "MN",  # :contentReference[oaicite:26]{index=26}
    "NP",  # :contentReference[oaicite:27]{index=27}
    "NE",  # :contentReference[oaicite:28]{index=28}
    "MK",  # :contentReference[oaicite:29]{index=29}
    "PY",  # :contentReference[oaicite:30]{index=30}
    "RW",  # :contentReference[oaicite:31]{index=31}
    "SM",  # :contentReference[oaicite:32]{index=32}
    "RS",  # :contentReference[oaicite:33]{index=33}
    "SK",  # :contentReference[oaicite:34]{index=34}
    "SS",  # :contentReference[oaicite:35]{index=35}
    "CH",  # :contentReference[oaicite:36]{index=36}
    "TJ",  # :contentReference[oaicite:37]{index=37}
    "TM",  # :contentReference[oaicite:38]{index=38}
    "UG",  # :contentReference[oaicite:39]{index=39}
    "UZ",  # :contentReference[oaicite:40]{index=40}
    "VA",  # :contentReference[oaicite:41]{index=41}
    "ZM",  # :contentReference[oaicite:42]{index=42}
    "ZW",  # :contentReference[oaicite:43]{index=43}
}

def has_sea_access(country_code):
    if not country_code:
        return False
    return country_code not in LANDLOCKED_COUNTRIES


def determine_transport(phy: dict, geo: dict):

    size = phy.get("size_class", "unknown")
    brachy = phy.get("brachycephalic", False)
    needs_carrier = phy.get("needs_carrier", "unknown")

    same_continent = geo.get("same_continent", False)
    origin_code = geo.get("origin_country_code")
    destination_code = geo.get("destination_country_code")

    # -----------------------------
    # CAR
    # -----------------------------
    car_allowed = same_continent

    if size == "large":
        car_mode = "special_trailer"
    elif needs_carrier is True:
        car_mode = "carrier"
    else:
        car_mode = "standard"

    # -----------------------------
    # TRAIN
    # -----------------------------
    train_allowed = same_continent

    if train_allowed:
        if size == "large":
            train_mode = "special_compartment"
        elif needs_carrier is True:
            train_mode = "carrier_required"
        else:
            train_mode = "leash"
    else:
        train_mode = None

    # -----------------------------
    # PLANE
    # -----------------------------

    if brachy:
        plane_allowed = True
        plane_mode = "restricted_brachycephalic"

    elif size == "small" and needs_carrier:
        plane_allowed = True
        plane_mode = "cabin"

    elif size in ["medium", "large"]:
        plane_allowed = True
        plane_mode = "cargo"

    else:
        plane_allowed = True
        plane_mode = "cargo"

    # -----------------------------
    # SEA
    # -----------------------------
    sea_allowed = (
        has_sea_access(origin_code) and
        has_sea_access(destination_code)
    )

    sea_mode = "containerized" if sea_allowed else None

    return {
        "car": {
            "allowed": car_allowed,
            "mode": car_mode if car_allowed else None
        },
        "train": {
            "allowed": train_allowed,
            "mode": train_mode
        },
        "plane": {
            "allowed": plane_allowed,
            "mode": plane_mode
        },
        "sea": {
            "allowed": sea_allowed,
            "mode": sea_mode
        }
    }