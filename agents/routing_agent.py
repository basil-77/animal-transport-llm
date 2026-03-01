# agents/routing_agent.py

def estimate_times(distance_km):

    car_speed = 70
    train_speed = 120
    plane_speed = 800

    car_time = round(distance_km / car_speed, 1)
    train_time = round(distance_km / train_speed, 1)
    plane_time = round(distance_km / plane_speed + 3, 1)

    return {
        "car": car_time,
        "train": train_time,
        "plane": plane_time
    }


if __name__ == "__main__":
    print(estimate_times(800))