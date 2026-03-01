import sys

from agents.perception_agent import classify_animal
from agents.policy_agent import policy_decision
from agents.geo_agent import get_distance


def main(image_path, origin, destination):

    perception = classify_animal(image_path)

    animal = perception["animal"]

    distance = get_distance(origin, destination)

    decision = policy_decision(animal, distance)

    print("Animal:", animal)
    print("Distance:", round(distance, 1), "km")
    print("Decision:")
    print(decision)


if __name__ == "__main__":

    image = sys.argv[1]
    origin = sys.argv[2]
    destination = sys.argv[3]

    main(image, origin, destination)