vehicle = {
    "class": "Vehicle",
    "brand": "Toyota",
    "fuel_type": "Petrol",
    "wheels": 4,
    "color": "White"
}

print("Vehicle Frame")
for slot, value in vehicle.items():
    print(slot, ":", value)
