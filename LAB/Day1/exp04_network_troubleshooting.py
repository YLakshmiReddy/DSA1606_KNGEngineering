internet = input("Can you access the internet? (yes/no): ").strip().lower()
wifi = input("Is Wi-Fi connected? (yes/no): ").strip().lower()

if internet == "yes":
    print("Diagnosis: Internet connection is working.")
elif wifi == "no":
    print("Diagnosis: Check Wi-Fi settings or router power.")
else:
    print("Diagnosis: Check router connectivity and network settings.")
