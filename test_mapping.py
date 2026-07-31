from src.mappings import *

print("=" * 50)

print("Final Classes")

for i, cls in enumerate(FINAL_CLASSES):
    print(i, cls)

print("=" * 50)

print("Plastic Bottle ->",
      CLASS_MAPPING["Plastic bottle"])

print("Cardboard ->",
      CLASS_MAPPING["Cardboard"])

print("Glass bottle ->",
      CLASS_MAPPING["Glass bottle"])

print("Tin ->",
      CLASS_MAPPING["Tin"])

print("Wood ->",
      CLASS_MAPPING["Wood"])

print("=" * 50)

print(FINAL_CLASS_TO_ID)