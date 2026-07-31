"""
Class mappings for ProjectVerba Waste Detection Dataset.

Original classes (42)
↓

Final classes (5)
"""

# ==========================================================
# Final Classes
# ==========================================================

FINAL_CLASSES = [
    "Plastic",
    "Paper",
    "Glass",
    "Metal",
    "Trash"
]

# ==========================================================
# ProjectVerba -> Final Classes
# ==========================================================

CLASS_MAPPING = {

    # -------------------------
    # Plastic
    # -------------------------

    "Combined plastic": "Plastic",
    "Milk bottle": "Plastic",
    "Plastic bag": "Plastic",
    "Plastic bottle": "Plastic",
    "Plastic can": "Plastic",
    "Plastic canister": "Plastic",
    "Plastic caps": "Plastic",
    "Plastic cup": "Plastic",
    "Plastic shaker": "Plastic",
    "Plastic shavings": "Plastic",
    "Plastic toys": "Plastic",
    "Stretch film": "Plastic",
    "Unknown plastic": "Plastic",
    "Zip plastic bag": "Plastic",

    # -------------------------
    # Paper
    # -------------------------

    "Cardboard": "Paper",
    "Cellulose": "Paper",
    "Paper": "Paper",
    "Paper bag": "Paper",
    "Paper cups": "Paper",
    "Paper shavings": "Paper",
    "Papier mache": "Paper",
    "Postal packaging": "Paper",
    "Tetra pack": "Paper",

    # -------------------------
    # Glass
    # -------------------------

    "Glass bottle": "Glass",

    # -------------------------
    # Metal
    # -------------------------

    "Aluminum can": "Metal",
    "Aluminum caps": "Metal",
    "Foil": "Metal",
    "Iron utensils": "Metal",
    "Metal shavings": "Metal",
    "Scrap metal": "Metal",
    "Tin": "Metal",

    # -------------------------
    # Trash
    # -------------------------

    "Aerosols": "Trash",
    "Ceramic": "Trash",
    "Container for household chemicals": "Trash",
    "Disposable tableware": "Trash",
    "Electronics": "Trash",
    "Furniture": "Trash",
    "Liquid": "Trash",
    "Organic": "Trash",
    "Printing industry": "Trash",
    "Textile": "Trash",
    "Wood": "Trash"
}

# ==========================================================
# Final Class IDs
# ==========================================================

FINAL_CLASS_TO_ID = {
    name: index
    for index, name in enumerate(FINAL_CLASSES)
}

ID_TO_FINAL_CLASS = {
    index: name
    for index, name in enumerate(FINAL_CLASSES)
}