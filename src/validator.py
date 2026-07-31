from pathlib import Path

ALLOWED_EXTENSIONS = {

".jpg",

".jpeg",

".png"

}


def allowed(filename):

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS