from model import load_model, predict_age
from utils import load_image

def main():
    model = load_model("weights/age_model.h5")
    image_path = input("Enter image path: ")

    image = load_image(image_path)
    age = predict_age(model, image)

    print(f"Estimated age: {age} years")

if __name__ == "__main__":
    main()
