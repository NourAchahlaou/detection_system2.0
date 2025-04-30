import base64

with open("D:/Professional Vault/Internship/my work/airbus v2/airbus_ui/public/Airbus-Logo.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    print(base64_image)
