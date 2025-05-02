from google import genai

client = genai.Client(api_key="AIzaSyDiO3SFDITVn3hktp53mnlFBl-c2Ucipfk")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few 100 words"
)
print(response.text)