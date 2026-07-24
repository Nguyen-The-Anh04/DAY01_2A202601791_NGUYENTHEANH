from google import genai

client = genai.Client(api_key=None)  # tự đọc GEMINI_API_KEY

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents="Xin chào. Hãy trả lời đúng một câu: Gemini API đang hoạt động."
    )

    print("=== RESPONSE ===")
    print(response.text)

except Exception as e:
    print("=== ERROR ===")
    print(type(e).__name__)
    print(e)