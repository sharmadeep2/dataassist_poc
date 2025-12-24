import os

# Remove legacy vars that cause the Pydantic error
for var in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_TYPE"):
    os.environ.pop(var, None)

print("OPENAI_API_BASE:", os.getenv("OPENAI_API_BASE"))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))
