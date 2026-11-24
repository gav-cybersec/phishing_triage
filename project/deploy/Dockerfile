# The Python version here is rewritten by the training notebook to match the environment that trained the models
FROM python:3.14-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Fail the build if the package is broken (wrong library versions, models missing, validation or scoring errors)
RUN python test_features.py && python smoke_test.py

EXPOSE 7860
CMD ["python", "app.py"]
