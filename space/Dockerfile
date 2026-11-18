# Hugging Face Docker Space for the Melbourne Housing Streamlit app.
#
# Hugging Face expects Spaces created programmatically to run as Docker Spaces on
# port 7860. This image serves the app with the pre-trained model artifact.
FROM python:3.12-slim

# Spaces run containers as user id 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app.py train_model.py ./
COPY --chown=user data ./data
COPY --chown=user model ./model
COPY --chown=user .streamlit ./.streamlit

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
