FROM python:3.10-slim

WORKDIR /code

# Copy requirements and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all repository files into the container
COPY . /code

# Expose Streamlit default port
EXPOSE 7860

# Command to run the Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
