FROM huggingface/transformers-pytorch-cpu

COPY / /opt/app/

RUN pip install -r /opt/app/requirements.txt

WORKDIR /opt/app

ENV LC_ALL=C.UTF-8

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9527", "--reload"]