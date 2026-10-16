FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.py server.py ./
ENV CHECKPOINT=/checkpoints/policy.pt
ENV VRAM_BUDGET_GB=2.5
VOLUME /checkpoints
EXPOSE 8090
CMD ["python", "server.py"]
