FROM public.ecr.aws/lambda/python:3.10 as lambda_builder


WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt main.py .

RUN python -m venv /opt/venv && . /opt/venv/bin/activate &&  pip install --no-cache-dir -r requirements.txt
    
# Stage 2: Production Stage
FROM public.ecr.aws/lambda/python:3.10 
COPY --from=lambda_builder /opt/venv /opt/venv
RUN yum -y install gcc-c++
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR ${LAMBDA_TASK_ROOT}

COPY . .

ENV PATH="/opt/venv/bin/:$PATH"

ENTRYPOINT ["python", "main.py"]

