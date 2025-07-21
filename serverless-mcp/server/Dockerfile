FROM public.ecr.aws/lambda/python:3.10 as lambda_builder


WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt app.py .

RUN python -m venv /opt/venv && . /opt/venv/bin/activate &&  pip install --no-cache-dir -r requirements.txt
    
# Stage 2: Production Stage
FROM public.ecr.aws/lambda/python:3.10 
COPY --from=lambda_builder /opt/venv /opt/venv
RUN yum -y install gcc-c++

WORKDIR ${LAMBDA_TASK_ROOT}

COPY . .

ENV PATH="/opt/venv/bin/:$PATH"

CMD ["app.lambda_handler"]

