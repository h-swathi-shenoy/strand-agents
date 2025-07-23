FROM public.ecr.aws/lambda/python:3.10 

RUN yum -y install gcc-c++
WORKDIR ${LAMBDA_TASK_ROOT}

COPY . .
COPY env.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}
RUN python app.py
CMD ["app.lambda_handler"]

