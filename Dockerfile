FROM  gcr.io/viden-314402/viden_runnable
#Use below image to modify and create new images for any other env than gcp
#FROM mukulbhave/viden_runnable
#ENV FLASK_APP=predict.py
#ENV PYTHONPATH=/viden_code/viden/python_code/yolo/
#ENV KMP_DUPLICATE_LIB_OK=TRUE
#WORKDIR /viden_code/viden/python_code/services
#SHELL ["conda", "run", "-n", "tf1", "/bin/bash", "-c"]
#RUN sudo chmod 777 -R /viden_code/viden/python_code/

#EXPOSE 5000
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tf1", "flask", "run", "--host=0.0.0.0" ]

COPY python_code/services/templates/index.html templates/index.html
COPY python_code/services/static/viden.css static/viden.css