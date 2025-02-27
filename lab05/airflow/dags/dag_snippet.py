from datetime import datetime

# The DAG object; we'll need this to instantiate a DAG.
from airflow import DAG

# Operators; we need these to define our tasks!
from airflow.decorators import task
from airflow.operators.bash import BashOperator

# A DAG represents a workflow, a collection of tasks
with DAG(dag_id="demo", start_date=datetime(2022, 1, 1), schedule="0 0 * * *") as dag:
    # Tasks are represented as operators
    hello = BashOperator(task_id="hello", bash_command="echo hello")

    # Tasks can also be declared using decorated python functions.
    # This is known as "taskflow".
    @task()
    def airflow():
        print("airflow")

    # Set dependencies between tasks
    hello >> airflow()