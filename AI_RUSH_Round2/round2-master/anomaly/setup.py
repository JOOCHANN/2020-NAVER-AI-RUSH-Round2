#nsml: pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
from distutils.core import setup

setup(
    name="user anomaly detection - AI Rush baseline",
    version="1",
    install_requires=[
            'numpy==1.19.0',
            'scikit-learn==0.23.1',
            'pandas==1.0.4'
        ]
)
