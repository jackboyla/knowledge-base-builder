from setuptools import setup

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="knowledge_graph_validator",
    version=version,
    packages=["knowledge_graph_validator"],
)
