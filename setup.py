from setuptools import find_packages,setup

setup(name="Agentic-Trading-Chatbot",version="0.0.1",
      author="Mahesh",author_email="maheshkpanda24@gmail.com",packages=find_packages(),
    install_requires=["langchain","lancedb","langgraph","tavily-python","polygon",
    ])