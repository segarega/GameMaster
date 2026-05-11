from setuptools import find_packages, setup

setup(
    name="gamemaster",
    version="1.0.0",
    description="Selector/static-GM-index proxy for AIInfluence Bannerlord mod",
    author="GameMaster",
    packages=find_packages(),
    py_modules=["main", "run", "gamemaster_gui", "gm_gui"],
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.0.0",
    ],
    entry_points={
        "console_scripts": [
            "gamemaster=run:main",
        ]
    },
)
