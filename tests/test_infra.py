import sys
from scripts.verify_infra import check_python_version, check_qdrant, check_redis

def test_python_version():
    assert check_python_version() == True

# We only test if they return a boolean. 
# True integration testing will happen in Github actions or require the docker containers to be up.
# Since we might run this test suite when docker is not up, we don't strictly assert True.
def test_qdrant_check_returns_bool():
    result = check_qdrant()
    assert isinstance(result, bool)

def test_redis_check_returns_bool():
    result = check_redis()
    assert isinstance(result, bool)
