import pytest

from authorai import db as dbmod

DIM = 8


@pytest.fixture()
def conn(tmp_path):
    connection = dbmod.connect(tmp_path / "test.db", embedding_dim=DIM)
    yield connection
    connection.close()
