import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest


@pytest.fixture(scope='session')
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pytest_sessionfinish(session, exitstatus):
    for d in os.listdir('runs'):
        if d.startswith('test_') or d.startswith('_tmp_test_'):
            shutil.rmtree(os.path.join('runs', d), ignore_errors=True)
