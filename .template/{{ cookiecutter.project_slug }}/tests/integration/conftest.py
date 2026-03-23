# Copyright 2025 Canonical Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixtures for Juju integration tests."""

import logging
import os
import pathlib
import sys
import time
import typing
from collections.abc import Iterator

import jubilant
import pytest

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.OptionGroup):
    parser.addoption(
        '--keep-models',
        action='store_true',
        default=False,
        help='keep temporarily-created models',
    )
    parser.addoption(
        '--model',
        action='store',
        default=None,
        help='Name of the model to create',
    )


@pytest.fixture(scope='session')
def charm() -> str:
    """Return the charm name."""
    return 'test'  # determined by test charms' charmcraft.yaml


@pytest.fixture(scope='module')
def juju(request: pytest.FixtureRequest, charm: str) -> Iterator[jubilant.Juju]:
    """Pytest fixture that wraps :meth:`jubilant.with_model`.

    This adds command line parameter ``--keep-models`` (see help for details).
    """
    model = request.config.getoption('--model')
    keep_models = typing.cast('bool', request.config.getoption('--keep-models'))
    if not model:
        with jubilant.temp_model(keep=keep_models) as juju:
            juju.model_config({'logging-config': '<root>=INFO;unit=DEBUG'})

            _deploy(juju)
            juju.wait(jubilant.all_active)
            yield juju
            logger.info('Collecting Juju logs ...')
            time.sleep(0.5)  # Wait for Juju to process logs.
            debug_log = juju.debug_log(limit=1000)
            status = juju.cli("status")
    else:
        model_name = str(model)
        juju = jubilant.Juju()
        juju.model = model_name
        juju.wait_timeout = 30 * 60
        try:
            juju.status()
        except jubilant.CLIError:
            juju.add_model(model_name)

        yield juju

        debug_log = juju.debug_log(limit=1000)
        status = juju.cli('status')

    test_passed = True

    if request.session.testsfailed:
        print(f'Last {1000} lines of debug logs...')
        print(debug_log)
        logger.info('Juju status after the test failure:')
        logger.info(status)
        test_passed = False

    if model is not None and not keep_models and test_passed:
        juju.destroy_model(model_name, destroy_storage=True, force=True)


def _deploy(juju: jubilant.Juju) -> None:
    substrate = os.environ['CHARMLIBS_SUBSTRATE']
    # tag = os.environ.get('CHARMLIBS_TAG', '')  # get the tag if needed
    path = pathlib.Path(__file__).parent / '.packed' / f'{substrate}.charm'  # set by pack.sh
    if substrate == 'k8s':
        juju.deploy(path, resources={'workload': 'ubuntu:latest'})  # name set in metadata.yaml
    else:
        juju.deploy(path)
