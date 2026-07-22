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

import json
import logging
import pathlib
import subprocess
import time
import typing
from collections.abc import Iterator
from datetime import datetime
from platform import machine

import jubilant
import pytest
from pytest_operator.plugin import OpsTest

logger = logging.getLogger(__name__)

LXD_CONTROLLER = 'lxd-controller'

PACKED_DIR = pathlib.Path(__file__).parent / '.packed'


@pytest.fixture(scope='module')
def juju(request: pytest.FixtureRequest) -> Iterator[jubilant.Juju]:
    """Pytest fixture that wraps :meth:`jubilant.with_model`.

    This adds command line parameter ``--keep-models`` (see help for details).
    """
    model = request.config.getoption('--model')
    keep_models = typing.cast('bool', request.config.getoption('--keep-models'))
    if not model:
        with jubilant.temp_model(keep=keep_models) as juju:
            juju.model_config({'logging-config': '<root>=INFO;unit=DEBUG'})
            yield juju
            logger.info('Collecting Juju logs ...')
            time.sleep(0.5)  # Wait for Juju to process logs.
            debug_log = juju.debug_log(limit=1000)
            status = juju.cli('status')
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


@pytest.fixture(scope='package')
def arch() -> str:
    """Fixture to provide the platform architecture for testing."""
    platforms = {
        'x86_64': 'amd64',
        'aarch64': 'arm64',
    }
    return platforms.get(machine(), 'amd64')


@pytest.fixture(scope='module')
def lxd_cloud(juju: jubilant.Juju):
    clouds = json.loads(juju.cli('clouds', '--format', 'json', include_model=False))
    for cloud, details in clouds.items():
        if details.get('type') == 'lxd':
            logger.info(f'Identified LXD cloud: {cloud}')
            yield cloud
            return

    logger.error('No LXD cloud found in Juju clouds. Available clouds: {clouds}')


@pytest.fixture(scope='module')
def lxd_controller(lxd_cloud: str, juju: jubilant.Juju):
    controllers = json.loads(juju.cli('controllers', '--format', 'json', include_model=False))
    logger.debug(f'Available controllers: {controllers}')
    for controller, details in controllers.get('controllers').items():
        if lxd_cloud == details.get('cloud'):
            logger.info(f'Identified LXD controller: {controller}')
            yield controller
            return

    logger.info(f'No controller with LXD cloud found. Available controllers: {controllers}')
    logger.info('Bootstrapping new LXD controller.')
    juju.bootstrap(lxd_cloud, controller=LXD_CONTROLLER)
    yield LXD_CONTROLLER

    juju.cli(
        'destroy-controller',
        LXD_CONTROLLER,
        '--destroy-all-models',
        '--destroy-storage',
        '--no-prompt',
        '--force',
        include_model=False,
    )


@pytest.fixture(scope='module')
def juju_lxd_model(request: pytest.FixtureRequest, juju: jubilant.Juju, lxd_cloud: str, lxd_controller: str):
    model = request.config.getoption('--model')
    keep_models = typing.cast('bool', request.config.getoption('--keep-models'))

    clouds_known = juju.cli('list-clouds', '--controller', lxd_controller, include_model=False)
    logger.debug(f'Known clouds: {clouds_known}')

    if not model:
        with jubilant.temp_model(cloud=lxd_cloud, controller=lxd_controller, keep=keep_models) as juju:
            juju.model_config({'logging-config': '<root>=INFO;unit=DEBUG'})
            yield juju
            logger.info('Collecting Juju logs ...')
            time.sleep(0.5)  # Wait for Juju to process logs.
            debug_log = juju.debug_log(limit=1000)
    else:
        model_name = str(model)
        juju = jubilant.Juju()
        juju.model = model_name
        juju.wait_timeout = 30 * 60
        try:
            juju.status()
        except jubilant.CLIError:
            juju.add_model(model_name, cloud=lxd_cloud, controller=lxd_controller)

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


@pytest.fixture
def application_charm() -> pathlib.Path:
    """Build the application charm."""
    return PACKED_DIR / 'application-charm_ubuntu-24.04-amd64.charm'


@pytest.fixture
def backward_compatibility_charm() -> pathlib.Path:
    """Build a v0 charm to integrate with a v1 client."""
    return PACKED_DIR / 'backward-compatibility-charm_ubuntu-24.04-amd64.charm'


@pytest.fixture
def database_charm() -> pathlib.Path:
    """Build the database charm."""
    return PACKED_DIR / 'database-charm_ubuntu-24.04-amd64.charm'


@pytest.fixture
def dummy_database_charm() -> pathlib.Path:
    """Build the database charm."""
    return PACKED_DIR / 'dummy-database-charm_ubuntu-24.04-amd64.charm'


@pytest.fixture
def kafka_charm() -> pathlib.Path:
    """Build the Kafka charm."""
    return PACKED_DIR / 'kafka-charm_ubuntu-24.04-amd64.charm'


@pytest.fixture
def kafka_connect_charm() -> pathlib.Path:
    """Build the Kafka Connect dummy charm."""
    return PACKED_DIR / 'kafka-connect-charm_ubuntu-24.04-amd64.charm'


@pytest.fixture
def opensearch_charm() -> pathlib.Path:
    """Build the OpenSearch charm.

    TODO we could simplify a lot of these charm builds by having a single test charm that includes
    all these relations. This might be easily achieved by merging this repo with the
    data-integrator charm repo.
    """
    return PACKED_DIR / 'opensearch-charm_ubuntu-24.04-amd64.charm'


@pytest.fixture(autouse=True)
async def without_errors(ops_test: OpsTest, request):
    """This fixture is to list all those errors that mustn't occur during execution."""
    # To be executed after the tests
    now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    yield
    whitelist = []
    if 'log_errors_allowed' in request.keywords:
        for marker in [mark for mark in request.node.iter_markers() if mark.name == 'log_errors_allowed']:
            for arg in marker.args:
                whitelist.append(arg)

        # All errors allowed
        if not whitelist:
            return

    whitelist.extend([
        'kvm-container-provisioner',
    ])

    _, dbg_log, _ = await ops_test.juju('debug-log', '--ms', '--replay')
    lines = dbg_log.split('\n')
    for _, line in enumerate(lines):
        logitems = line.split(' ')
        if not line or len(logitems) < 3:
            continue
        if logitems[1] < now:
            continue
        if logitems[2] == 'ERROR':
            assert any(white in line for white in whitelist)


@pytest.fixture(scope='module')
def install_etcdctl():
    subprocess.run(['bash', f'{pathlib.Path(__file__).parent}/scripts/install_etcdctl.sh'])
    yield
