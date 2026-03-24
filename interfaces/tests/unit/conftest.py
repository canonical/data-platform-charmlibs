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

"""Fixtures for unit tests, typically mocking out parts of the external system."""

import os
from importlib.metadata import PackageNotFoundError, version
from unittest.mock import PropertyMock

import pytest
from ops import JujuVersion
from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def juju_has_secrets(mocker: MockerFixture):
    """This fixture will force the usage of secrets whenever run on Juju 3.x.

    NOTE: This is needed, as normally JujuVersion is set to 0.0.0 in tests
    (i.e. not the real juju version)
    """
    juju_version = ''
    if juju_version := os.environ.get('LIBJUJU_VERSION_SPECIFIER', ''):
        juju_version.replace('==', '')
        juju_version = juju_version[2:].split('.')[0]
    else:
        try:
            juju_version = version('juju')
        except PackageNotFoundError:
            juju_version: str = '3.6.1.0'

    if juju_version < '3':
        mocker.patch.object(JujuVersion, 'has_secrets', new_callable=PropertyMock).return_value = False
        return False
    else:
        mocker.patch.object(JujuVersion, 'has_secrets', new_callable=PropertyMock).return_value = True
        return True


@pytest.fixture
def only_with_juju_secrets(juju_has_secrets):
    """Pretty way to skip Juju 3 tests."""
    if not juju_has_secrets:
        pytest.skip('Secrets test only applies on Juju 3.x')
