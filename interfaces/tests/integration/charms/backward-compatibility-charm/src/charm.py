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

"""Client charm that creates connection to database charm.

This charm is meant to be used only for testing of the libraries in this repository.
This uses the v0 of data interfaces to ensure that the compatibility is not broken.
"""

import logging

import common
import ops
from ops.framework import StoredState
from ops.model import ActiveStatus

from charms.data_platform_libs.v0.data_interfaces import DatabaseCreatedEvent, DatabaseRequires

logger = logging.getLogger(__name__)


class Charm(common.Charm):
    """Charm the application."""

    _stored = StoredState()

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        # Default charm events.
        self.framework.observe(self.on.start, self._on_start)

        # Charm events defined in the database provides charm library.
        self.database = DatabaseRequires(self, 'backward-database', 'bwclient')
        self.framework.observe(self.database.on.database_created, self._on_resource_created)

    def _on_start(self, _) -> None:
        """Only sets an active status."""
        self.unit.status = ActiveStatus('Backward compatibility charm ready!')

    def _on_resource_created(self, event: DatabaseCreatedEvent) -> None:
        """Event triggered when a new database is requested."""
        relation_id = event.relation.id
        username = event.username
        password = event.password
        database = event.database

        logger.info(
            f'Database {database} created for relation {relation_id} with user {username} and password {password}'
        )
        self.unit.status = ActiveStatus('backward_database_created')


if __name__ == '__main__':  # pragma: nocover
    ops.main(Charm)
