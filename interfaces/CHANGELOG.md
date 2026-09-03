# 1.2.0 - 1 September 2026

* Add prefixes and custom username capability

# 1.1.1 - 28 August 2026

* Add integration test coverage for cross-model relations in etcd

# 1.1.0 - 03 June 2026

* Add capability to encrypt and decrypt specific fields and store them in relation data
* In cross-model relations, store 'mtls-cert' encrypted to relation data instead of a secret
* requires `cryptography` >= 48.0.0 to be added to a charms dependencies

# 1.0.2 - 31 March 2026

* Fix import for `ValkeyResponseModel`

# 1.0.1 - 27 March 2026

* Add support for `valkey_client` interface

# 1.0.0 - 24 March 2026

* Initial release of the library
