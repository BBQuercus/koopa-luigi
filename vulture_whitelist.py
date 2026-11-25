# Vulture whitelist: names vulture reports as unused that are in fact used. Not imported by anything.
# Run the dead-code check with `make lint-dead`.

_.side_effect  # Mock configuration in tests/test_cli.py
