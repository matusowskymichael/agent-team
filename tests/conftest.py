"""Project-wide pytest plugins and shared reporting support."""

pytest_plugins = (
    "pytester",
    "tests.reporting.allure_pytest_plugin",
)
