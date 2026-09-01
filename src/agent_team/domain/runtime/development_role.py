"""Development role values."""

from enum import StrEnum


class DevelopmentRole(StrEnum):
    """Role responsible for a development task."""

    BUSINESS_ANALYST = "business_analyst"
    SOFTWARE_ARCHITECT = "software_architect"
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    QA_ENGINEER = "qa_engineer"
    CODE_REVIEWER = "code_reviewer"
    DELIVERY_MANAGER = "delivery_manager"
