"""Trusted task verification profile rules."""

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.task_verification_contract import (
    TaskVerificationContract,
)

BACKEND_VERIFICATION_PROFILE = "backend"
FRONTEND_VERIFICATION_PROFILE = "frontend"

_CONTRACTS_BY_ROLE = {
    DevelopmentRole.BACKEND_DEVELOPER: TaskVerificationContract(
        profile_name=BACKEND_VERIFICATION_PROFILE,
        required_checks=(BACKEND_VERIFICATION_PROFILE,),
    ),
    DevelopmentRole.FRONTEND_DEVELOPER: TaskVerificationContract(
        profile_name=FRONTEND_VERIFICATION_PROFILE,
        required_checks=(FRONTEND_VERIFICATION_PROFILE,),
    ),
}


def default_verification_contract(
    role: DevelopmentRole,
) -> TaskVerificationContract | None:
    """Return the default verification contract for an implementation role."""
    return _CONTRACTS_BY_ROLE.get(role)


def valid_verification_profiles(role: DevelopmentRole) -> frozenset[str]:
    """Return trusted verification profile names for an assigned role."""
    contract = default_verification_contract(role)
    if contract is None:
        return frozenset()
    return frozenset({contract.profile_name})


def is_valid_verification_contract(
    role: DevelopmentRole,
    contract: TaskVerificationContract | None,
) -> bool:
    """Return whether a verification contract is valid for a task role."""
    expected = default_verification_contract(role)
    if expected is None:
        return contract is None
    return contract == expected
