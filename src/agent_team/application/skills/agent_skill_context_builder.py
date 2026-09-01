"""Runtime context builder for available Agent Skill metadata."""

from dataclasses import dataclass

from agent_team.domain.skills.agent_skill_metadata import AgentSkillMetadata


@dataclass(frozen=True, slots=True)
class AgentSkillContextBuilder:
    """Build concise model-visible metadata for assigned Agent Skills."""

    def build_context(
        self,
        metadata: tuple[AgentSkillMetadata, ...],
    ) -> str:
        """Return skill names and descriptions without instruction bodies."""
        if not metadata:
            return "Available skills: none."

        lines = [
            "Available skills:",
        ]
        for item in sorted(metadata, key=lambda skill: skill.name.value):
            version = f" version {item.version}" if item.version else ""
            digest = item.content_hash[:12]
            lines.append(
                f"- {item.name.value}{version}: {item.description} "
                f"(hash {digest})",
            )
        return "\n".join(lines)
