"""Sylo framework integrations package.

Available integrations:
    - ``sylo.integrations.langgraph``      — LangGraph StateGraph wrapper
    - ``sylo.integrations.openai_agents``  — OpenAI Agents SDK wrapper
    - ``sylo.integrations.crewai``         — CrewAI Crew wrapper

Each integration is lazily imported so the corresponding framework
package is only required when you actually use that integration.
"""
