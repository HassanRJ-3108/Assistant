# src/first/crew.py
from crewai_core import Agent, Crew, Process, Task

from crewai.project import CrewBase, agent, crew, task
# Hum external tools (jaise SerperDevTool) remove kar dete hain taake koi external search na ho.
# from crewai_tools import SerperDevTool

@CrewBase
class PersonalAIAssistantCrew():
    """Personal AI Assistant Crew for Hassan RJ"""

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            # tools ki jagah koi external tool nahi lagana, taake sirf personal info use ho.
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='output/report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Personal AI Assistant Crew"""
        return Crew(
            agents=self.agents,   # Automatically created by the @agent decorator
            tasks=self.tasks,     # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
