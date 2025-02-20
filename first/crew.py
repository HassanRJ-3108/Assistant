# src/first/crew.py
try:
    from crewai import Agent, Crew, Process, Task
    from crewai.project import CrewBase, agent, crew, task
except ImportError as e:
    print(f"Error importing CrewAI: {e}")
    # You might want to add a fallback mechanism here

@CrewBase
class PersonalAIAssistantCrew():
    """Personal AI Assistant Crew for Hassan RJ"""

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
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
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
