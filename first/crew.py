try:
    from crewai import Agent, Crew, Process, Task
    from crewai.project import CrewBase, agent, crew, task
except ImportError as e:
    print(f"Error importing CrewAI: {e}")

@CrewBase
class PersonalAIAssistantCrew():
    """Personal AI Assistant Crew for Hassan RJ"""
    config_path = "first/config/tasks.yaml"

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
        task_config = {
            **self.tasks_config['research_task'],
            "agent": self.researcher()
        }
        return Task.model_validate(task_config)

    @task
    def reporting_task(self) -> Task:
        task_config = {
            **self.tasks_config['reporting_task'],
            "agent": self.reporting_analyst(),
            "output_file": 'output/report.md'
        }
        return Task.model_validate(task_config)

    @crew
    def crew(self) -> Crew:
        """Creates the Personal AI Assistant Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )