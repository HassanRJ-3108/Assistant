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
        # Create a complete task configuration dictionary
        task_config = {
            **self.tasks_config['research_task'],  # Spread existing config
            "agent": self.researcher(),  # Add required agent
        }
        
        # Use model_validate to create the task
        return Task.model_validate(task_config)

    @task
    def reporting_task(self) -> Task:
        # Create a complete task configuration dictionary
        task_config = {
            **self.tasks_config['reporting_task'],  # Spread existing config
            "agent": self.reporting_analyst(),  # Add required agent
            "output_file": 'output/report.md'
        }
        
        # Use model_validate to create the task
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