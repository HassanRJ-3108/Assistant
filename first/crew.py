import os
# Force Python implementation for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.project import CrewBase, agent, crew, task
except ImportError:
    # Basic implementation if crewai import fails
    from dataclasses import dataclass
    
    @dataclass
    class Agent:
        config: dict
        verbose: bool = False
    
    @dataclass
    class Task:
        config: dict
        output_file: str = None
    
    class Process:
        sequential = "sequential"
    
    @dataclass
    class Crew:
        agents: list
        tasks: list
        process: str
        verbose: bool = False
        
        def kickoff(self, inputs):
            return f"Processing query: {inputs.get('user_query', '')}"
    
    class CrewBase:
        pass
    
    def agent(func): return func
    def task(func): return func
    def crew(func): return func

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