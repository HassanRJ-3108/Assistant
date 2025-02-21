import os
import asyncio
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_google_genai import ChatGoogleGenerativeAI
import nest_asyncio

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

@CrewBase
class PersonalAIAssistantCrew():
    """Personal AI Assistant Crew for Hassan RJ"""
    config_path = "first/config/tasks.yaml"

    def __init__(self):
        super().__init__()
        # Ensure event loop exists
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        # Initialize LLM once
        self.llm = self.get_llm()

    def get_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    @agent
    def researcher(self) -> Agent:
        config = self.agents_config['researcher']
        return Agent(
            llm=self.llm,
            **{k: v for k, v in config.items() if k != 'llm'},
            verbose=True,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        config = self.agents_config['reporting_analyst']
        return Agent(
            llm=self.llm,
            **{k: v for k, v in config.items() if k != 'llm'},
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