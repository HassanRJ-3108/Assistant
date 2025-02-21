import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_google_genai import ChatGoogleGenerativeAI

@CrewBase
class PersonalAIAssistantCrew():
    """Personal AI Assistant Crew for Hassan RJ"""
    config_path = "first/config/tasks.yaml"

    def get_llm(self):
        """Get the Gemini language model"""
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    @agent
    def researcher(self) -> Agent:
        config = self.agents_config['researcher']
        return Agent(
            llm=self.get_llm(),  # Use Gemini model
            **{k: v for k, v in config.items() if k != 'llm'},
            verbose=True,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        config = self.agents_config['reporting_analyst']
        return Agent(
            llm=self.get_llm(),  # Use Gemini model
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