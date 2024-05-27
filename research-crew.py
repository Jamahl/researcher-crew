import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, PDFSearchTool, DirectorySearchTool

# Set API keys
os.environ["SERPER_API_KEY"] = "Your Serper API Key"
os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"

# Ask the user for the research topic
topic = input("Please enter the research topic: ")

# Define the Master Researcher Agent using gpt-4
master_researcher = Agent(
    role='Master Researcher',
    goal='Oversee the research project and ensure comprehensive coverage of the topic',
    backstory='Experienced in managing research projects and synthesizing diverse information',
    tools=[],
    verbose=True,
    memory=True,
    model='gpt-4',
    allow_delegation=True
)

# Define the General Researcher Agent using gpt-3.5-turbo
general_researcher = Agent(
    role='General Researcher',
    goal='Conduct thorough web searches on the topic',
    backstory='Skilled in gathering information from various online sources and compiling comprehensive reports',
    tools=[SerperDevTool()],
    verbose=True,
    memory=True,
    model='gpt-3.5-turbo',
    allow_delegation=False
)

# Define the Report Researcher Agent using gpt-3.5-turbo
report_researcher = Agent(
    role='Report Researcher',
    goal='Find and summarize industry and scientific reports',
    backstory='Proficient in analyzing and summarizing detailed reports',
    tools=[PDFSearchTool(), DirectorySearchTool()],
    verbose=True,
    memory=True,
    model='gpt-3.5-turbo',
    allow_delegation=False
)

# Define the Q+A Researcher Agent using gpt-4
qa_researcher = Agent(
    role='Q+A Researcher',
    goal='Review initial reports and generate questions for improvement',
    backstory='Expert in critical analysis and enhancing the quality of reports',
    tools=[],
    verbose=True,
    memory=True,
    model='gpt-4',
    allow_delegation=False
)

# Define the General Research Task
general_research_task = Task(
    description='Conduct thorough research on the topic covering various aspects such as history, current trends, applications, and failures.',
    expected_output=f'A comprehensive report titled "{topic} report stage 1".',
    tools=[SerperDevTool()],
    agent=general_researcher
)

# Define the Report Summary Task
report_summary_task = Task(
    description='Find and summarize relevant industry and scientific reports on the topic.',
    expected_output=f'A summarized report titled "{topic} report stage 2".',
    tools=[PDFSearchTool(), DirectorySearchTool()],
    agent=report_researcher
)

# Define the Q+A Compilation Task
qa_compilation_task = Task(
    description='Compile reports from the General Researcher and Report Researcher, generate 10 questions to improve the report.',
    expected_output='A set of questions for refining the report.',
    tools=[],
    agent=qa_researcher
)

# Define the Comprehensive Report Task
comprehensive_report_task = Task(
    description='Use feedback from the Q+A Researcher to create a comprehensive report.',
    expected_output=f'A detailed final report titled "{topic} report stage 3".',
    tools=[],
    agent=master_researcher
)

# Form the crew and initiate the process
crew = Crew(
    agents=[master_researcher, general_researcher, report_researcher, qa_researcher],
    tasks=[general_research_task, report_summary_task, qa_compilation_task, comprehensive_report_task],
    process=Process.sequential  # Ensures tasks are executed in order
)

# Kickoff the crew with the user-defined topic
result = crew.kickoff(inputs={'topic': topic})
print(result)
