import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, PDFSearchTool, DirectorySearchTool
import openai
import agentops
import markdown
from weasyprint import HTML

# Set API keys
os.environ["SERPER_API_KEY"] = 'your-serper-api-key'
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'
os.environ["AGENTOPS_API_KEY"] = 'your-agentops-api-key'

# Initialize OpenAI with the API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize AgentOps
agentops.init(os.environ["AGENTOPS_API_KEY"])

# Ask the user for the research topic
topic = input("Please enter the research topic: ")

# Define the Master Researcher Agent using gpt-4
master_researcher = Agent(
    role='Big Daddy Researcher',
    goal='Oversee the research project and ensure comprehensive coverage of the topic suitable for professional consumption',
    backstory='You are experienced in managing research projects and synthesizing diverse information across multiple sources and managing multiple agents. You will make sure that no more than 600,000 tokens are used for the output of this report. OR, you will ensure the final report generated does not exceed 250 words. ',
    tools=[],
    verbose=True,
    memory=True,
    model='gpt-4',
    allow_delegation=True,
    agentops=agentops  # Integrate AgentOps for cost tracking
)

# Define the General Researcher Agent using gpt-3.5-turbo
general_researcher = Agent(
    role='Biscuit Researcher',
    goal='Conduct thorough web searches on the topic looking at a minimum of 5 reputable sources',
    backstory='You are skilled in gathering information from various online sources and compiling comprehensive reports',
    tools=[SerperDevTool()],
    verbose=True,
    memory=True,
    model='gpt-3.5-turbo',
    allow_delegation=False,
    agentops=agentops  # Integrate AgentOps for cost tracking
)

# Define the Report Researcher Agent using gpt-3.5-turbo
report_researcher = Agent(
    role='Report finder Rupert',
    goal='Find and summarize industry and scientific reports on the topic, looking at multiple sources to gain different information',
    backstory='Proficient in analyzing and summarizing detailed reports including looking at unstructured data',
    tools=[PDFSearchTool(), DirectorySearchTool()],
    verbose=True,
    memory=True,
    model='gpt-3.5-turbo',
    allow_delegation=False,
    agentops=agentops  # Integrate AgentOps for cost tracking
)

# Define the Q+A Researcher Agent using gpt-4
qa_researcher = Agent(
    role='Investigator',
    goal='Review initial reports and generate questions for improvement, find 5 topics that have not been covered so far in the reserach report and feed back to the Big Daddy Researcher Master',
    backstory='Expert in critical analysis and enhancing the quality of reports',
    tools=[],
    verbose=True,
    memory=True,
    model='gpt-4',
    allow_delegation=False,
    agentops=agentops  # Integrate AgentOps for cost tracking
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
    tools=[PDFSearchTool()],
    agent=qa_researcher
)

# Define the Comprehensive Report Task
comprehensive_report_task = Task(
    description='Use feedback from the Q+A Researcher to create a comprehensive report.',
    expected_output=f'A detailed final report titled "{topic} report stage 3".',
    tools=[SerperDevTool()],
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

# Print the cost tracking information
cost_info = agentops.get_cost_summary()
print(cost_info)

# Save the final result to a markdown-formatted text file
output_txt_file = f"{topic}_final_report.md"
with open(output_txt_file, "w") as file:
    file.write(result['output'])  # Assuming result is a dictionary with an 'output' key

# Convert the markdown file to HTML
html_content = markdown.markdown(result['output'])

# Save the HTML content as a PDF file
output_pdf_file = f"{topic}_final_report.pdf"
HTML(string=html_content).write_pdf(output_pdf_file)

print(f"The final report has been saved to {output_txt_file} and {output_pdf_file}")
