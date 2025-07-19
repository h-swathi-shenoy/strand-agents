import uvicorn
import os
from strands import Agent
from strands.models import BedrockModel
from tika import parser
from fastapi import  FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

app = FastAPI()

bedrock_model = BedrockModel(
  model_id="amazon.nova-lite-v1:0", 
  region_name="us-east-1",
  temperature=0.3,
  streaming=True, # Enable/disable streaming
)


job_description_agent = Agent(
    system_prompt=(f"""
      You are an experienced resume evaluator. Extract the componenets of the resume in the following format
        
        Extract all Must Have Responsibilities:
        -Technical Skills: Essential technical skills
        -Experience: Minimum years and type of relevant experience
        -Qualifications: Mandatory degrees or certifications
        -Core Responsibilities: Key duties that are non-negotiable


        -Extract all Good-to-Have Requirements:

        -Additional Skills: Preferred technical skills
        -Extra Qualifications: Bonus education or certifications
        -Bonus Experience: Extra experience that could add value


        Extract any Additional Screening Criteria:

        -Filtering statements that affect eligibility
        -Work policy conditions
        -Availability constraints
        -Discriminatory or biased phrasing
        -Anything else that significantly affects who should or shouldn't apply

    
        """
    ), model = bedrock_model
)


resume_jd_agent_quantative= Agent(
system_prompt = (f"""
You are a recruiter evaluating a candidate's resume against a given job_description json inputs. Compare the candidates resume against given job description. Evaluate whether the candidate meets the necessary requirements.

Step 1: Quantitative Check
Perform a Boolean (true/false) check for each requirement based on the candidate's resume:

-For each skill listed in must_have_requirements and good_to_have_requirements of jobdescription, determine if the candidate possesses it. Return true or false for each.
-For each core_responsibility mentioned in jobdescription, determine if the candidate resume has demonstrated it in their past work. Return true or false.
-For each additional_screening_criteria in jobdescription, return a boolean value indicating whether the candidates resume meets the condition (e.g., full-time, onsite position, work authorization, etc.).

"""),
model = bedrock_model
)


resume_jd_agent_qualitativecheck = Agent(
system_prompt = ("""

Step 2: Qualitative Check
Now, switch to a recruiter-style qualitative assessment. Use your intuition like a human — go beyond what's explicitly stated. Read between the lines, infer intent, and use contextual clues from the resume and the JD to judge fit. Reference the results from Step 1 as part of your reasoning.

    Assess the following:

    -Inferred Skills: What skills can you infer from the candidate's projects or roles?
    -Project Gravity: Were the projects academic or real-world, high-impact, production-ready, etc.?
    -Ownership and Initiative: Did the candidate lead the work? Show initiative? Or just follow directions?
    -Transferability to Role: How well would their experience transfer to this particular role? Will they onboard quickly?
    -Bonus Experience & Extra Qualifications: If the JD lists any bonus criteria (e.g., fintech, B2B SaaS), consider that a positive signal even if not part of Step 1.
    -Recruiter Style Summary: Provide a recruiter style summary of the full profile of the candidate, consider the results of both the quantitative and qualitative assessment.

"""),model = bedrock_model
)


resume_finalrecommendation = Agent(
system_prompt = ("""

Step 2: Final Recommendation
Now, as a recruiter, take the final call if the candidate is the right fit for the role.Based on the summary provided. Keep the tone practitcal.


"""),model = bedrock_model
)


def parse_pdf(document_path:str):
    """
    Parse the pdf to string
    """
    parsed_pdf = parser.from_file(document_path)
    str_content = parsed_pdf['content'] 
    return str_content


class MustHaveRequirementSkills(BaseModel):
    """"""
    technical_skills: str
    experience: str
    qualifications: List[str]
    core_responsibilities: List[str]
            
class GoodToHaveRequirementSkills(BaseModel):
    """"""
    additional_skills: List[str]
    experience: str
    qualifications: List[str]
    core_responsibilities: List[str]

class AdditionalScreeningCriteria(BaseModel):
    """"""
    workpolicy_conditions: str
    general_constrains: List[str]


class JobDescripiton(BaseModel):
    """Schema for Job Description"""
    original_job_description: str = Field(description="The complete original job description text")
    musthave_requirementskills: List[MustHaveRequirementSkills] = Field(description="List of must have requirement skills")
    goodtohave_requirementskills: List[GoodToHaveRequirementSkills]= Field(description="List of good to have requirement skills")
    additional_screening_criteria: List[AdditionalScreeningCriteria] = Field(description = "List of additional screening Criteria")


def multi_agent_process_workflow(parsed_jd, parsed_resume):

    job_description_response = job_description_agent.structured_output(JobDescripiton, parsed_jd)
    
    # Step 1: Qualitative Resume Analysis

    append_para = [str(job_description_response), parsed_resume]
    jd_resume = "\n\n".join(append_para)

    resume_agent_quanitativecheck = resume_jd_agent_quantative(jd_resume)


    #Step 2: Qualitative Resume Analysis
    resume_agent_qualitativecheck = resume_jd_agent_qualitativecheck(str(resume_agent_quanitativecheck))

    # Step 3: Final Report
    final_report = resume_finalrecommendation(str(resume_agent_qualitativecheck))
    for chunk in final_report.message['content'][0]['text']:
        yield chunk

        
@app.post("/events")
async def events(request: Request):
    query_json = await request.json()
    parsed_job_description= query_json['job_description']
    parsed_resume_description = query_json['resume']
    return StreamingResponse(multi_agent_process_workflow(parsed_job_description, parsed_resume_description), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))

