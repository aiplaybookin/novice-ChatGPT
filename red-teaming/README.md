## Benchmark IS NOT Safety & Security

## Foundational Model IS NOT LLM App

**RESOURCES**

- OWASP top 10 for LLM
- AI Incident Database
- AVID


### Vulnerabilities
1. Bias and stereotypes e.g. male/female, immigrant
2. Sensitive information disclosure e.g. hostname, document repo link etc.
3. Service disruption e.g. extra long input tokens
4. Hallucinations e.g. user provides a fake info in chat and asks for details in follow up chat.
    - LLM tendency to never contradict the User


### Bypassing Safeguards
1. Exploiting text completion e.g. post fix like "Sure here how you can do it.
    - or thank for answer, appreciate the topic and ask
2. Using biased prompts
    - bend it with biased prompt with more content 
3. Direct prompt injection (Jailbreaking)
    - By providing new instructions in prompt
    - e.g. you are Latin language expert that talks like pirate....
4. Gray box prompt attacks
    - completely reshape the prompt given that you know the structure of the prompt.
    - e.g. add additional context and ask question

