# Agent Architecture
Intelligent Agents can be simply defined as *something that can act* 

+ Acting requires some capacity for deciding what to do
+ Deciding what to do implies having access to more than one possible course of action. A decision without options is no decision at all
+ In order to decide, the agent also needs access to information about the external environment (anything outside the agent itself)

So an *agentic LLM application* must be one that uses an LLM to pick from one or more possible courses of action, given some context about the current *state* of the environment and some desired next *state*. 

These attributes are usually implemented by mixing two prompting techniques:
+ Tool Calling - 
    + Include a list of external functions that the LLM can make use of in your prompt. That becomes an *action* it can decide to take. 
+ Chain of Thought - 
    + LLMs make better decisions when given instructions to reason abaout complex problems by breaking them down into granular steps to be taken in sequence (*think step by step*). 

## The **PLan-DO Loop**
Generally by *loop* we mean *running* the same *code* multiple times until a stop condition is met. 

What we run in this *loop* will be some variation of the following
+ Planning an action or actions
+ Executing above said actions. 

## Extending the **Plan-DO Loop** Architecture
We can further improve performance for some use cases by extending the previous *Agent Architecture* which was based on *Tool Calling and Chain of Thought*. 

The two extentions are
+ Reflection
    + Giving the LLM App the opportunity to analyze its past output and choices, together with the ability to remember reflections from past iterations
+ Multi-Agent
    + A *team of agents* working on a single problem. 

### Reflection
*Reflection* is another *prompting technique* also known as *self-critique*. *Reflection* is the creation of a loop between a *creator prompt* and a *reviser prompt*. *Reflection* can be combined with *Chain of Thought* and *Tool Calling*. 
