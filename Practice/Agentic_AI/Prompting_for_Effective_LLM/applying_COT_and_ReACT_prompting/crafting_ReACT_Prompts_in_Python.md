A good ReAct system prompt has four key parts:

The Role and Goal: Who is the agent? What is its purpose?
The THINK/ACT Instruction: How must the agent format its reasoning and actions?
The Tool Definitions: What tools can the agent use, and how do they work?
A Complete Example: A full example of a multi-turn interaction.

1. Define the Role and Goal
We start by telling the agent its identity and overall mission.

You are a Supply Chain Logistics Coordinator. Your goal is to diagnose shipment delays by gathering information from different systems.

2. Explain the THINK/ACT Cycle
Next, we give it explicit instructions on how to structure its response. This is the core of the ReAct framework.

You must use a step-by-step reasoning process. For each step, respond with a single THINK/ACT message.

THINK: First, you will reason about the problem and determine the next logical action to take.
ACT: Based on your thought process, you will call ONE of the available tools.

3. Define the Available Tools
You must clearly list every tool the agent can use. For each tool, you must provide its name, its parameters, a short description, and an example of its input (ACT) and output (OBSERVE).

---
## Available Tools

1.  **get_shipment_status(tracking_id: str)**
    * Use this to get the last known location and status of a shipment.
    * Example Input: `ACT: get_shipment_status(tracking_id="XYZ123")`
    * Example Output: `OBSERVE: {"status": "Delayed", "location": "Chicago Rail Yard"}`

2.  **check_facility_alerts(facility_name: str)**
    * Use this to check for operational alerts (e.g., weather delays, closures) at a specific facility.
    * Example Input: `ACT: check_facility_alerts(facility_name="Chicago Rail Yard")`
    * Example Output: `OBSERVE: {"alert": "Severe Weather Alert: All operations suspended."}`

3.  **final_answer(summary: str)**
    * Use this tool ONLY when you have diagnosed the problem and can provide a complete summary.
    * Example Input: `ACT: final_answer(summary="Shipment XYZ123 is delayed in Chicago due to a severe weather-related closure at the rail yard.")`

4. Provide a Complete Example Interaction
Finally, you put it all together by showing the agent a complete, multi-step example of how to use these components to solve a problem. This few-shot example is often the most effective way for the agent to learn the pattern.

---
## Example

**User:** Find out why shipment XYZ123 is delayed.

**AI Assistant:**
THINK: I need to find out why shipment XYZ123 is delayed. My first step is to get the current status and location of the shipment using the `get_shipment_status` tool.
ACT: get_shipment_status(tracking_id="XYZ123")

**(System provides `OBSERVE: {"status": "Delayed", "location": "Chicago Rail Yard"}` and the AI continues...)**
