
## 🔄 Summary of Data Flow

```
1. User → Green Agent (A2A)
   Message: "Evaluate this purple agent"

2. Green Agent → MCP Server (HTTP)
   Request: "Initialize environment"
   
3. MCP Server → SmartHomeEnvironment (function call)
   Action: Create devices
   
4. SmartHomeEnvironment → MCP Server (return)
   Result: Environment ready

5. MCP Server → Green Agent (HTTP response)
   Result: {"status": "initialized", "device_count": 2}

6. Green Agent → MCP Server (HTTP)
   Request: "Evaluate task"

7. MCP Server → HomeBenchEvaluator (function call)
   Action: Run evaluation

8. HomeBenchEvaluator → Purple Agent (A2A)
   Message: "Turn on the living room light"

9. Purple Agent → HomeBenchEvaluator (A2A)
   Response: "living_room.light.turn_on()"

10. HomeBenchEvaluator → SmartHomeEnvironment (function call)
    Action: Execute operation

11. HomeBenchEvaluator → MetricsCalculator (function call)
    Action: Compute metrics

12. MetricsCalculator → HomeBenchEvaluator (return)
    Result: {EM: 1.0, P: 1.0, R: 1.0, F1: 1.0}

13. HomeBenchEvaluator → MCP Server (return)
    Result: TaskResult

14. MCP Server → Green Agent (HTTP response)
    Result: {"success": true, "score": 1.0}

15. Green Agent → User (A2A)
    Message: "Evaluation Complete!"
```

## 📊 Components Responsibility

| Component | What It Does | Already Done?  |
|-----------|--------------|----------------|
| **User** | Sends evaluation request | ✅ (You)        |
| **Green Agent** | Orchestrates workflow | ✅ Complete     |
| **MCPClient** | HTTP wrapper for tools | ✅ Complete     |
| **MCP Server** | Exposes tools via HTTP | ✅ Complete     |
| **SmartHomeEnvironment** | Manages device states | ❌To Implement  |
| **HomeBenchEvaluator** | Runs evaluation flow | ❌To Implement  |
| **AgentCommunicator** | Talks to purple agent | ❌To Implement  |
| **MetricsCalculator** | Computes scores | ❌To Implement  |
| **OperationParser** | Parses text | ❌ To Implement |
| **Purple Agent** | Executes tasks | ✅ (External)   |

