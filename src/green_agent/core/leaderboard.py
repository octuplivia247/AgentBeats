import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AgentScore:
    """Represents an agent's evaluation score"""
    agent_id: str
    unique_id: int
    accuracy: float  # avg_exact_match
    avg_precision: float
    avg_recall: float
    avg_f1: float
    total_tasks: int
    perfect_tasks: int
    
    @classmethod
    def from_evaluation_result(cls, result: Dict[str, Any]) -> 'AgentScore':
        """Create AgentScore from evaluation result dictionary"""
        eval_score = result['agent_evaluation_score']
        return cls(
            agent_id=result['agent_id'],
            unique_id=result['unique_id'],
            accuracy=eval_score['avg_exact_match'],
            avg_precision=eval_score['avg_precision'],
            avg_recall=eval_score['avg_recall'],
            avg_f1=eval_score['avg_f1'],
            total_tasks=eval_score['total_tasks'],
            perfect_tasks=eval_score['perfect_tasks']
        )


class Leaderboard:
    """
    Manages agent leaderboard rankings based on evaluation scores.
    
    Features:
    - Load evaluation results from JSON files
    - Rank agents by accuracy (avg_exact_match)
    - Display top N agents
    - Send leaderboard data to API endpoint
    
    Usage:
        leaderboard = Leaderboard()
        leaderboard.load_from_file('evaluation_results.json')
        leaderboard.add_from_file('another_agent_results.json')
        top_10 = leaderboard.get_top_agents(10)
        leaderboard.display_leaderboard(10)
        leaderboard.send_to_api('https://api.example.com/leaderboard')
    """
    
    def __init__(self):
        """Initialize empty leaderboard"""
        self.agents: List[AgentScore] = []
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load agent evaluation results from JSON file.
        Handles both single agent object and array of agents.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both list of agents and single agent object
        if isinstance(data, list):
            for agent_data in data:
                agent_score = AgentScore.from_evaluation_result(agent_data)
                self.agents.append(agent_score)
                print(f"✓ Loaded agent: {agent_score.agent_id} (Accuracy: {agent_score.accuracy:.2%})")
        else:
            # Single agent object
            agent_score = AgentScore.from_evaluation_result(data)
            self.agents.append(agent_score)
            print(f"✓ Loaded agent: {agent_score.agent_id} (Accuracy: {agent_score.accuracy:.2%})")
    
    def load_from_directory(self, dirpath: str, pattern: str = "*.json") -> None:
        """
        Load all evaluation results from a directory.
        """
        directory = Path(dirpath)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {dirpath}")
        
        json_files = list(directory.glob(pattern))
        
        if not json_files:
            print(f"⚠ No files matching '{pattern}' found in {dirpath}")
            return
        
        print(f"\nLoading {len(json_files)} evaluation files...")
        loaded = 0
        for filepath in json_files:
            try:
                self.load_from_file(str(filepath))
                loaded += 1
            except Exception as e:
                print(f"✗ Error loading {filepath.name}: {e}")
        
        print(f"\n✓ Successfully loaded {loaded}/{len(json_files)} agents")
    
    def add_from_file(self, filepath: str) -> None:
        """Alias for load_from_file for consistency"""
        self.load_from_file(filepath)
    
    def add_agent(self, agent_score: AgentScore) -> None:
        """
        Add an agent score directly (useful for programmatic usage).
        """
        self.agents.append(agent_score)
    
    def get_ranked_agents(self) -> List[AgentScore]:
        """
        Get all agents ranked by accuracy (descending).
        """
        return sorted(self.agents, key=lambda x: x.accuracy, reverse=True)
    
    def get_top_agents(self, n: int = 10) -> List[AgentScore]:
        """
        Get top N agents by accuracy.
        """
        ranked = self.get_ranked_agents()
        return ranked[:n]
    
    def display_leaderboard(self, n: int = 10, show_details: bool = False) -> None:
        """
        Display formatted leaderboard in console.
        """
        top_agents = self.get_top_agents(n)
        
        if not top_agents:
            print("⚠ No agents in leaderboard")
            return
        
        print("\n" + "="*80)
        print(f"🏆 TOP {min(n, len(top_agents))} AGENTS LEADERBOARD")
        print("="*80)
        
        # Header
        if show_details:
            print(f"{'Rank':<6} {'Agent ID':<25} {'Accuracy':<12} {'F1':<10} {'Perfect/Total'}")
            print("-"*80)
        else:
            print(f"{'Rank':<6} {'Agent ID':<40} {'Accuracy':<12} {'Perfect/Total'}")
            print("-"*80)
        
        # Agent rows
        for rank, agent in enumerate(top_agents, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            
            if show_details:
                print(f"{emoji} {rank:<4} {agent.agent_id:<25} {agent.accuracy:>10.2%}  "
                      f"{agent.avg_f1:>8.2%}  {agent.perfect_tasks}/{agent.total_tasks}")
            else:
                print(f"{emoji} {rank:<4} {agent.agent_id:<40} {agent.accuracy:>10.2%}  "
                      f"{agent.perfect_tasks}/{agent.total_tasks}")
        
        print("="*80 + "\n")
    
    def to_json(self, n: int = 10) -> str:
        """
        Convert top N agents to JSON string.
        """
        top_agents = self.get_top_agents(n)
        leaderboard_data = {
            "leaderboard": [
                {
                    "rank": rank,
                    "agent_id": agent.agent_id,
                    "accuracy": agent.accuracy,
                    "unique_id": agent.unique_id,
                    "perfect_tasks": agent.perfect_tasks,
                    "total_tasks": agent.total_tasks
                }
                for rank, agent in enumerate(top_agents, 1)
            ],
            "total_agents": len(self.agents),
            "displayed": len(top_agents)
        }
        return json.dumps(leaderboard_data, indent=2)
    
    def save_leaderboard(self, filepath: str, n: int = 10) -> None:
        """
        Save top N agents leaderboard to JSON file.
        """
        leaderboard_json = self.to_json(n)
        
        with open(filepath, 'w') as f:
            f.write(leaderboard_json)
        
        print(f"✓ Leaderboard saved to: {filepath}")
    
    def send_to_api(
        self, 
        url: str, 
        n: int = 10, 
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ) -> bool:
        """
        Send top N agents leaderboard to API endpoint.
        """
        leaderboard_json = self.to_json(n)
        
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        
        try:
            print(f"\n📤 Sending leaderboard to: {url}")
            response = requests.post(
                url,
                data=leaderboard_json,
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                print(f"✓ Successfully sent leaderboard (Status: {response.status_code})")
                return True
            else:
                print(f"✗ Failed to send leaderboard (Status: {response.status_code})")
                print(f"  Response: {response.text[:200]}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"✗ Connection error: Could not reach {url}")
            print("  (This is expected if using a dummy URL)")
            return False
        except requests.exceptions.Timeout:
            print(f"✗ Request timeout after {timeout} seconds")
            return False
        except Exception as e:
            print(f"✗ Error sending leaderboard: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of all agents.
        """
        if not self.agents:
            return {
                "total_agents": 0,
                "avg_accuracy": 0.0,
                "max_accuracy": 0.0,
                "min_accuracy": 0.0
            }
        
        accuracies = [agent.accuracy for agent in self.agents]
        
        return {
            "total_agents": len(self.agents),
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "max_accuracy": max(accuracies),
            "min_accuracy": min(accuracies),
            "top_agent": self.get_top_agents(1)[0].agent_id if self.agents else None
        }


# Example usage / testing
if __name__ == "__main__":
    # Initialize leaderboard
    leaderboard = Leaderboard()
    
    # Load from evaluation_results.json (where actual results are stored)
    try:
        leaderboard.load_from_file('evaluation_results.json')
        
        # Display leaderboard
        leaderboard.display_leaderboard(n=10, show_details=True)
        
        # Get statistics
        stats = leaderboard.get_statistics()
        print("📊 Leaderboard Statistics:")
        print(f"  Total Agents: {stats['total_agents']}")
        print(f"  Average Accuracy: {stats['avg_accuracy']:.2%}")
        print(f"  Top Agent: {stats['top_agent']}\n")
        
        # Save leaderboard to file
        leaderboard.save_leaderboard('leaderboard_top10.json', n=10)
        
        # Send to dummy API
        leaderboard.send_to_api('https://api.example.com/leaderboard/submit', n=10)
        
    except FileNotFoundError:
        print("✗ Error: evaluation_results.json not found!")
        print("  Please ensure evaluation results have been generated first.")
        print("  Run your evaluation code to create evaluation_results.json")
    except KeyError as e:
        print(f"✗ Error: Missing required field in evaluation_results.json: {e}")
        print("  Please check the file format matches the expected structure.")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        