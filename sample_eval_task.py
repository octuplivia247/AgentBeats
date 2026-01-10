import asyncio
from src.utils.my_a2a import send_message, wait_agent_ready

async def test():
    url = 'http://localhost:9001'  # Green Agent URL
    
    # 1. Wait for the Green Agent to be ready (max 5 seconds)
    if await wait_agent_ready(url, timeout=5):
        
        # 2. Compose an evaluation request message with XML tags
        msg = '''<purple_agent_url>http://localhost:9000</purple_agent_url>
<evaluation_config>{"task_ids": [0, 1, 2]}</evaluation_config>'''
        
        # 3. Send the message to the Green Agent
        resp = await send_message(url, msg)
        print(resp)
    else:
        print('Agent not ready')


if __name__ == "__main__":
    asyncio.run(test())