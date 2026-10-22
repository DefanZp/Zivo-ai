from agent import create_zivo_agent

class ZivoAgentService:

    def __init__(self, checkpointer):
        self.agent = create_zivo_agent(checkpointer)

    def chat(self, thread_id, user_message, cart_api_token):
        
        config = {
            "configurable": {
                "thread_id": thread_id  
            }
        }

        return self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            },
            config,
            context={
                "cart_api_token": cart_api_token
            }
        )

