from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

@tool
def calculator(a:float, b:float) -> str:
    """
    Useful for performing basic arithmeric calculations with numbers.
    """
    print('Tool has been called')
    if a is None or b is None:
        return "Error: Missing parameters 'a' or 'b'."

    try:
        return f'The sum of {a} and {b} is {a+b}'
    except ValueError:
        return "Error: Both 'a' and 'b' must be numbers."
    
@tool
def hello(username:str) -> str:
    '''
    Usefull for gretting a user
    '''
    print('Tool has been called')
    return f'Hello {username}, I hope you are well today'

# Función principal
def main():
    print("Starting AI Calculator Agent...")
    print("You can ask me to perform basic calculations.")
    print("Type 'quit' anytime to exit.\n")

    # Modelo base y herramientas
    model = ChatOpenAI(temperature=0)
    tools = [calculator, hello]

    # Crear el agente estilo ReAct
    agent_executor = create_react_agent(model, tools)

    # Bucle de interacción
    while True:
        user_input = input('You: ').strip()
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break

        print('Assistant: ', end='')

        try:
            for chunk in agent_executor.stream({'messages': [HumanMessage(content=user_input)]}):
                if 'agent' in chunk and 'messages' in chunk['agent']:
                    for message in chunk['agent']['messages']:
                        print(message.content, end='')
        except Exception as e:
            print("\nOcurrió un error al procesar tu solicitud.")
            print("Detalle:", e)

        print()

if __name__ == '__main__':
    main()