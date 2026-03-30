from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATION = 10
MODEL = "qwen3:1.7b"

# ------ Langchain tools decorators ------


@tool()
def get_product_price(product: str) -> float:
    """Look the product price in the catalog"""
    print(f"  >> Executing get_product_price({product})")

    prices = {"laptop": 1299.23, "headphone": 149.42, "keyboard": 89.40}
    return prices.get(product, 0)


@tool()
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply the discount tier to a price and return final result
    Available Tier : bronze , sliver, gold
    """

    print(f"  >> Executing apply_discount(price={price} discount_tier:{discount_tier})")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 22}

    discount = discount_percentage.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# ------ Agent Loop ------


# @traceable(name="Langchain-Agent-Loop")
def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}

    # llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0)
    llm = init_chat_model(model=f"google_genai:gemini-2.5-flash-lite", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=query),
    ]

    #  for loop to iterate over all AI Messages
    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n ------ Iteration {iteration} -------")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # No tool calls means the LLM has a final answer
        if not tool_calls:
            print(f"\n Final Answer =======>  {ai_message.content}")
            break

        # Append ai_message ONCE before processing all tool calls
        messages.append(ai_message)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")
            tool_call_id = tool_call.get("id")

            print(f"[Tool Selected] {tool_name} with args {tool_args}")

            tool_to_use = tool_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' not found")

            observation = tool_to_use.invoke(tool_args)
            print(f"[Tool Result]: {observation}")

            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )


if __name__ == "__main__":
    print("Hello langchain agents (.bind_tools)")
    print()
    result = run_agent("what is laptop price and check gold discount")
    print()
