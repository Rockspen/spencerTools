import os
import difflib
from typing import Tuple
from dotenv import load_dotenv
import time

import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
EDITOR_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
CREATOR_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")

EDITOR_SYS_PROMPT = (
    "You are a meticulous senior editor.\n"
    "Task: Review the user's DRAFT and propose clear, actionable improvements\n"
    "to clarity, structure, tone, grammar, and originality.\n"
    "ensure any  story is brief and keep it to under 300 words."
    "Return TWO sections in Markdown exactly in this format:\n\n"
    "### SUGGESTIONS\n"
    "- Use bullet points. Make each suggestion concise but specific.\n"
    "- Only include changes that improve the piece.\n\n"
    "### REWRITTEN\n"
    "Provide a single improved version that applies your suggestions while\n"
    "preserving the author's intent and voice.\n"
)

CREATOR_SYS_PROMPT = (
    "You are a helpful content creator. Your task is to take a story suggestion from the user and write an original story based on it. You should then accept feedback and suggestions to refine the draft while preserving the author's intent. When asked to revise, return only the full revised content (no commentary)."
)

# --- Helper Functions ---

def parse_editor_response(text: str) -> Tuple[str, str]:
    """Extract SUGGESTIONS and REWRITTEN from the editor's markdown response."""
    lower = text.lower()
    sugg_idx = lower.find("### suggestions")
    rew_idx = lower.find("### rewritten")

    if sugg_idx == -1 or rew_idx == -1:
        return text.strip(), ""

    if sugg_idx < rew_idx:
        suggestions = text[sugg_idx + len("### suggestions"):rew_idx].strip()
        rewritten = text[rew_idx + len("### rewritten"):].strip()
    else:
        rewritten = text[rew_idx + len("### rewritten"):sugg_idx].strip()
        suggestions = text[sugg_idx + len("### suggestions"):].strip()

    return suggestions, rewritten

# --- LLM Clients ---

def make_llm(model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    """Factory function for creating LLM clients."""
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

# --- Chainlit Event Handlers ---

@cl.on_chat_start
async def on_chat_start():
    """Initializes the chatbot session."""
    try:
        cl.user_session.set("creator_llm", make_llm(CREATOR_MODEL, 0.5))
        cl.user_session.set("editor_llm", make_llm(EDITOR_MODEL, 0.2))
    except Exception as e:
        await cl.Message(content=f"Error initializing LLMs: {e}\nPlease check your API keys.").send()
        return

    # Initialize state
    cl.user_session.set("content", "")
    cl.user_session.set("suggestions", "")
    cl.user_session.set("edited_version", "")
    cl.user_session.set("context", "initial_idea")

    await cl.Message(
        content="**Welcome to the AI Authoring Assistant!**\n\n"
                "I'm here to help you write a story. "
                "Let's start with your idea."
    ).send()

    await cl.Message(
        content="What story would you like to create? Provide a suggestion or idea."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handles user messages for story creation, revision, and editing."""
    creator_llm = cl.user_session.get("creator_llm")
    context = cl.user_session.get("context", "initial_idea")
    content = cl.user_session.get("content", "")
    user_input = message.content

    # Clear context after receiving the message
    cl.user_session.set("context", None)

    if context == 'initial_idea':
        prompt = f"{CREATOR_SYS_PROMPT}\n\nHere is the user's suggestion: {user_input}\n\nPlease write a story based on this suggestion."
        msg = cl.Message(content="")
        await msg.send()

        async for chunk in creator_llm.astream(prompt):
            await msg.stream_token(chunk.content)

        new_content = msg.content
        cl.user_session.set("content", new_content)
        await cl.Message(content="Initial draft created. Now, let's get the editor's feedback...").send()
        await editor_review()

    elif context == 'revise':
        prompt = (
            f"{CREATOR_SYS_PROMPT}\n\n"
            f"Here is the current draft between <draft> tags. Revise it per the user instructions.\n\n"
            f"<draft>\n{content}\n</draft>\n\n"
            f"User instructions: {user_input}\n\n"
            f"Return only the revised draft."
        )
        msg = cl.Message(content="")
        await msg.send()

        async for chunk in creator_llm.astream(prompt):
            await msg.stream_token(chunk.content)

        revised_content = msg.content
        cl.user_session.set("content", revised_content)
        await cl.Message(content="Draft revised. Sending back to the editor...").send()
        await editor_review()

    elif context == 'edit':
        cl.user_session.set("content", user_input)
        await cl.Message(content="Draft updated with your edits. Let's get new feedback...").send()
        await editor_review()

async def editor_review():
    """
    Gets editor feedback on the current content, stores it in the session,
    and calls the function to display the results to the user.
    """
    editor_llm = cl.user_session.get("editor_llm")
    content = cl.user_session.get("content", "")

    if not content:
        return

    prompt = (
        f"{EDITOR_SYS_PROMPT}\n\n"
        f"Here is the DRAFT between <draft> tags.\n\n"
        f"<draft>\n{content}\n</draft>\n"
    )

    async with cl.Step(name="Editor Review") as step:
        step.input = prompt
        response = await editor_llm.ainvoke(prompt)
        suggestions, rewritten = parse_editor_response(response.content)

        cl.user_session.set("suggestions", suggestions)
        cl.user_session.set("edited_version", rewritten)
        step.output = f"Suggestions: {suggestions}\n\nRewritten: {rewritten}"

    await display_editor_results()

async def display_editor_results():
    """Displays the editor's feedback and action buttons to the user."""
    suggestions = cl.user_session.get("suggestions", "")
    edited_version = cl.user_session.get("edited_version", "")

    if not suggestions and not edited_version:
        await cl.Message(content="The editor had no suggestions. What would you like to do next?").send()
        return

    suggestions_msg = f"**Editor's Suggestions:**\n\n{suggestions}"
    await cl.Message(content=suggestions_msg).send()

    if edited_version:
        preview = (edited_version[:600] + "..." if len(edited_version) > 600 else edited_version)
        await cl.Message(
            content=f"**Editor's Rewrite Preview:**\n\n{preview}"
        ).send()

    actions = [
        cl.Action(name="accept", value="accept", label="✅ Accept Rewrite"),
        cl.Action(name="revise", value="revise", label="✍️ Revise with AI"),
        cl.Action(name="edit", value="edit", label="📝 Edit Manually"),
        cl.Action(name="diff", value="diff", label="🔄 Show Diff"),
        cl.Action(name="finish", value="finish", label="🏁 Finish & Save"),
    ]
    await cl.Message(content="What would you like to do next?", actions=actions).send()


@cl.action_callback("accept")
async def on_accept(action: cl.Action):
    edited_version = cl.user_session.get("edited_version")
    if edited_version:
        cl.user_session.set("content", edited_version)
        await cl.Message(content="Accepted the editor's rewrite. Rerunning review...").send()
        await editor_review()
    else:
        await cl.Message(content="No rewritten version available to accept.").send()

@cl.action_callback("revise")
async def on_revise(action: cl.Action):
    cl.user_session.set("context", "revise")
    await cl.Message(
        content="Describe how you'd like the AI to revise the draft (e.g., change the tone, shorten it, target a different audience)."
    ).send()

@cl.action_callback("edit")
async def on_edit(action: cl.Action):
    cl.user_session.set("context", "edit")
    await cl.Message(
        content="Please paste your full edited version of the draft below."
    ).send()

@cl.action_callback("diff")
async def on_diff(action: cl.Action):
    content = cl.user_session.get("content", "")
    edited_version = cl.user_session.get("edited_version", "")

    if not edited_version:
        await cl.Message(content="No editor version to diff against.").send()
        return

    diff_result = show_diff(content, edited_version)
    await cl.Message(content=f"**Diff:**\n```diff\n{diff_result}\n```").send()
    # Re-display options
    await display_editor_results()

@cl.action_callback("finish")
async def on_finish(action: cl.Action):
    content = cl.user_session.get("content", "")
    if not content:
        await cl.Message(content="Nothing to save!").send()
        return

    await cl.Message(content=f"**Final Content:**\n\n{content}").send()

    # Save to file
    fname = time.strftime("draft_%Y-%m-%d_%H%M.md")
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        abs_path = os.path.abspath(fname)
        await cl.Message(content=f"Saved ✅  `{abs_path}`").send()
    except Exception as e:
        await cl.Message(content=f"Failed to save file: {e}").send()


def show_diff(a: str, b: str) -> str:
    """Generates a unified diff string."""
    a_lines = a.splitlines(keepends=False)
    b_lines = b.splitlines(keepends=False)
    diff = difflib.unified_diff(a_lines, b_lines, fromfile="current", tofile="editor", lineterm="")
    return "\n".join(diff)
