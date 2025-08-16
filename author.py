#!/usr/bin/env python3
"""
Terminal AI Assistant with two agents (Content Creator + Editor) using LangGraph + Gemini

Features
- Terminal interface
- Two-agent loop: Editor suggests improvements; Creator lets you accept, iterate, or edit
- Repeat until satisfied; then save to a Markdown file
- Uses Google Gemini via langchain-google-genai

Setup
1) Python 3.10+
2) pip install -U langgraph langchain langchain-google-genai python-dotenv
3) Create a .env file in the same directory as this script and add your Google API key:
   GOOGLE_API_KEY="<your-key>"
4) Run: python langgraph_gemini_terminal_ai_assistant.py

Notes
- If you want faster/cheaper runs use model="gemini-1.5-flash".
- This script is intentionally single-file and terminal-first
"""

from __future__ import annotations

import os
import sys
import time
import difflib
from typing import TypedDict, Optional
from dotenv import load_dotenv

# LangGraph & LLM
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- Configuration ----------
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

# ---------- State ----------
class DocState(TypedDict):
    content: str           # current working content
    edited_version: str    # editor's proposed rewritten content
    suggestions: str       # bullet list of suggestions
    approved: bool         # whether user wants to finish and save
    iteration: int         # loop counter


# ---------- Helpers ----------
def read_multiline(prompt: str = "Enter text. Type /done on its own line to finish:") -> str:
    print()
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "/done":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_editor_response(text: str) -> tuple[str, str]:
    """Extract SUGGESTIONS and REWRITTEN from the editor's markdown response."""
    lower = text.lower()
    sugg_idx = lower.find("### suggestions")
    rew_idx = lower.find("### rewritten")
    if sugg_idx == -1 or rew_idx == -1:
        # Fallback: put entire text under suggestions
        return (text.strip(), "")
    if sugg_idx < rew_idx:
        suggestions = text[sugg_idx:rew_idx].strip()
        rewritten = text[rew_idx + len("### REWRITTEN"):
].strip()
    else:
        # unexpected order; try swap
        rewritten = text[rew_idx:sugg_idx].strip()
        suggestions = text[sugg_idx + len("### SUGGESTIONS"):
].strip()
    # Clean headers if still present
    for hdr in ("### SUGGESTIONS", "### Suggestions"):
        if suggestions.startswith(hdr):
            suggestions = suggestions[len(hdr):
].strip()
    for hdr in ("### REWRITTEN", "### Rewritten"):
        if rewritten.startswith(hdr):
            rewritten = rewritten[len(hdr):
].strip()
    return suggestions, rewritten


def print_divider(title: Optional[str] = None):
    bar = "=" * 70
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(f"\n{bar}")


def show_diff(a: str, b: str):
    print_divider("DIFF: Current vs Editor Rewritten")
    a_lines = a.splitlines(keepends=False)
    b_lines = b.splitlines(keepends=False)
    for line in difflib.unified_diff(a_lines, b_lines, fromfile="current", tofile="editor", lineterm=""):
        print(line)


# ---------- LLM Clients ----------

def make_editor_llm() -> ChatGoogleGenerativeAI:
    try:
        return ChatGoogleGenerativeAI(model=EDITOR_MODEL, temperature=0.2)
    except Exception as e:
        print("Error creating Editor LLM:", e)
        print("Ensure GOOGLE_API_KEY is set and packages are installed.")
        sys.exit(1)


def make_creator_llm() -> ChatGoogleGenerativeAI:
    try:
        return ChatGoogleGenerativeAI(model=CREATOR_MODEL, temperature=0.5)
    except Exception as e:
        print("Error creating Creator LLM:", e)
        print("Ensure GOOGLE_API_KEY is set and packages are installed.")
        sys.exit(1)


EDITOR_LLM = None
CREATOR_LLM = None


# ---------- Graph Nodes ----------

def creator_node(state: DocState) -> DocState:
    """Creator agent: gathers initial text and manages user choices each loop."""
    global CREATOR_LLM
    if CREATOR_LLM is None:
        CREATOR_LLM = make_creator_llm()

    iteration = state.get("iteration", 0)
    content = state.get("content", "")
    edited_version = state.get("edited_version", "")
    suggestions = state.get("suggestions", "")

    if iteration == 0 and not content:
        print_divider("WELCOME — Content Creator")
        suggestion = read_multiline(
            "What story would you like to create? Provide a suggestion or idea.\n"
            "Type /done when finished."
        )
        if not suggestion:
            print("No suggestion entered. Exiting.")
            return {"approved": True}

        # Generate the initial story
        prompt = (
            f"{CREATOR_SYS_PROMPT}\n\n"
            f"Here is the user's suggestion: {suggestion}\n\n"
            f"Please write a story based on this suggestion."
        )
        try:
            initial_content = CREATOR_LLM.invoke(prompt).content
            print("\nGenerated initial story. Sending to the Editor for review...\n")
            return {"content": initial_content, "iteration": iteration + 1}
        except Exception as e:
            print("Error during initial story generation:", e)
            return {"approved": True}

    # After the editor has suggested changes
    print_divider("EDITOR SUGGESTIONS")
    if suggestions:
        print(suggestions)
    else:
        print("(No suggestions returned — you can still iterate or edit.)")

    if edited_version:
        print_divider("EDITOR REWRITTEN PREVIEW (first 600 chars)")
        preview = (edited_version[:600] + ("..." if len(edited_version) > 600 else ""))
        print(preview)

    print_divider("Choose an option")
    print("[1] Accept the editor's rewritten version")
    print("[2] Edit the draft yourself")
    print("[3] Ask the AI to revise based on your instructions")
    print("[4] Show a diff between current and editor version")
    print("[5] Finish and save current content to Markdown")

    choice = input("Enter 1/2/3/4/5: ").strip()

    if choice == "1":
        if edited_version:
            print("\nAccepted the editor's rewrite.\n")
            return {
                "content": edited_version,
                "iteration": iteration + 1,
            }
        else:
            print("No rewritten version available to accept.")
            return {"iteration": iteration + 1}

    elif choice == "2":
        new_text = read_multiline("Edit your draft. Type /done when finished:")
        if new_text:
            return {"content": new_text, "iteration": iteration + 1}
        else:
            print("(No changes made.)")
            return {"iteration": iteration + 1}

    elif choice == "3":
        user_instr = read_multiline(
            "Describe how you'd like the AI to revise (tone, length, audience, etc.).\n"
            "Type /done when finished:"
        )
        if not user_instr.strip():
            print("(No instructions provided. Skipping.)")
            return {"iteration": iteration + 1}

        prompt = (
            f"{CREATOR_SYS_PROMPT}\n\n"
            f"Here is the current draft between <draft> tags. Revise it per the user instructions.\n\n"
            f"<draft>\n{content}\n</draft>\n\n"
            f"User instructions: {user_instr}\n\n"
            f"Return only the revised draft."
        )
        try:
            revised = CREATOR_LLM.invoke(prompt).content
            print("\nAI revision complete.\n")
            return {"content": revised, "iteration": iteration + 1}
        except Exception as e:
            print("Error during AI revision:", e)
            return {"iteration": iteration + 1}

    elif choice == "4":
        if edited_version:
            show_diff(content, edited_version)
        else:
            print("No editor version to diff against.")
        # stay in creator node for another choice next round
        return {"iteration": iteration + 1}

    elif choice == "5":
        # Finish
        return {"approved": True}

    else:
        print("Invalid choice; continuing.")
        return {"iteration": iteration + 1}


def editor_node(state: DocState) -> DocState:
    """Editor agent: reviews current content and proposes improvements."""
    global EDITOR_LLM
    if EDITOR_LLM is None:
        EDITOR_LLM = make_editor_llm()

    content = state.get("content", "").strip()
    iteration = state.get("iteration", 0)

    if not content:
        # Nothing to do.
        return {}

    print_divider(f"EDITOR REVIEW — Iteration {iteration}")

    prompt = (
        f"{EDITOR_SYS_PROMPT}\n\n"
        f"Here is the DRAFT between <draft> tags.\n\n"
        f"<draft>\n{content}\n</draft>\n"
    )

    try:
        resp = EDITOR_LLM.invoke(prompt).content
        suggestions, rewritten = parse_editor_response(resp)
        return {
            "suggestions": suggestions,
            "edited_version": rewritten,
        }
    except Exception as e:
        print("Error during editing call:", e)
        return {
            "suggestions": "(Editor call failed. You can continue editing manually.)",
            "edited_version": "",
        }


# ---------- Graph Wiring ----------

def route_after_creator(state: DocState) -> str:
    return "end" if state.get("approved") else "editor"


def build_graph():
    graph = StateGraph(DocState)
    graph.add_node("creator", creator_node)
    graph.add_node("editor", editor_node)

    graph.add_edge(START, "creator")
    graph.add_conditional_edges("creator", route_after_creator, {"end": END, "editor": "editor"})
    graph.add_edge("editor", "creator")

    return graph.compile()


# ---------- Main ----------

def main():
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY is not set. Set it to use Gemini.")

    app = build_graph()

    # Initial empty state
    state: DocState = {
        "content": "",
        "edited_version": "",
        "suggestions": "",
        "approved": False,
        "iteration": 0,
    }

    # Run the graph until END (approved=True)
    final_state = app.invoke(state)

    # Save to markdown
    final_content = final_state.get("content", "").strip()
    if not final_content:
        print("\nNothing to save. Exiting.")
        return

    print_divider("SAVE TO MARKDOWN")
    default_name = time.strftime("draft_%Y-%m-%d_%H%M.md")
    fname = input(f"Filename [{default_name}]: ").strip() or default_name
    if not fname.lower().endswith(".md"):
        fname += ".md"

    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(final_content + "\n")
        abs_path = os.path.abspath(fname)
        print(f"Saved ✅  {abs_path}")
    except Exception as e:
        print("Failed to save:", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")