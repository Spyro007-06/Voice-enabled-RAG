When clearing a conversation, if focus was inside `chatMessages`, it will be lost because all elements in `chatMessages` are deleted. This violates "To prevent accessibility focus loss when dynamically replacing child elements (e.g., re-rendering tabs or lists with replaceChildren)... explicitly restore focus to the newly created element at that same index afterwards." but in this case, we're clearing, so there's no new element.
We should check if `chatMessages.contains(document.activeElement)`, and if so, shift focus somewhere logical (e.g., `elements.queryInput.focus()`).

Let's look at `frontend/js/app.js`:
```javascript
  // 8. New / Clear Chat
  const resetConversation = () => {
    // If focus is in chat messages, move it before we destroy it
    if (elements.chatMessages && elements.chatMessages.contains(document.activeElement)) {
      if (elements.queryInput) elements.queryInput.focus();
    }
    if (elements.chatMessages) elements.chatMessages.replaceChildren();
    if (elements.welcomeState) elements.welcomeState.hidden = false;
```
Does this solve an existing problem? If I tab to an element inside a chat message and press the "Clear Chat" shortcut? No wait, "Clear Chat" is clicked from the side menu, so focus would be on the "Clear Chat" button itself!
Wait, but if it is triggered by keyboard shortcut... is there a keyboard shortcut?
No, the only shortcut is Escape.
But maybe if focus is inside `chatMessages`, maybe there is no keyboard shortcut. Let's look closely at `resetConversation()`.

Wait, in `.Jules/palette.md`:
"## 2026-10-06 - Preventing Focus Loss on Hidden/Removed Elements
**Learning:** To prevent accessibility focus loss, dynamically hiding or removing elements (like a recording bar or an error card) must explicitly shift keyboard focus to a logical adjacent element (e.g., the related text input) before hiding or removing it.
**Action:** When hiding or removing a container or element, check if it contains the document.activeElement (using .contains()) and, if so, transfer focus to an appropriate target (like the main text input) prior to hiding or removing."

We should apply this to `replaceChildren()` of `chatMessages` just in case, but let's see if there are other removed elements that don't do this.
Let's search for `.remove()` or `.hidden = true` in the codebase.
