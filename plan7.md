```markdown
1. **Analyze the problem**: The user wants a micro-UX enhancement that improves accessibility or usability. In this application, elements are dynamically hidden and replaced which can lead to keyboard focus loss (breaking accessibility). Specifically, when users interact with the `welcomeState` (e.g. clicking a suggestion or the hero microphone), `welcomeState` is dynamically hidden. If the active element was inside `welcomeState`, focus is lost and resets to the `<body>`. Additionally, when clearing a conversation via the menu drawer, if the previous focus (before the menu opened) was inside `chatMessages`, it gets lost when `chatMessages.replaceChildren()` is called.
2. **Modify `frontend/js/app.js` to preserve focus when `welcomeState` is hidden**: In `applyAppState`, check if `welcomeState` contains the `document.activeElement` before hiding it. If so, shift focus to `elements.queryInput`.
3. **Modify `frontend/js/app.js` to preserve focus when `chatMessages` are cleared**: In `resetConversation`, check if `previousDrawerFocus` is inside `chatMessages`. If so, reassign `previousDrawerFocus` to `elements.queryInput` before clearing the children.
4. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
5. **Submit the changes via PR.**
```
