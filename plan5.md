When clearing the chat in `frontend/js/app.js`:

```javascript
  const resetConversation = () => {
    // If we're closing drawers and our previous focus was inside chatMessages,
    // we're about to replace chatMessages, which destroys previousDrawerFocus.
    // So we should update previousDrawerFocus to be queryInput.
    if (previousDrawerFocus && elements.chatMessages && elements.chatMessages.contains(previousDrawerFocus)) {
        previousDrawerFocus = elements.queryInput;
    }

    if (elements.chatMessages) elements.chatMessages.replaceChildren();
```
This is a micro-UX improvement for keyboard accessibility. If the user was exploring a chat response, opened the menu, and selected "Clear Conversation", focus returns intelligently to the `queryInput` (the composer) after the menu closes, rather than being lost to the `<body>` (because the original focused element inside `chatMessages` was destroyed).

Wait! What about the `welcomeState` suggestion buttons?
If I tab to a suggestion button in `welcomeState` and hit Enter, it triggers:
```javascript
        btn.onclick = () => {
          if (elements.queryInput) {
            elements.queryInput.value = '';
            elements.queryInput.style.height = 'auto'; // Reset auto-grow height
          }
          ChatManager.handleTextSubmission(question);
        };
```
Then `handleTextSubmission(question)` -> `state.setState(APP_STATE.RETRIEVING)` -> `applyAppState()` -> `if (elements.welcomeState) elements.welcomeState.hidden = true;`.
Because `document.activeElement` was the `suggestion-btn` inside `welcomeState`, hiding `welcomeState` causes focus to be lost to the `<body>`.
To fix this, before calling `handleTextSubmission(question)`, we should focus `elements.queryInput` if the button has focus.
```javascript
        btn.onclick = () => {
          if (elements.queryInput) {
            elements.queryInput.value = '';
            elements.queryInput.style.height = 'auto'; // Reset auto-grow height

            // Prevent focus loss when welcomeState is hidden
            if (document.activeElement === btn) {
              elements.queryInput.focus();
            }
          }
          ChatManager.handleTextSubmission(question);
        };
```
This EXACTLY matches the pattern taught in `palette.md`:
"To prevent accessibility focus loss, dynamically hiding or removing elements (like a recording bar or an error card) must explicitly shift keyboard focus to a logical adjacent element (e.g., the related text input) before hiding or removing it."
Here, `welcomeState` is hidden.

Wait, are there other places where `welcomeState` is clicked?
In `frontend/js/app.js`, `languageTabs`:
```javascript
        btn.onclick = () => {
          state.setLanguage(code);
        };
```
This just changes the language, it doesn't hide `welcomeState` (it doesn't start a submission).

In `frontend/js/app.js`, hero mic buttons:
```javascript
  if (elements.centralVoiceBtn) elements.centralVoiceBtn.onclick = toggleRecording;
  if (elements.welcomeMicBtn) elements.welcomeMicBtn.onclick = toggleRecording;
```
If we click `centralVoiceBtn`, `toggleRecording` calls `startVoiceRecording` -> `APP_STATE.RECORDING` -> `welcomeState.hidden = true`.
If `document.activeElement === centralVoiceBtn` or `welcomeMicBtn`, `welcomeState` is hidden. But wait, `centralVoiceBtn` and `welcomeMicBtn` are inside `welcomeState`!
Let's check where they are in `index.html`.
