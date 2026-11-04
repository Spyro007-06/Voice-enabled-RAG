Wait, line 283 in `frontend/js/app.js`:
```javascript
    onError: (err) => {
      if (elements.recordingBar) elements.recordingBar.hidden = true;
```
If `recordingBar` has focus when the error occurs, focus will be lost. Let's fix that. Wait, the recording bar might not have focus if the error occurs immediately (e.g., no mic detected). But what if the user focused the cancel or stop button, and then it errors out (e.g. mic unplugged)? It's a possible edge case.

Let's check `frontend/js/app.js` at line 248:
```javascript
  if (isRecording || isBusy) {
    if (elements.welcomeState) elements.welcomeState.hidden = true;
  }
```
If focus is inside `welcomeState` (e.g. language chip or suggestion button) when the user clicks it or hits Enter, `welcomeState` is hidden. Focus is lost!
Let's see if this is true.
If the user focuses a suggestion button in `welcomeState` and presses Enter, `ChatManager.handleTextSubmission(question)` is called, state changes to `RETRIEVING`, `applyAppState` is called, and `welcomeState.hidden = true`. Focus is lost!

Let's test this in JS.
