## 2024-05-24 - Skip to Main Content Nav
**Learning:** The application was missing a skip-to-content link, which is essential for keyboard and screen reader accessibility to bypass top navigation.
**Action:** Adding a visually hidden `.skip-to-content` anchor that appears on `:focus` provides this functionality with no disruption to visual layout for mouse users.

## 2026-09-23 - Hide Decorative Text Icons from Screen Readers
**Learning:** Purely decorative emojis and text-based icons (like ✦, 🗑, 🔊) placed next to text labels can cause redundant or confusing announcements for screen reader users (e.g., announcing "Wastebasket, Clear Conversation").
**Action:** Always add `aria-hidden="true"` to spans or elements containing these decorative text icons to ensure a clean, focused auditory experience.
## 2024-05-18 - Global Escape Key Modal Close
**Learning:** Adding explicit Escape key handling within the active modal's keydown listener improves keyboard accessibility. Users naturally reach for 'Escape' to dismiss active overlays, so providing this behavior consistently on all generic modals provides a smoother UX.
**Action:** Extend the existing modal `keydown` trap handler to check for `e.key === 'Escape'` and close the specific `_activeModalId` so users don't have to navigate to a close button.

## 2024-10-25 - Resetting Auto-Growing Textareas
**Learning:** When using JavaScript to programmatically manage a `<textarea>`'s auto-growing height via the `oninput` event (e.g. `e.target.style.height = e.target.scrollHeight + 'px'`), programmatically clearing the value (e.g. `textarea.value = ''`) does not trigger the `oninput` event. This leaves the textarea "stuck" at its previously expanded height, which feels clunky and broken to users after submitting a long message.
**Action:** Whenever programmatically clearing the value of an auto-growing textarea, you must also manually reset its inline height style (e.g. `textarea.style.height = 'auto'`) to restore it to its default, unexpanded state.

## 2026-10-04 - Preventing Focus Loss on Dynamically Disabled Buttons
**Learning:** To prevent accessibility focus loss, dynamically disabled interactive elements (like a submit button clicked via keyboard) must explicitly shift keyboard focus to a logical adjacent element (e.g., the related text input) before being disabled.
**Action:** When a button's disabled state is managed dynamically, check if it's currently the `document.activeElement` and, if so, transfer focus to an appropriate target (like the main text input) prior to setting `disabled = true`.

## 2023-10-06 - Drawer Focus Management
**Learning:** Side drawers and non-modal overlays that change the context of the UI need robust focus management. Failing to track the previously focused element before opening a drawer or failing to explicitly shift focus onto the drawer when it opens disrupts keyboard and screen reader navigation flow.
**Action:** Always capture `document.activeElement` before opening a drawer, shift focus to the drawer's first focusable element, and explicitly restore focus to the tracked element upon the drawer's closure.
