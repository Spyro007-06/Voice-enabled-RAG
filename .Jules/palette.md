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

## 2026-10-05 - Maintaining Accessibility Focus in Drawers/Overlays
**Learning:** When implementing or modifying side drawers or non-modal overlays, it is critical to maintain accessibility focus management. Otherwise, screen reader and keyboard users lose their place on the page when the overlay closes.
**Action:** Always track `document.activeElement` before opening a drawer, programmatically shift focus to the drawer's first focusable element upon opening, and explicitly restore focus to the originally tracked element when the drawer closes.

## 2026-10-06 - Preventing Focus Loss on Hidden/Removed Elements
**Learning:** To prevent accessibility focus loss, dynamically hiding or removing elements (like a recording bar or an error card) must explicitly shift keyboard focus to a logical adjacent element (e.g., the related text input) before hiding or removing it.
**Action:** When hiding or removing a container or element, check if it contains the `document.activeElement` (using `.contains()`) and, if so, transfer focus to an appropriate target (like the main text input) prior to hiding or removing.

## 2026-10-07 - Resetting Composer State when Clearing Conversation
**Learning:** When clearing a conversation, the composer input must be fully reset to its default state. This includes clearing the text value, resetting any dynamically applied inline styles (like auto-grow height), updating the character counter, and disabling the send button.
**Action:** In `resetConversation` or similar functions, ensure the query input value is cleared, its height style is set to 'auto', the character counter is reset (e.g. to '0 / 2000'), and the send button is disabled.

## 2026-10-08 - Reliable Focus Detection in Parent Elements
**Learning:** When explicitly shifting keyboard focus away from a parent element that is about to be hidden or removed, use `parentElement.contains(document.activeElement)` rather than strict equality (`===`) to reliably detect if focus is anywhere inside the element (including child buttons or inputs).
**Action:** Replace `document.activeElement === element` with `element.contains(document.activeElement)` when checking if focus is within an element block before removing or hiding it.
