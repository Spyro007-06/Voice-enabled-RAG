## 2024-05-24 - Skip to Main Content Nav
**Learning:** The application was missing a skip-to-content link, which is essential for keyboard and screen reader accessibility to bypass top navigation.
**Action:** Adding a visually hidden `.skip-to-content` anchor that appears on `:focus` provides this functionality with no disruption to visual layout for mouse users.

## 2026-09-23 - Hide Decorative Text Icons from Screen Readers
**Learning:** Purely decorative emojis and text-based icons (like ✦, 🗑, 🔊) placed next to text labels can cause redundant or confusing announcements for screen reader users (e.g., announcing "Wastebasket, Clear Conversation").
**Action:** Always add `aria-hidden="true"` to spans or elements containing these decorative text icons to ensure a clean, focused auditory experience.

## 2026-10-27 - Offcanvas Drawer Focus Management
**Learning:** Screen reader and keyboard navigation breaks when opening/closing offcanvas drawers (like the info drawers or navigation menu). If focus isn't moved inside the opened drawer, keyboard users get stuck interacting with the page behind it. If focus isn't restored properly upon closing, users are thrown back to the top of the page.
**Action:** When a drawer opens, store `document.activeElement` and shift focus to the first interactive element (like the `.drawer-close` button). When the drawer closes, return focus to the stored element. If that element is hidden, define a logical fallback (e.g., the main menu button).
