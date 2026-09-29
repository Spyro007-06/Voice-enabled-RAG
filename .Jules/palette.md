## 2024-05-24 - Skip to Main Content Nav
**Learning:** The application was missing a skip-to-content link, which is essential for keyboard and screen reader accessibility to bypass top navigation.
**Action:** Adding a visually hidden `.skip-to-content` anchor that appears on `:focus` provides this functionality with no disruption to visual layout for mouse users.

## 2026-09-23 - Hide Decorative Text Icons from Screen Readers
**Learning:** Purely decorative emojis and text-based icons (like ✦, 🗑, 🔊) placed next to text labels can cause redundant or confusing announcements for screen reader users (e.g., announcing "Wastebasket, Clear Conversation").
**Action:** Always add `aria-hidden="true"` to spans or elements containing these decorative text icons to ensure a clean, focused auditory experience.

## 2026-09-29 - Decorative Elements Accessibility
**Learning:** Screen readers announce character symbols like `×` (multiply) out of context, and decorative emojis or SVGs provide redundant, unnecessary audio clutter when the parent element already has a descriptive `aria-label`.
**Action:** Wrap such purely decorative text/symbols in `<span aria-hidden="true">` or apply `aria-hidden="true"` directly to decorative SVG or emoji elements to ensure a focused auditory experience.
