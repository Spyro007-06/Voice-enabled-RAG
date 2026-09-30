## 2024-05-24 - Skip to Main Content Nav
**Learning:** The application was missing a skip-to-content link, which is essential for keyboard and screen reader accessibility to bypass top navigation.
**Action:** Adding a visually hidden `.skip-to-content` anchor that appears on `:focus` provides this functionality with no disruption to visual layout for mouse users.

## 2026-09-23 - Hide Decorative Text Icons from Screen Readers
**Learning:** Purely decorative emojis and text-based icons (like ✦, 🗑, 🔊) placed next to text labels can cause redundant or confusing announcements for screen reader users (e.g., announcing "Wastebasket, Clear Conversation").
**Action:** Always add `aria-hidden="true"` to spans or elements containing these decorative text icons to ensure a clean, focused auditory experience.

## 2024-05-24 - Screen Reader Icon Redundancy
**Learning:** Screen readers announce both the aria-label and the visible text character (like '×') of an icon-only button, causing redundant or confusing announcements.
**Action:** Applied aria-hidden="true" to decorative text-based icons inside buttons that already possess an aria-label.
