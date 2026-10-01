"""Agent-controlled Playwright browser tools.

Provides a :class:`BrowserController` that owns a single Chromium instance and
exposes a collection of LiveKit ``@function_tool`` tools so jzbedin can open,
read, and fully interact with web pages. All browser operations are serialized
through an ``asyncio.Lock`` so concurrent tool calls never race on the page.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any
from urllib.parse import quote

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

logger = logging.getLogger("browser.tools")

_DEFAULT_ACTION_TIMEOUT_MS = 10_000
_DEFAULT_NAVIGATION_TIMEOUT_MS = 30_000
_READ_CHUNK_CHARS = 4000
_INSPECT_MAX_ITEMS = 40
_EVAL_MAX_CHARS = 4000

# Evaluated in the page to collect visible, interactive elements. Each node
# carries a role, accessible name, and current value so the model can target it
# with webkit.get_by_role() etc. via the "role=...&name=..." convention.
_INSPECT_JS = """
(() => {
  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim().slice(0, 80);
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    return style && style.display !== "none" && style.visibility !== "hidden"
      && el.getClientRects().length > 0;
  };
  const roleFor = (el) => {
    const explicit = el.getAttribute("role");
    const known = ["button", "link", "textbox", "checkbox", "radio", "combobox",
      "searchbox", "heading", "navigation", "menuitem", "tab"];
    if (explicit && known.includes(explicit)) return explicit;
    const tag = el.tagName;
    if (tag === "A") return "link";
    if (tag === "BUTTON") return "button";
    if (tag === "SELECT") return "combobox";
    if (tag === "TEXTAREA") return "textbox";
    if (tag === "INPUT") {
      const type = (el.getAttribute("type") || "text").toLowerCase();
      if (type === "checkbox") return "checkbox";
      if (type === "radio") return "radio";
      if (["button", "submit", "reset"].includes(type)) return "button";
      if (type === "search") return "searchbox";
      return "textbox";
    }
    if (tag === "H1" || tag === "H2" || tag === "H3") return "heading";
    return null;
  };
  const nameFor = (el, role) => {
    const explicit = clean(el.getAttribute("aria-label")) || clean(el.getAttribute("title"));
    if (explicit) return explicit;
    if (el.id) {
      const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (label) return clean(label.textContent);
    }
    if (["link", "button", "heading"].includes(role)) return clean(el.textContent);
    if (el.hasAttribute("placeholder")) return clean(el.getAttribute("placeholder"));
    return "";
  };
  const out = [];
  const seen = new Set();
  const nodes = document.querySelectorAll(
    "a[href], button, input, textarea, select, [role], h1, h2, h3"
  );
  for (const el of nodes) {
    if (out.length >= 60) break;
    if (el.disabled) continue;
    if (el.getAttribute("aria-hidden") === "true") continue;
    if (!visible(el)) continue;
    const role = roleFor(el);
    if (!role) continue;
    const name = nameFor(el, role);
    if (!name) continue;
    const key = role + "|" + name;
    if (seen.has(key)) continue;
    seen.add(key);
    const value =
      ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) ? clean(el.value) : "";
    out.push({ role: role, name: name, value: value });
  }
  return out;
})()
"""


def _format_inspect_nodes(nodes: list[dict[str, Any]]) -> list[str]:
    """Turn inspected DOM nodes into numbered, round-trippable target strings."""
    lines: list[str] = []
    for index, node in enumerate(nodes[:_INSPECT_MAX_ITEMS], 1):
        target = f"role={node['role']}&name={node['name']}"
        if node.get("value"):
            target += f"  (current value: {node['value']})"
        lines.append(f"{index}. {target}")
    return lines


def _headless_default() -> bool:
    """Read JZBEDIN_BROWSER_HEADLESS; default to headed (visible browser).

    Set JZBEDIN_BROWSER_HEADLESS=1 to hide the browser window.
    """
    return os.environ.get("JZBEDIN_BROWSER_HEADLESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _parse_target(target: str) -> dict[str, str]:
    """Turn a target string into a dict of locator criteria.

    Accepted forms (case-insensitive keys):
      - ``role=<role>&name=<name>`` as emitted by inspect_page
      - ``placeholder=<text>``
      - ``text=<text>``
      - ``css=<selector>``
    Anything else is treated as a raw CSS selector.
    """
    target = target.strip()
    if "=" in target:
        parts = target.split("&")
        if parts and all("=" in part for part in parts):
            parsed = {
                key.strip().lower(): value.strip()
                for key, value in (part.split("=", 1) for part in parts)
            }
            if parsed.keys() & {"role", "placeholder", "text", "css"}:
                return parsed
    return {"css": target}


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


class BrowserController:
    """Owns one Chromium instance shared by every browser tool.

    The browser is launched lazily on the first tool call and torn down with
    :meth:`aclose`. Instantiate one controller per agent session so each room
    gets a private browser.
    """

    def __init__(
        self,
        *,
        headless: bool | None = None,
        width: int = 1280,
        height: int = 960,
    ) -> None:
        self._pw: Any | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._lock = asyncio.Lock()
        self._action_timeout_ms = _DEFAULT_ACTION_TIMEOUT_MS
        self._navigation_timeout_ms = _DEFAULT_NAVIGATION_TIMEOUT_MS
        self._headless = _headless_default() if headless is None else headless
        self._viewport = {"width": width, "height": height}
        self._screenshot_count = 0
        # Set by confirm_browser_action and consumed by the first consequential
        # action that follows, so the prompt's "ask before submitting" rule is
        # enforced rather than merely suggested.
        self._approved_action: str | None = None

    async def _ensure_locked(self) -> Page:
        """Return the active page, launching the browser on first use.

        Must only be called while holding ``self._lock``.
        """
        if self._page is not None:
            return self._page
        if self._pw is None:
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=self._headless,
                args=["--no-sandbox"],
            )
            logger.info(
                "Launched agent-controlled chromium (headless=%s)",
                self._headless,
            )
        browser = self._browser
        assert browser is not None
        context: BrowserContext = await browser.new_context(viewport=self._viewport)
        page = await context.new_page()
        page.set_default_timeout(self._action_timeout_ms)
        page.set_default_navigation_timeout(self._navigation_timeout_ms)
        self._page = page
        return page

    def _target_locators(self, page: Page, parsed: dict[str, str]) -> list[Any]:
        locators: list[Any] = []
        if parsed.get("role"):
            role = parsed["role"]
            name = parsed.get("name")
            if name:
                locators.append(page.get_by_role(role, name=name))
                # Fallbacks: accessible names can be computed from the
                # placeholder or visible text, which get_by_role may miss.
                locators.append(page.get_by_placeholder(name))
                locators.append(page.get_by_text(name, exact=False))
            else:
                locators.append(page.get_by_role(role))
        if parsed.get("placeholder"):
            locators.append(page.get_by_placeholder(parsed["placeholder"]))
        if parsed.get("text"):
            locators.append(page.get_by_text(parsed["text"], exact=False))
        if parsed.get("css"):
            locators.append(page.locator(parsed["css"]))
        if not locators:
            locators.append(page.locator(parsed.get("css", "")))
        return locators

    async def _resolve_target(self, page: Page, target: str) -> Any:
        parsed = _parse_target(target)
        for locator in self._target_locators(page, parsed):
            try:
                if await locator.count() > 0:
                    return locator.first
            except PlaywrightTimeoutError:
                continue
        raise ToolError(
            f"No element found matching '{target}'. Run inspect_page to see "
            "what elements are on the page."
        )

    async def open_url(self, context: RunContext, url: str) -> str:
        """Open a URL in the agent-controlled browser, replacing the current page.

        If the URL is already the current page, it is reloaded instead. Returns
        the title and URL of the loaded page.

        Args:
            url: The full URL to open, including the scheme, for example https://www.example.com
        """
        async with self._lock:
            page = await self._ensure_locked()
            try:
                if page.url and page.url.rstrip("/") == url.rstrip("/"):
                    await page.reload(wait_until="domcontentloaded")
                else:
                    await page.goto(url, wait_until="domcontentloaded")
                title = await page.title()
                return f"Opened page\nTitle: {title}\nURL: {page.url}"
            except PlaywrightTimeoutError as exc:
                raise ToolError(
                    f"Timed out loading {url}. The site may be slow or unreachable."
                ) from exc
            except Exception as exc:
                raise ToolError(f"Failed to open {url}: {exc}") from exc

    async def go_back(self, context: RunContext) -> str:
        """Go back one step in the browser history."""
        async with self._lock:
            page = await self._ensure_locked()
            try:
                previous = await page.go_back(wait_until="domcontentloaded")
            except PlaywrightTimeoutError as exc:
                raise ToolError("Timed out navigating back.") from exc
            except Exception as exc:
                raise ToolError(f"Failed to go back: {exc}") from exc
            if previous is None:
                return "No previous page in history."
            return f"Went back\nTitle: {await page.title()}\nURL: {page.url}"

    async def go_forward(self, context: RunContext) -> str:
        """Go forward one step in the browser history."""
        async with self._lock:
            page = await self._ensure_locked()
            try:
                nxt = await page.go_forward(wait_until="domcontentloaded")
            except PlaywrightTimeoutError as exc:
                raise ToolError("Timed out navigating forward.") from exc
            except Exception as exc:
                raise ToolError(f"Failed to go forward: {exc}") from exc
            if nxt is None:
                return "No next page in history."
            return f"Went forward\nTitle: {await page.title()}\nURL: {page.url}"

    async def reload(self, context: RunContext) -> str:
        """Reload the current page."""
        async with self._lock:
            page = await self._ensure_locked()
            try:
                await page.reload(wait_until="domcontentloaded")
            except PlaywrightTimeoutError as exc:
                raise ToolError("Timed out reloading the page.") from exc
            except Exception as exc:
                raise ToolError(f"Failed to reload the page: {exc}") from exc
            return f"Reloaded page\nTitle: {await page.title()}\nURL: {page.url}"

    async def open_new_tab(self, context: RunContext, url: str) -> str:
        """Open a new browser tab at the given URL. The new tab becomes the active page."""
        async with self._lock:
            page = await self._ensure_locked()
            try:
                new_page = await page.context.new_page()
                await new_page.goto(url, wait_until="domcontentloaded")
                self._page = new_page
                index = page.context.pages.index(new_page)
                title = await new_page.title()
                return f"Opened new tab {index}\nTitle: {title}\nURL: {new_page.url}"
            except PlaywrightTimeoutError as exc:
                raise ToolError(f"Timed out loading {url} in the new tab.") from exc
            except Exception as exc:
                raise ToolError(f"Failed to open new tab: {exc}") from exc

    async def switch_tab(self, context: RunContext, index: int) -> str:
        """Switch to another open browser tab by its index, which starts at 0."""
        async with self._lock:
            page = await self._ensure_locked()
            pages = page.context.pages
            if not 0 <= index < len(pages):
                raise ToolError(
                    f"Tab index {index} is out of range; {len(pages)} tabs are open."
                )
            self._page = pages[index]
            await self._page.bring_to_front()
            return f"Switched to tab {index}\nTitle: {await self._page.title()}\nURL: {self._page.url}"

    async def close_tab(self, context: RunContext, index: int | None = None) -> str:
        """Close a tab by index, or the active tab when no index is given."""
        async with self._lock:
            page = await self._ensure_locked()
            pages = page.context.pages
            if len(pages) <= 1:
                raise ToolError("Cannot close the last open tab.")
            target = pages[index] if index is not None else page
            await target.close()
            self._page = page.context.pages[-1]
            await self._page.bring_to_front()
            return f"Closed a tab. Active tab now: {await self._page.title()}."

    async def read_page(
        self, context: RunContext, start: int = 0, length: int = _READ_CHUNK_CHARS
    ) -> str:
        """Read readable text from the current page.

        Long pages are chunked: pass a larger ``start`` to continue reading from
        where the previous chunk ended.

        Args:
            start: Character offset into the page text to begin reading from.
            length: Maximum number of characters to return.
        """
        async with self._lock:
            page = await self._ensure_locked()
            try:
                body = await page.locator("body").inner_text(
                    timeout=self._action_timeout_ms
                )
            except Exception as exc:
                raise ToolError(f"Could not read the page: {exc}") from exc
            body = re.sub(r"\n{3,}", "\n\n", body or "")
            text = f"Page title: {await page.title()}\nURL: {page.url}\n\n{body}"
            total = len(text)
            chunk = text[start : start + length]
            if not chunk.strip():
                return "(No readable text on this page.)"
            if start + length < total:
                chunk += f"\n--- text continues; call read_page with start={start + length} to continue ---"
            return chunk

    async def inspect_page(self, context: RunContext) -> str:
        """List the interactive elements and landmarks on the current page.

        Every numbered line is a target string you can pass to click,
        type_text, press_key, select_option, check, or hover.
        """
        async with self._lock:
            page = await self._ensure_locked()
            try:
                nodes = await page.evaluate(_INSPECT_JS)
            except Exception as exc:
                raise ToolError(f"Could not inspect the page: {exc}") from exc
            header = f"Page title: {await page.title()}\nURL: {page.url}"
            lines = _format_inspect_nodes(nodes or [])
            if not lines:
                return header + "\nNo interactive elements detected."
            return header + "\n" + "\n".join(lines)

    async def click(self, context: RunContext, target: str) -> str:
        """Click an element on the current page.

        Args:
            target: The element to click, exactly as returned by inspect_page, for
                example 'role=button&name=Sign in', or a CSS selector prefixed with 'css='.
        """
        async with self._lock:
            page = await self._ensure_locked()
            locator = await self._resolve_target(page, target)
            try:
                await locator.click(timeout=self._action_timeout_ms)
            except PlaywrightTimeoutError as exc:
                raise ToolError(
                    f"Could not click '{target}'; the element may be hidden or unavailable."
                ) from exc
            except Exception as exc:
                raise ToolError(f"Click failed: {exc}") from exc
            return f"Clicked {target}."

    async def type_text(
        self, context: RunContext, target: str, text: str, submit: bool = False
    ) -> str:
        """Type text into a field on the current page.

        Args:
            target: The field to fill, from inspect_page, for example
                'role=textbox&name=Search'.
            text: The text to enter into the field.
            submit: If true, press Enter after typing (run a search or submit a form).
        """
        async with self._lock:
            page = await self._ensure_locked()
            locator = await self._resolve_target(page, target)
            if submit:
                context.disallow_interruptions()
            try:
                await locator.fill(text, timeout=self._action_timeout_ms)
                if submit:
                    await locator.press("Enter", timeout=self._action_timeout_ms)
            except PlaywrightTimeoutError as exc:
                raise ToolError(f"Could not type into '{target}'.") from exc
            except Exception as exc:
                raise ToolError(f"Typing failed: {exc}") from exc
            return f"Typed text into {target}." + (" Pressed Enter." if submit else "")

    async def press_key(
        self, context: RunContext, key: str, target: str | None = None
    ) -> str:
        """Press a keyboard key. Useful for submitting forms or hotkeys.

        Args:
            key: The key to press, such as Enter, Tab, Escape, ArrowDown, or a
                combination like Control+l.
            target: Optional element to focus first, from inspect_page.
        """
        async with self._lock:
            page = await self._ensure_locked()
            if target:
                locator = await self._resolve_target(page, target)
                try:
                    await locator.press(key, timeout=self._action_timeout_ms)
                except PlaywrightTimeoutError as exc:
                    raise ToolError(f"Could not press {key} on '{target}'.") from exc
            else:
                await page.keyboard.press(key)
            return f"Pressed {key}."

    async def select_option(
        self,
        context: RunContext,
        target: str,
        value: str | None = None,
        label: str | None = None,
    ) -> str:
        """Select an option in a dropdown (role=combobox or role=listbox).

        Args:
            target: The dropdown element, from inspect_page.
            value: The option's value attribute to select.
            label: The visible option text to select, instead of its value.
        """
        async with self._lock:
            page = await self._ensure_locked()
            locator = await self._resolve_target(page, target)
            try:
                if label:
                    await locator.select_option(
                        label=label, timeout=self._action_timeout_ms
                    )
                else:
                    await locator.select_option(
                        value=value, timeout=self._action_timeout_ms
                    )
            except PlaywrightTimeoutError as exc:
                raise ToolError(f"Could not select an option on '{target}'.") from exc
            except Exception as exc:
                raise ToolError(f"Selection failed: {exc}") from exc
            return f"Selected an option in {target}."

    async def check(
        self, context: RunContext, target: str, checked: bool = True
    ) -> str:
        """Check or uncheck a checkbox or radio button.

        Args:
            target: The checkbox or radio element, from inspect_page.
            checked: True to check the element, False to uncheck it.
        """
        async with self._lock:
            page = await self._ensure_locked()
            locator = await self._resolve_target(page, target)
            action = "Check" if checked else "Uncheck"
            try:
                await locator.set_checked(checked, timeout=self._action_timeout_ms)
            except PlaywrightTimeoutError as exc:
                raise ToolError(f"Could not {action.lower()} '{target}'.") from exc
            except Exception as exc:
                raise ToolError(f"{action} failed: {exc}") from exc
            return f"{action}ed {target}."

    async def hover(self, context: RunContext, target: str) -> str:
        """Move the mouse over an element, for example to reveal a menu.

        Args:
            target: The element to hover over, from inspect_page.
        """
        async with self._lock:
            page = await self._ensure_locked()
            locator = await self._resolve_target(page, target)
            try:
                await locator.hover(timeout=self._action_timeout_ms)
            except PlaywrightTimeoutError as exc:
                raise ToolError(f"Could not hover over '{target}'.") from exc
            except Exception as exc:
                raise ToolError(f"Hover failed: {exc}") from exc
            return f"Hovered over {target}."

    async def scroll(
        self, context: RunContext, direction: str = "down", amount: int = 800
    ) -> str:
        """Scroll the current page.

        Args:
            direction: one of up, down, top, or bottom.
            amount: How far to scroll, in pixels, when direction is up or down.
        """
        async with self._lock:
            page = await self._ensure_locked()
            if direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                delta = amount if direction == "down" else -amount
                await page.mouse.wheel(0, delta)
            return f"Scrolled {direction}."

    async def wait(self, context: RunContext, ms: int = 1000) -> str:
        """Wait a short time for a page to finish loading or updating.

        Args:
            ms: How long to wait, in milliseconds.
        """
        async with self._lock:
            page = await self._ensure_locked()
            await asyncio.sleep(ms / 1000)
            return f"Waited {ms} ms on {page.url}."

    async def screenshot(self, context: RunContext, path: str | None = None) -> str:
        """Capture a screenshot of the current page and save it as a PNG file.

        Args:
            path: Optional full path to save the PNG to. Defaults to a numbered
                file inside the JZBEDIN_BROWSER_SCREENSHOT_DIR directory.
        """
        async with self._lock:
            page = await self._ensure_locked()
            if path is None:
                default_dir = os.environ.get(
                    "JZBEDIN_BROWSER_SCREENSHOT_DIR", "screenshots"
                )
                os.makedirs(default_dir, exist_ok=True)
                self._screenshot_count += 1
                path = os.path.join(
                    default_dir, f"jzbedin_{self._screenshot_count}.png"
                )
            elif os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                await page.screenshot(path=path)
            except Exception as exc:
                raise ToolError(f"Screenshot failed: {exc}") from exc
            return f"Screenshot saved to {os.path.abspath(path)}."

    async def eval_js(self, context: RunContext, expression: str) -> str:
        """Run JavaScript in the current page and return its result.

        Use for advanced control that the other tools do not cover, such as
        reading computed styles or values in custom widgets. The expression must
        be a complete statement; return a value with 'return' if it is inside a
        function body.

        Args:
            expression: The JavaScript expression to evaluate.
        """
        async with self._lock:
            page = await self._ensure_locked()
            try:
                result = await page.evaluate(expression)
            except Exception as exc:
                raise ToolError(f"JavaScript evaluation failed: {exc}") from exc
            try:
                text = json.dumps(result, ensure_ascii=False, default=str)
            except TypeError:
                text = str(result)
            return _truncate(text, _EVAL_MAX_CHARS)

    async def search_the_web(self, context: RunContext, query: str) -> str:
        """Search the web by opening DuckDuckGo results in the browser.

        Call this when no specific website has been named and a general internet
        lookup is needed. After calling, use inspect_page or read_page to read
        the results before answering, and open an interesting result with open_url.

        For weather requests, include the requested location and the words
        "current weather" in the query.

        Args:
            query: The search query, for example "current weather in Paris".
        """
        url = "https://duckduckgo.com/?q=" + quote(query)
        opened = await self.open_url(context, url)
        return (
            opened
            + "\nThis is a search results page. Use inspect_page or read_page to read the results."
        )

    async def confirm_browser_action(
        self, context: RunContext, action: str, confirmed: bool = True
    ) -> str:
        """Record the user's explicit approval for one consequential action.

        Call this only after the user has agreed in words to the exact action
        you named. Submitting a form or pressing Enter to submit is then
        allowed once; anything beyond that needs asking again. Consequential
        actions include sending, submitting, purchasing, deleting, posting, or
        confirming something on a website.

        Args:
            action: The exact consequential action the user approved, for
                example "submit the payment form".
            confirmed: Whether the user agreed. Pass false to withdraw approval.
        """
        if not confirmed:
            self._approved_action = None
            return (
                f"Action NOT confirmed: {action}. Ask the user again before proceeding."
            )
        self._approved_action = action
        logger.info("User confirmed browser action: %s", action)
        return f"Confirmed: {action}. Proceed with this exact action, once."

    async def aclose(self) -> None:
        """Close the browser and release resources."""
        async with self._lock:
            page, self._page = self._page, None
            browser, self._browser = self._browser, None
            pw, self._pw = self._pw, None
            if page is not None:
                try:
                    await page.context.close()
                except Exception:
                    logger.warning("Error closing browser context", exc_info=True)
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    logger.warning("Error closing browser", exc_info=True)
            if pw is not None:
                try:
                    await pw.stop()
                except Exception:
                    logger.warning("Error stopping playwright", exc_info=True)

    # -- consolidated tool surface -------------------------------------------------
    #
    # The methods below are the ones the model actually sees. The granular
    # primitives above stay public because they are the tested units and because
    # the consolidated tools are thin wrappers over them. The split matters:
    # the confirmation gate lives in the wrappers, not the primitives, so the
    # policy applies to tool calls and not to internal browser use.

    async def _consume_approval(self, action: str) -> None:
        """Clear a recorded approval, or refuse the action."""
        approved, self._approved_action = self._approved_action, None
        if not approved:
            raise ToolError(
                f"Blocked: {action} is consequential and has not been approved. "
                "Tell the user what you are about to do, wait for them to agree, "
                "then call confirm_browser_action with that exact action."
            )

    async def browse(
        self,
        context: RunContext,
        url: str | None = None,
        start: int = 0,
        length: int = _READ_CHUNK_CHARS,
    ) -> str:
        """Open a page and read it, in one call. This is the main way to look at a website.

        Give the site's own address when the user named a site, for example
        https://www.example.com. Leave url empty to re-read the page that is
        already open, which is also the right call after a failed interaction.

        Returns the page title, its URL, its readable text, and a numbered list
        of the interactive elements on it. Every line of that element list is a
        target you can pass straight to browse_click, browse_interact, or
        browse_key.

        Long pages come back in chunks; pass a larger start to keep reading from
        where the last one stopped.

        Args:
            url: Address to open. Omit to re-read the current page.
            start: Character offset to begin reading from.
            length: Maximum characters of page text to return.
        """
        async with self._lock:
            page = await self._ensure_locked()
            try:
                if url is not None:
                    if page.url and page.url.rstrip("/") == url.rstrip("/"):
                        await page.reload(wait_until="domcontentloaded")
                    else:
                        await page.goto(url, wait_until="domcontentloaded")
                title = await page.title()
                body = await page.locator("body").inner_text(
                    timeout=self._action_timeout_ms
                )
            except PlaywrightTimeoutError as exc:
                target = url or page.url
                raise ToolError(
                    f"Timed out loading {target}. The site may be slow, blocked, "
                    "or unreachable."
                ) from exc
            except Exception as exc:
                raise ToolError(f"Could not open or read the page: {exc}") from exc

            body = re.sub(r"\n{3,}", "\n\n", body or "")
            text = f"Page title: {title}\nURL: {page.url}\n\n{body}"
            total = len(text)
            chunk = text[start : start + length]
            if not chunk.strip():
                chunk = "(No readable text on this page.)"
            elif start + length < total:
                chunk += (
                    f"\n--- text continues; call browse with start={start + length} "
                    "to continue ---"
                )

            try:
                nodes = await page.evaluate(_INSPECT_JS)
            except Exception:
                nodes = None
            lines = _format_inspect_nodes(nodes or [])
            elements = "\n".join(lines) if lines else "No interactive elements found."
            return (
                f"{chunk}\n\nInteractive elements (pass these as targets):\n{elements}"
            )

    async def browse_click(self, context: RunContext, target: str) -> str:
        """Click an element on the current page.

        Args:
            target: The element to click, exactly as browse listed it, for
                example 'role=button&name=Sign in'. A CSS selector prefixed with
                'css=' also works.
        """
        return await self.click(context, target)

    async def browse_interact(
        self,
        context: RunContext,
        target: str,
        action: str,
        value: str | None = None,
        submit: bool = False,
    ) -> str:
        """Fill in a field, choose from a dropdown, tick a box, or reveal a menu.

        Args:
            target: The element to act on, exactly as browse listed it, for
                example 'role=textbox&name=Search'.
            action: One of 'type', 'select', 'check', 'uncheck', or 'hover'.
            value: The text to type, or for 'select' the option's value
                attribute. Leave empty to use the option's visible label.
            submit: Only for 'type'. If true, press Enter afterwards to run the
                search or submit the form. Submitting needs the user's approval
                first: call confirm_browser_action, then repeat this call.
        """
        if action == "type":
            if value is None:
                raise ToolError("action 'type' needs a value to type.")
            if submit:
                await self._consume_approval(f"submit the form at {target}")
            return await self.type_text(context, target, value, submit=submit)
        if action == "select":
            if value is None:
                return await self.select_option(context, target, label=None)
            return await self.select_option(context, target, label=value)
        if action == "check":
            return await self.check(context, target, checked=True)
        if action == "uncheck":
            return await self.check(context, target, checked=False)
        if action == "hover":
            return await self.hover(context, target)
        raise ToolError(
            f"Unknown action '{action}'. Use 'type', 'select', 'check', "
            "'uncheck', or 'hover'."
        )

    async def browse_nav(self, context: RunContext, action: str) -> str:
        """Move through browser history or reload the page.

        Args:
            action: One of 'back', 'forward', or 'reload'.
        """
        if action == "back":
            return await self.go_back(context)
        if action == "forward":
            return await self.go_forward(context)
        if action == "reload":
            return await self.reload(context)
        raise ToolError(
            f"Unknown action '{action}'. Use 'back', 'forward', or 'reload'."
        )

    async def browse_tabs(
        self,
        context: RunContext,
        action: str,
        index: int | None = None,
        url: str | None = None,
    ) -> str:
        """List, open, switch, or close browser tabs. Tab indexes start at 0.

        Args:
            action: One of 'list', 'open', 'switch', or 'close'.
            index: Which tab, for 'switch' and 'close'.
            url: Address to open, for 'open'.
        """
        if action == "list":
            async with self._lock:
                page = await self._ensure_locked()
                pages = page.context.pages
                if len(pages) == 1:
                    return "One tab open."
                rows = [
                    f"{i}{' (active)' if p is page else ''}: {await p.title()} - {p.url}"
                    for i, p in enumerate(pages)
                ]
                return "\n".join(rows)
        if action == "open":
            if url is None:
                raise ToolError("action 'open' needs a url to open.")
            return await self.open_new_tab(context, url)
        if action == "switch":
            if index is None:
                raise ToolError("action 'switch' needs a tab index.")
            return await self.switch_tab(context, index)
        if action == "close":
            return await self.close_tab(context, index)
        raise ToolError(
            f"Unknown action '{action}'. Use 'list', 'open', 'switch', or 'close'."
        )

    async def browse_key(
        self, context: RunContext, key: str, target: str | None = None
    ) -> str:
        """Press a keyboard key, for example Enter, Tab, Escape, or Control+l.

        Args:
            key: The key to press.
            target: Optional element to focus first, as listed by browse.
        """
        if key.strip().lower() == "enter":
            await self._consume_approval("press Enter to submit")
        return await self.press_key(context, key, target)

    def build_tools(self) -> list[Any]:
        """Create the LiveKit function tools for the consolidated browser API.

        Ten tools instead of the original twenty-one. The important fold is
        browse, which returns a page's text and its interactive elements
        together, so the common open-then-inspect-then-read sequence collapses
        from three round trips to one.
        """
        names = [
            "browse",
            "browse_click",
            "browse_interact",
            "browse_nav",
            "browse_tabs",
            "browse_key",
            "scroll",
            "screenshot",
            "confirm_browser_action",
        ]
        return [function_tool(getattr(self, name), name=name) for name in names]
