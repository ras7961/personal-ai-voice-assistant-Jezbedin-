from pathlib import Path

import pytest
from livekit.agents.llm import ToolError

from browser_tools import (
    BrowserController,
    _format_inspect_nodes,
    _parse_target,
    _truncate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "test_page.html"
FIXTURE2 = Path(__file__).parent / "fixtures" / "test_page2.html"


class _StubCtx:
    def disallow_interruptions(self) -> None:
        pass


def test_parse_target_forms() -> None:
    assert _parse_target("role=button&name=Sign in") == {
        "role": "button",
        "name": "Sign in",
    }
    assert _parse_target("placeholder=Search") == {"placeholder": "Search"}
    assert _parse_target("text=Read more") == {"text": "Read more"}
    assert _parse_target("css=#submit") == {"css": "#submit"}
    assert _parse_target("#submit") == {"css": "#submit"}
    assert _parse_target("input[type=text]") == {"css": "input[type=text]"}


def test_parse_target_malformed_falls_back_to_css() -> None:
    assert _parse_target("ok=yes=no") == {"css": "ok=yes=no"}


def test_format_inspect_nodes() -> None:
    nodes = [
        {"role": "link", "name": "About", "value": ""},
        {"role": "textbox", "name": "Search here", "value": ""},
        {"role": "checkbox", "name": "Agree", "value": "off"},
    ]
    lines = _format_inspect_nodes(nodes)
    assert lines[0] == "1. role=link&name=About"
    assert lines[1] == "2. role=textbox&name=Search here"
    assert "Agree" in lines[2]
    assert "off" in lines[2]


def test_format_inspect_nodes_caps_items() -> None:
    nodes = [{"role": "link", "name": f"L{i}", "value": ""} for i in range(60)]
    assert len(_format_inspect_nodes(nodes)) == 40


def test_truncate() -> None:
    assert _truncate("abc", 5) == "abc"
    out = _truncate("abcdef", 4)
    assert out.startswith("abcd")
    assert "truncated" in out


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _chromium_available(),
    reason="playwright chromium not installed; run `uv run playwright install chromium`",
)


@pytest.fixture()
async def controller() -> BrowserController:
    controller = BrowserController()
    yield controller
    await controller.aclose()


async def test_open_url_and_read_page(controller: BrowserController) -> None:
    result = await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    assert "Test Page" in result
    text = await controller.read_page(_StubCtx(), length=2000)
    assert "Page title: Test Page" in text
    assert "a static page for browser tests" in text


async def test_inspect_page_lists_targets(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    await controller.wait(_StubCtx(), 300)
    report = await controller.inspect_page(_StubCtx())
    assert "role=link&name=About" in report
    assert "role=textbox&name=Search here" in report
    assert "role=button&name=Go" in report
    assert "role=combobox&name=Pick one" in report


async def test_click_and_eval_js_round_trip(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    await controller.type_text(_StubCtx(), "role=textbox&name=Search here", "cats")
    await controller.click(_StubCtx(), "role=button&name=Go")
    assert (
        await controller.eval_js(
            _StubCtx(), "document.getElementById('out').textContent"
        )
        == '"searched:cats"'
    )


async def test_select_option_by_label(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    await controller.select_option(
        _StubCtx(), "role=combobox&name=Pick one", label="Bravo"
    )
    assert (
        await controller.eval_js(
            _StubCtx(), "document.getElementById('out').textContent"
        )
        == '"picked:b"'
    )


async def test_check_by_css(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    await controller.check(_StubCtx(), "css=#agree")
    assert (
        await controller.eval_js(_StubCtx(), "document.getElementById('agree').checked")
        == "true"
    )


async def test_navigation_history(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    await controller.open_url(_StubCtx(), FIXTURE2.as_uri())
    back = await controller.go_back(_StubCtx())
    assert "Test Page" in back
    forward = await controller.go_forward(_StubCtx())
    assert "Second Page" in forward


async def test_reload(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    result = await controller.reload(_StubCtx())
    assert "Test Page" in result


async def test_scroll_and_press_key(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    assert (await controller.scroll(_StubCtx(), direction="down")).startswith(
        "Scrolled down."
    )
    assert (await controller.scroll(_StubCtx(), direction="top")).startswith(
        "Scrolled top."
    )
    await controller.press_key(_StubCtx(), "Home")
    await controller.press_key(_StubCtx(), "Enter", target="role=button&name=Go")


async def test_tabs(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    opened = await controller.open_new_tab(_StubCtx(), FIXTURE2.as_uri())
    assert "Second Page" in opened
    await controller.switch_tab(_StubCtx(), 0)
    assert "Test Page" in await controller.read_page(_StubCtx(), length=200)
    await controller.close_tab(_StubCtx())
    assert "Second Page" in await controller.eval_js(_StubCtx(), "document.title")


async def test_screenshot(controller: BrowserController, tmp_path: Path) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    target = tmp_path / "shot.png"
    result = await controller.screenshot(_StubCtx(), str(target))
    assert "shot.png" in result
    assert target.exists()
    assert target.stat().st_size > 0


async def test_confirm_browser_action(controller: BrowserController) -> None:
    ok = await controller.confirm_browser_action(_StubCtx(), "submit the payment")
    assert ok.startswith("Confirmed:")
    refused = await controller.confirm_browser_action(
        _StubCtx(), "submit the payment", confirmed=False
    )
    assert "NOT confirmed" in refused


async def test_missing_element_raises_tool_error(controller: BrowserController) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    with pytest.raises(ToolError):
        await controller.click(_StubCtx(), "role=button&name=Does Not Exist")


async def test_switch_tab_out_of_range_raises_tool_error(
    controller: BrowserController,
) -> None:
    await controller.open_url(_StubCtx(), FIXTURE.as_uri())
    with pytest.raises(ToolError):
        await controller.switch_tab(_StubCtx(), 5)


async def test_search_the_web_builds_duckduckgo_url(
    controller: BrowserController, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakePage:
        def __init__(self) -> None:
            self.url = "about:blank"
            self.visited: list[str] = []

        async def goto(self, url: str, **kwargs: object) -> None:
            self.visited.append(url)
            self.url = url

        async def title(self) -> str:
            return "DuckDuckGo"

    fake = _FakePage()

    async def fake_ensure() -> _FakePage:
        return fake

    monkeypatch.setattr(controller, "_ensure_locked", fake_ensure)
    result = await controller.search_the_web(_StubCtx(), "current weather in Paris")
    assert fake.url.startswith("https://duckduckgo.com/?q=")
    assert "current%20weather%20in%20Paris" in fake.url
    assert "search results" in result.lower()


# --- consolidated tool surface -------------------------------------------------
#
# browse collapses open-then-inspect-then-read into one call, so these assert
# the fold works and, more importantly, that the submission gate actually
# blocks. The old confirm_browser_action only returned a string, so the
# "ask before submitting" rule was a suggestion with nothing behind it.


async def test_browse_reads_text_and_targets_in_one_call(
    controller: BrowserController,
) -> None:
    result = await controller.browse(_StubCtx(), FIXTURE.as_uri())
    assert "Page title: Test Page" in result
    assert "a static page for browser tests" in result
    assert "role=link&name=About" in result
    assert "role=button&name=Go" in result
    assert "role=combobox&name=Pick one" in result
    assert "Interactive elements" in result


async def test_browse_without_url_rereads_current_page(
    controller: BrowserController,
) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    again = await controller.browse(_StubCtx())
    assert "Page title: Test Page" in again
    assert "role=button&name=Go" in again


async def test_browse_interact_covers_fields(controller: BrowserController) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    await controller.browse_interact(
        _StubCtx(), "role=textbox&name=Search here", "type", value="cats"
    )
    await controller.browse_click(_StubCtx(), "role=button&name=Go")
    assert (
        await controller.eval_js(
            _StubCtx(), "document.getElementById('out').textContent"
        )
        == '"searched:cats"'
    )

    await controller.browse_interact(
        _StubCtx(), "role=combobox&name=Pick one", "select", value="Bravo"
    )
    assert (
        await controller.eval_js(
            _StubCtx(), "document.getElementById('out').textContent"
        )
        == '"picked:b"'
    )

    await controller.browse_interact(_StubCtx(), "css=#agree", "check")
    assert (
        await controller.eval_js(_StubCtx(), "document.getElementById('agree').checked")
        == "true"
    )


async def test_browse_interact_rejects_unknown_action(
    controller: BrowserController,
) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    with pytest.raises(ToolError):
        await controller.browse_interact(_StubCtx(), "css=#agree", "poke")


async def test_browse_interact_type_requires_a_value(
    controller: BrowserController,
) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    with pytest.raises(ToolError):
        await controller.browse_interact(
            _StubCtx(), "role=textbox&name=Search here", "type"
        )


async def test_browse_nav_actions(controller: BrowserController) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    await controller.browse(_StubCtx(), FIXTURE2.as_uri())
    assert "Test Page" in await controller.browse_nav(_StubCtx(), "back")
    assert "Second Page" in await controller.browse_nav(_StubCtx(), "forward")
    assert "Second Page" in await controller.browse_nav(_StubCtx(), "reload")
    with pytest.raises(ToolError):
        await controller.browse_nav(_StubCtx(), "sideways")


async def test_browse_tabs_actions(controller: BrowserController) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    assert await controller.browse_tabs(_StubCtx(), "list") == "One tab open."
    await controller.browse_tabs(_StubCtx(), "open", url=FIXTURE2.as_uri())
    assert "Second Page" in await controller.browse_tabs(_StubCtx(), "list")
    assert "Test Page" in await controller.browse_tabs(_StubCtx(), "switch", index=0)
    await controller.browse_tabs(_StubCtx(), "close")
    assert await controller.browse_tabs(_StubCtx(), "list") == "One tab open."
    with pytest.raises(ToolError):
        await controller.browse_tabs(_StubCtx(), "teleport")


async def test_submitting_a_form_is_blocked_without_approval(
    controller: BrowserController,
) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    with pytest.raises(ToolError, match="not been approved"):
        await controller.browse_interact(
            _StubCtx(),
            "role=textbox&name=Search here",
            "type",
            value="cats",
            submit=True,
        )


async def test_approval_allows_exactly_one_submit(
    controller: BrowserController,
) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    await controller.confirm_browser_action(_StubCtx(), "submit the search")
    # The approval is what is under test here; this fixture has no form, so
    # Enter goes nowhere. The typed value landing is the observable sign the
    # interaction was actually permitted and carried out.
    await controller.browse_interact(
        _StubCtx(),
        "role=textbox&name=Search here",
        "type",
        value="cats",
        submit=True,
    )
    assert (
        await controller.eval_js(_StubCtx(), "document.getElementById('search').value")
        == '"cats"'
    )
    # The approval is consumed, so a second submit needs asking again.
    with pytest.raises(ToolError, match="not been approved"):
        await controller.browse_interact(
            _StubCtx(),
            "role=textbox&name=Search here",
            "type",
            value="dogs",
            submit=True,
        )


async def test_refusing_confirmation_withdraws_approval(
    controller: BrowserController,
) -> None:
    await controller.confirm_browser_action(_StubCtx(), "submit the payment")
    await controller.confirm_browser_action(_StubCtx(), "submit the payment", False)
    with pytest.raises(ToolError, match="not been approved"):
        await controller.browse_key(_StubCtx(), "Enter")


async def test_enter_key_requires_approval(controller: BrowserController) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    with pytest.raises(ToolError, match="not been approved"):
        await controller.browse_key(_StubCtx(), "Enter")
    await controller.confirm_browser_action(_StubCtx(), "press Enter to submit")
    assert await controller.browse_key(_StubCtx(), "Enter") == "Pressed Enter."


async def test_non_enter_keys_need_no_approval(controller: BrowserController) -> None:
    await controller.browse(_StubCtx(), FIXTURE.as_uri())
    assert await controller.browse_key(_StubCtx(), "Tab") == "Pressed Tab."


def test_consolidated_tools_replace_the_old_surface() -> None:
    names = {tool._info.name for tool in BrowserController().build_tools()}
    assert names == {
        "browse",
        "browse_click",
        "browse_interact",
        "browse_nav",
        "browse_tabs",
        "browse_key",
        "scroll",
        "screenshot",
        "confirm_browser_action",
    }
    # The redundant in-browser search path is no longer offered: search_web
    # covers lookups, and two overlapping search tools with no decision rule
    # was part of why Jzbedin mis-selected.
    assert "search_the_web" not in names
    assert "eval_js" not in names
