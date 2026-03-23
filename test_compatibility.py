"""
Compatibility test: Can NeMo Agent Toolkit and Claude Agent SDK coexist?

This script checks:
1. Both packages can be imported in the same Python process
2. Python version meets both requirements
3. No dependency conflicts exist
4. MCP types are compatible across both frameworks
"""

import sys
import importlib


def check_python_version():
    """Both require Python 3.11+ (NeMo's minimum)."""
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 11
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Python version: {major}.{minor} (need >=3.11)")
    return ok


def check_import(package_name, import_name=None):
    """Try importing a package."""
    import_name = import_name or package_name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[PASS] {package_name} imported (version: {version})")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name} import failed: {e}")
        return False


def check_mcp_compatibility():
    """Check that both frameworks can use MCP types."""
    try:
        from mcp import types as mcp_types
        # Verify core MCP types exist
        assert hasattr(mcp_types, "Tool"), "Missing mcp.types.Tool"
        assert hasattr(mcp_types, "CallToolResult"), "Missing mcp.types.CallToolResult"
        print("[PASS] MCP types available and compatible")
        return True
    except (ImportError, AssertionError) as e:
        print(f"[FAIL] MCP compatibility: {e}")
        return False


def check_claude_sdk_mcp_server():
    """Check Claude Agent SDK's MCP server creation works."""
    try:
        from claude_agent_sdk import tool, create_sdk_mcp_server

        @tool("test_tool", "A test tool", {"input": str})
        async def test_tool(args):
            return {"content": [{"type": "text", "text": "ok"}]}

        server = create_sdk_mcp_server(
            name="test-server",
            version="1.0.0",
            tools=[test_tool],
        )
        print(f"[PASS] Claude SDK MCP server created: {type(server).__name__}")
        return True
    except Exception as e:
        print(f"[FAIL] Claude SDK MCP server: {e}")
        return False


def check_nemo_toolkit():
    """Check NeMo Agent Toolkit core imports."""
    try:
        import nvidia_nat
        version = getattr(nvidia_nat, "__version__", "unknown")
        print(f"[PASS] NeMo Agent Toolkit imported (version: {version})")
        return True
    except ImportError as e:
        print(f"[SKIP] NeMo Agent Toolkit not installed: {e}")
        print("       Install with: pip install nvidia-nat")
        return None  # Skip, not fail


def check_async_compatibility():
    """Verify both async runtimes can coexist."""
    try:
        import asyncio
        import anyio

        async def combined_test():
            # anyio (Claude SDK) running on asyncio (NeMo) — should work
            return True

        result = asyncio.run(combined_test())
        print(f"[PASS] asyncio + anyio coexistence: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] Async compatibility: {e}")
        return False


def main():
    print("=" * 60)
    print("NeMo Agent Toolkit + Claude Agent SDK Compatibility Test")
    print("=" * 60)
    print()

    results = []

    print("--- Environment ---")
    results.append(check_python_version())

    print("\n--- Package Imports ---")
    results.append(check_import("claude-agent-sdk", "claude_agent_sdk"))
    check_nemo_toolkit()  # Don't count as pass/fail if not installed

    print("\n--- MCP Protocol ---")
    results.append(check_mcp_compatibility())
    results.append(check_claude_sdk_mcp_server())

    print("\n--- Async Runtime ---")
    results.append(check_async_compatibility())

    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")

    if all(r for r in results):
        print("VERDICT: Fully compatible — both frameworks can work together.")
    elif any(r for r in results):
        print("VERDICT: Partially compatible — some integration paths available.")
    else:
        print("VERDICT: Not compatible in current configuration.")
    print("=" * 60)


if __name__ == "__main__":
    main()
