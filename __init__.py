from datetime import datetime
from dataclasses import dataclass, field
import traceback
import sys
import time
import pdb
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import io
import threading
import itertools


# ANSI color codes
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
ITALIC = "\033[3m"
BOLD = "\033[1m"


# TODO: Inplement debugger:
def debug_function(func, *args, **kwargs):
    """Run a function under pdb, inside its scope."""
    def tracer(frame, event, arg):
        # Only break inside the target function
        if frame.f_code is func.__code__:
            sys.settrace(None)  # disable tracing so pdb takes over
            pdb.set_trace()
        return tracer
    sys.settrace(tracer)
    try:
        return func(*args, **kwargs)
    finally:
        sys.settrace(None)



def get_root_cause_location(exc: BaseException) -> Optional[dict]:
    """
    Extracts the deepest frame where the exception originated.
    Returns a dictionary with exception info, or None if unavailable.
    """
    _, _, tb = sys.exc_info()
    if tb is None:
        return {"exception": exc, "filename": "<unknown>", "line_no": 0, "function": "<unknown>", "code": ""}

    tb_list = traceback.extract_tb(tb)
    last_frame = tb_list[-1] if tb_list else None
    if last_frame:
        return {
            "exception": exc,
            "filename": last_frame.filename,
            "line_no": last_frame.lineno,
            "function": last_frame.name,
            "code": last_frame.line,
        }
    return None


@dataclass
class TestResult:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    expected: Any
    result: Any
    error: Optional[Dict[str, Any]]
    duration: float = 0.0
    stdout: List[Tuple[float, str]] = field(default_factory=list)
    time: datetime = datetime.now()

@dataclass
class TestCase:
    expected: Any
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    func: Optional[Callable] = None

@dataclass
class UnitTestReport:
    passed: List[Tuple[Union[TestCase, TestResult]]] = field(default_factory=list)
    failed: List[Tuple[Union[TestCase, TestResult]]] = field(default_factory=list)


class TimedStdout(io.StringIO):
    """Custom stdout buffer that records timestamps of prints."""

    def __init__(self, start_time: float):
        super().__init__()
        self.start_time = start_time
        self.records: List[Tuple[float, str]] = []

    def write(self, s: str):
        if s.strip():  # ignore empty writes like '\n'
            elapsed = time.perf_counter() - self.start_time
            self.records.append((elapsed, s.rstrip()))
        return super().write(s)

class TestBlock:
    def __init__(self, func: Optional[Callable] = None,
                 testcases: Optional[List[Union[Tuple, Callable, TestCase]]] = None):
        self.func = func
        self.testcases: List[TestCase] = []

        if testcases:
            for case in testcases:
                if isinstance(case, TestCase):
                    # Already a TestCase object
                    if case.func is None:
                        case.func = func
                    self.testcases.append(case)
                elif isinstance(case, tuple):
                    if len(case) == 2 and callable(case[0]):
                        # (lambda_func, expected)
                        self.testcases.append(TestCase(expected=case[1], func=case[0]))
                    elif len(case) == 3:
                        # (args, kwargs, expected)
                        if func is None:
                            raise ValueError("func must be provided for (args, kwargs, expected) testcases")
                        self.testcases.append(TestCase(expected=case[2], func=func, args=case[0], kwargs=case[1]))
                    else:
                        raise ValueError(f"Invalid testcase tuple: {case}")
                else:
                    raise ValueError(f"Invalid testcase type: {case}")

    def test(self) -> Tuple[List[TestResult], List[TestResult]]:
        passed, failed = [], []

        for case in self.testcases:
            f = case.func or self.func
            args = case.args
            kwargs = case.kwargs
            expected = case.expected

            start = time.perf_counter()
            stdout_buffer = TimedStdout(start)
            sys_stdout_backup = sys.stdout
            sys.stdout = stdout_buffer  # redirect prints

            try:
                result = f(*args, **kwargs)
                duration = time.perf_counter() - start
                output = stdout_buffer.records

                if result == expected:
                    passed.append(TestResult(args, kwargs, expected, result, None, duration, output))
                else:
                    failed.append(TestResult(args, kwargs, expected, result, None, duration, output))
            except Exception as e:
                duration = time.perf_counter() - start
                output = stdout_buffer.records
                failed.append(TestResult(args, kwargs, expected, None, get_root_cause_location(e), duration, output))
            finally:
                sys.stdout = sys_stdout_backup  # restore stdout

        return passed, failed


class UnitTester:
    """
    Simple unit testing framework with debugging + timing features.
    """

    def __init__(self, show_traceback: bool = False,
                 slow_threshold: float = 0.5, enable_spinner: bool = True, print_result: bool = True) -> None:
        self.blocks: List[TestBlock] = []
        self.show_traceback = show_traceback
        self.debug_on_fail = False # TODO: Implement debugger
        self.slow_threshold = slow_threshold
        self.enable_spinner = enable_spinner
        self.print_result = print_result

    def add_block(self, func: Optional[Callable] = None, testcases: Optional[Union[List[Union[Tuple, Callable]], TestBlock, List[TestBlock], List[TestCase]]] = None) -> None:

        if not testcases:
            return  # nothing to add

        # If the first element is a TestBlock, treat the whole list as blocks
        if isinstance(testcases, TestBlock):
            self.blocks.append(testcases)  # add all TestBlock objects
        if isinstance(testcases[0], TestBlock):
            self.blocks.extend(testcases)
        elif isinstance(testcases[0], TestCase):
            if func is None:
                raise ValueError("func must be provided when passing raw testcases")
            self.blocks.append(TestBlock(func, testcases))
        else:
            if func is None:
                raise ValueError("func must be provided when passing raw testcases")
            self.blocks.append(TestBlock(func, testcases))

    def get_unit_test_report(self):
        """
        Run all test blocks and return a structured UnitTestReport
        without printing anything.
        """
        report = UnitTestReport()

        for block in self.blocks:
            # Run tests
            passed, failed = block.test()

            # Map TestResults back to their original TestCases
            for tc, result in zip(block.testcases, passed):
                report.passed.append((tc, result))
            for tc, result in zip(block.testcases[len(passed):], failed):
                report.failed.append((tc, result))

        return report

    def run(self) -> UnitTestReport:
        if not self.print_result:
            return self.get_unit_test_report()

        total_passed, total_failed = 0, 0
        all_failures: List[Tuple[str, TestResult]] = []
        all_results: List[Tuple[str, TestResult]] = []

        report = UnitTestReport()  # <-- collect results here

        global_start = time.perf_counter()

        for block in self.blocks:
            block_start = time.perf_counter()
            func_name = getattr(block.func, "__name__", "<lambda>") if block.func else "<lambda>"
            print(f"{YELLOW}Running tests for: {ITALIC}{func_name} [t={format_time(datetime.now())}]{RESET}")

            # Spinner setup
            done_flag = [False]

            def spinner():
                real_stdout = sys.__stdout__  # always write to real terminal
                for c in itertools.cycle(['\\', '|', '/', '-']):
                    if done_flag[0]:
                        break
                    real_stdout.write(f'\r{YELLOW}{c} Running tests... {ITALIC}[t+{timestampt(time.perf_counter()-block_start)}]{RESET}')
                    real_stdout.flush()
                    time.sleep(0.1)
                real_stdout.write('\r' + ' ' * 40 + '\r')
                real_stdout.flush()

            if getattr(self, "enable_spinner", True):
                spinner_thread = threading.Thread(target=spinner)
                spinner_thread.start()

            # --- Run tests ---
            passed, failed = block.test()

            if getattr(self, "enable_spinner", True):
                done_flag[0] = True
                spinner_thread.join()
                sys.stdout.write(f"\r")

            results = passed + failed
            total_passed += len(passed)
            total_failed += len(failed)
            all_failures.extend((func_name, f) for f in failed)
            all_results.extend((func_name, r) for r in results)

            # Add to report
            for tc, result in zip(block.testcases, passed):
                report.passed.append((tc, result))
            for tc, result in zip(block.testcases[len(passed):], failed):
                report.failed.append((tc, result))

            avg_time = sum(r.duration for r in results) / len(results) if results else 0
            slow_flag = f" {YELLOW}⚠ slow{RESET}" if avg_time > getattr(self, "slow_threshold", 0.5) else ""
            if not failed:
                print(f"{GREEN}{BOLD}{len(passed)}/{len(results)} testcases passed! {ITALIC}[avg {timestampt(avg_time)}, "
                      f"{ITALIC}t={format_time(datetime.now())}{RESET}{GREEN}]{slow_flag}{RESET}")
            else:
                print(f"{RED}{BOLD}{len(passed)}/{len(results)} testcases passed! {ITALIC}[avg {timestampt(avg_time)}, "
                      f"t={format_time(datetime.now())}{RESET}{RED}]{slow_flag}{RESET}")

            for r in results:
                if r in passed:
                    slow_flag = f" {YELLOW}⚠ slow{RESET}" if r.duration > getattr(self, "slow_threshold", 0.5) else ""
                    print(
                        f"\t{GREEN} ↳ {func_name}({', '.join(f'{r}' for r in r.args)}{', '.join(f'{k}={v}' for k, v in r.kwargs.items())}) → {r.expected} {ITALIC}[t+{timestampt(r.duration)}]{slow_flag}{RESET}\n")
                    if r.stdout:
                        self._print_stdout(r)
                else:
                    self._print_failure(func_name, r)

            # TODO:
            # if failed and getattr(self, "debug_on_fail", False):
            #     print(f"{YELLOW}--- Entering debugger for failed test ---{RESET}")

        global_duration = time.perf_counter() - global_start
        summary_color = GREEN if total_failed == 0 else RED
        total_tests = total_passed + total_failed
        print(f"\nSummary: {summary_color}{total_passed}{RESET}/{total_tests} "
              f"{summary_color}({total_passed / total_tests * 100:.4f}%){RESET} testcases passed!")
        print(f"Total runtime: {BOLD}{timestampt(global_duration)}{RESET}")

        if all_results:
            slowest_func, slowest_result = max(all_results, key=lambda t: t[1].duration)
            print(
                f"Slowest test: {BOLD}{slowest_func}("
                f"{', '.join(str(r) for r in slowest_result.args)}"
                f"{', ' if slowest_result.args and slowest_result.kwargs else ''}"
                f"{', '.join(f'{k}={v}' for k, v in slowest_result.kwargs.items())}"
                f"){RESET}"
            )


        if all_failures:
            print(f"{RED}Failures:{RESET}")
            for func_name, f in all_failures:
                self._print_failure(func_name, f)

        return report  # <-- return structured results

    def _print_failure(self, func_name: str, failure: TestResult) -> None:
        timing = f"{timestampt(failure.duration)}"
        slow_flag = f" {YELLOW}⚠ slow{RESET}" if failure.duration > self.slow_threshold else ""
        print(RED, end='')
        if failure.error is None:
            print(
                f"\t↳ {func_name}({', '.join(f'{r}' for r in failure.args)}{', '.join(f'{k}={v}' for k, v in failure.kwargs.items())}); expected {failure.expected}, "
                f"got {BOLD}{failure.result}{RESET}{RED} {ITALIC}[t+{timing}]{slow_flag}{RESET}\n"
            )
        else:
            err = failure.error
            print(
                f"\t↳ {func_name}({', '.join(f'{r}' for r in failure.args)}{', '.join(f'{k}={v}' for k, v in failure.kwargs.items())}); expected {failure.expected}, "
                f"raised {BOLD}{type(err['exception']).__name__}: '{err['exception']}' {ITALIC}[t+{timing}]{slow_flag}{RESET}\n"
                f"\t\t{RED}↳ caught in file: {BOLD}'{err['filename']}' at line {err['line_no']}{RESET}\n"
                f"\t\t\t{RED}↳ {err['code']}\n"
            )
            if self.show_traceback:
                print("\tFull traceback:")
                traceback.print_exception(type(err['exception']), err['exception'], err['exception'].__traceback__)
        if failure.stdout:
            self._print_stdout(failure, RED)
        print(RESET, end='')

    def _print_stdout(self, result: TestResult, color: str = GREEN, tabbing: int = 2):
        print(f"{'    '*tabbing}{color}↳ stdout {ITALIC}[t={format_time(result.time)}]:{RESET}")
        for line in result.stdout:
            print(f"{color}{'    '*tabbing}\t↳ {ITALIC}[t+{timestampt(line[0])}]:\t{RESET}{color}{line[1]}")
        print(RESET)

def timestampt(t: float) -> str:
    """
    Format a time in seconds into a human-readable string with SI units:
    ns, µs, ms, or s, using 3 significant digits.
    """
    if t < 1e-6:
        # nanoseconds
        value = t * 1e9
        unit = "ns"
    elif t < 1e-3:
        # microseconds
        value = t * 1e6
        unit = "µs"
    elif t < 1:
        # milliseconds
        value = t * 1e3
        unit = "ms"
    else:
        # seconds
        value = t
        unit = "s"

    # Use .4f if value < 10, else .2f for readability
    if value < 10:
        formatted = f"{value:.4f}"
    else:
        formatted = f"{value:.2f}"

    return f"{formatted}{unit}"

def format_time(dt = datetime.now()):
    ms = dt.microsecond // 1000
    us = dt.microsecond % 1000
    return f"{dt.strftime('%H:%M:%S;')}{ms:03d}ms;{us:03d}µs"
