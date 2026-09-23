from __future__ import annotations

import keyword
import unicodedata

from slanq.ast_nodes import (
    AmplitudeList,
    Assign,
    AugAssign,
    BinaryOp,
    Block,
    BoolType,
    Break,
    BuiltinDecl,
    Call,
    ClassicalDecl,
    ComplexType,
    Continue,
    Declaration,
    Expression,
    ExprStatement,
    For,
    If,
    Index,
    IntType,
    Literal,
    LoopVarDecl,
    Name,
    Node,
    ParamArrayDecl,
    ParamDecl,
    ProbList,
    ProcessDef,
    ProcParam,
    Program,
    QIf,
    QIntType,
    QuantumDecl,
    RealtimeDecl,
    RealtimeIf,
    RealtimeWhile,
    Span,
    Statement,
    Symbol,
    Type,
    UnaryOp,
    UnknownValue,
    is_quantum,
    qubit_count,
)
from slanq.builtin import (
    BUILTIN_SCOPE,
    RANGE,
    ArgKind,
    ConstEvalError,
    const_int,
    const_value,
)
from slanq.diagnostics import DiagnosticBag
from slanq.passes import BEFORE_DECLARATION, expand_processes, track_values, unroll_loops
from slanq.visitor import NodeVisitor, iter_child_nodes

Scope = dict[str, Symbol]


def analyze(ast: Program, bag: DiagnosticBag) -> None:
    scope = _build_symbol_table(ast, bag)
    _resolve_names(ast, bag, scope)
    _check_process_calls(ast, bag)
    expand_processes(ast, bag)
    track_values(ast)
    _check_loops(ast, bag)
    unroll_loops(ast, bag)
    # Twice: the unroller needs the loop bound's value, and unrolling then
    # produces one copy of the body per iteration, each with its own values.
    track_values(ast)
    _check_types(ast, bag)
    _check_affine(ast, bag)


def _build_symbol_table(ast: Program, bag: DiagnosticBag) -> Scope:
    """Collect the top-level declarations so they can be referenced before their
    definition; nested scopes are populated during name resolution instead."""
    scope: Scope = {}
    for statement in ast.statements:
        if not isinstance(statement, Declaration):
            continue
        if statement.name in BUILTIN_SCOPE:
            _error(bag, _shadowing_message(statement.name), statement.span)
            continue
        if _is_unusable_target_name(statement.name):
            _error(bag, _unusable_name_message(statement.name), statement.span)
            continue
        if statement.name in scope:
            _error(bag, f"duplicate declaration of '{statement.name}'", statement.span)
            continue
        scope[statement.name] = statement
    return scope


def _shadowing_message(name: str) -> str:
    return f"'{name}' is built into the language and cannot be redeclared"


# Bare identifiers the generated Python file emits as a module-level or
# build_circuit()-scope name, unconditionally or depending on which features
# a program uses -- a Slanq declaration reusing one of these would either
# break the generated file's syntax (a Python keyword) or shadow an internal
# codegen name (silently breaking or crashing it at runtime, since it is a
# plain identifier, not a string, and every occurrence in the emitted source
# is this exact bare word). Must be kept in sync with codegen.py's literals --
# but only for the names that do not start with '_', since those the leading
# underscore rule already reserves wholesale, however many of them appear
# later (`_ancilla_N`, `_qif_body_N`, `_round`, `_sqrt`).
_RESERVED_CODEGEN_NAMES = frozenset(
    {
        "sys",
        "math",
        "cmath",
        "np",
        "QuantumCircuit",
        "QuantumRegister",
        "ClassicalRegister",
        "Parameter",
        "StatePreparation",
        "XGate",
        "CDKMRippleCarryAdder",
        "HRSCumulativeMultiplier",
        "circuit",
        "build_circuit",
        "build_bound_circuit",
        "bindings",
        "used",
    }
)


def _is_unusable_target_name(name: str) -> bool:
    return (
        keyword.iskeyword(name)
        or name.startswith("_")
        or name in _RESERVED_CODEGEN_NAMES
        or not _survives_into_python(name)
    )


def _survives_into_python(name: str) -> bool:
    """Slanq names may use letters beyond ASCII, and so may Python -- but not
    quite the same set, and Python also normalizes identifiers to NFKC, which
    can silently merge two distinct names into one. A name that is already
    normalized and is a Python identifier reaches the generated file as
    itself."""
    return name.isidentifier() and unicodedata.normalize("NFKC", name) == name


def _unusable_name_message(name: str) -> str:
    if keyword.iskeyword(name):
        return (
            f"'{name}' is a reserved Python keyword and cannot be used as a "
            "Slanq name, because it would break the generated file"
        )
    if name.startswith("_"):
        return f"'{name}' starts with '_', which is reserved for the compiler's own generated names"
    if not _survives_into_python(name):
        return (
            f"'{name}' cannot be used as a Slanq name, because it would not "
            "reach the generated file unchanged"
        )
    return f"'{name}' is reserved for the compiler's own generated code"


class _NameResolver(NodeVisitor):
    def __init__(self, global_scope: Scope, bag: DiagnosticBag) -> None:
        self.scopes: list[Scope] = [dict(BUILTIN_SCOPE), global_scope]
        self.bag = bag

    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, Declaration):
            self._declare(node)

    def visit_Block(self, node: Block) -> None:
        self.scopes.append({})
        for statement in node.statements:
            self.visit(statement)
        self.scopes.pop()

    def visit_For(self, node: For) -> None:
        # The loop variable is not yet in scope in the range expression itself.
        self.visit(node.iterable)
        self.scopes.append({})
        self._declare(node.binding)
        self.visit(node.body)
        self.scopes.pop()

    def visit_ProcessDef(self, node: ProcessDef) -> None:
        self._declare(node)
        self.scopes.append({})
        for parameter in node.params:
            self._declare(parameter)
        self.visit(node.body)
        self.scopes.pop()

    def visit_Name(self, node: Name) -> None:
        symbol = self._lookup(node.name)
        if symbol is None:
            _error(self.bag, f"undefined name '{node.name}'", node.span)
        else:
            node.resolved_symbol = symbol

    def _declare(self, symbol: Symbol) -> None:
        current = self.scopes[-1]
        existing = current.get(symbol.name)
        if existing is symbol:
            return
        if symbol.name in BUILTIN_SCOPE:
            _error(self.bag, _shadowing_message(symbol.name), symbol.span)
            return
        if _is_unusable_target_name(symbol.name):
            _error(self.bag, _unusable_name_message(symbol.name), symbol.span)
            return
        if existing is not None:
            _error(self.bag, f"duplicate declaration of '{symbol.name}'", symbol.span)
            return
        current[symbol.name] = symbol

    def _lookup(self, name: str) -> Symbol | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None


def _resolve_names(ast: Program, bag: DiagnosticBag, scope: Scope) -> None:
    _NameResolver(scope, bag).visit(ast)


class _Checker(NodeVisitor):
    """Shared plumbing for the checks that evaluate constant subexpressions."""

    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag

    def _value(self, expression: Expression) -> int | float | bool | complex | None:
        try:
            return const_value(expression)
        except ConstEvalError as exc:
            self._reject(str(exc), expression)
            return None

    def _int(self, expression: Expression) -> int | None:
        try:
            return const_int(expression)
        except ConstEvalError as exc:
            self._reject(str(exc), expression)
            return None

    def _no_value_reason(self, expression: Expression) -> str:
        """A suffix naming the variable that has no value here, when one is
        the reason the expression could not be evaluated."""
        for node in _walk(expression):
            if isinstance(node, Name) and isinstance(node.effective_value, UnknownValue):
                return f" -- '{node.name}': {node.effective_value.reason}"
        return ""

    def _reject(self, message: str, node: Node) -> None:
        _error(self.bag, message, node.span)

    def _warn(self, message: str, node: Node) -> None:
        self.bag.warning(message, line=node.span.start_line, column=node.span.start_col)


class _IndexChecker(_Checker):
    def visit_Index(self, node: Index) -> None:
        self.generic_visit(node)

        size = _quantum_size_of(node.base)
        if size is None:
            self._check_param_array_index(node)
            return

        index = self._int(node.index)
        if index is None:
            self._reject(
                "a quantum register index must be an integer the compiler can compute"
                + self._no_value_reason(node.index),
                node.index,
            )
            return

        self._check_bounds(node, size)

    def _check_param_array_index(self, node: Index) -> None:
        symbol = node.base.resolved_symbol
        if not isinstance(symbol, ParamArrayDecl):
            return
        if self._int(node.index) is None:
            self._reject(
                "a param array index must be an integer the compiler can compute"
                + self._no_value_reason(node.index),
                node.index,
            )
            return
        self._check_bounds(node, symbol.size)

    def _check_bounds(self, node: Index, size: int) -> None:
        index = self._int(node.index)
        if index is not None and not 0 <= index < size:
            self._reject(
                f"index {index} is out of range for '{node.base.name}' of size {size}",
                node.index,
            )


class _LoopVarAssignChecker(_Checker):
    """Runs before unrolling, which substitutes the loop variable away: after
    it, the target of `i = 5` is a literal and the mistake is unrecognisable."""

    def visit_Assign(self, node: Assign) -> None:
        self.generic_visit(node)
        self._check(node.target)

    def visit_AugAssign(self, node: AugAssign) -> None:
        self.generic_visit(node)
        self._check(node.target)

    def _check(self, target: Expression) -> None:
        if isinstance(target, Name) and isinstance(target.resolved_symbol, LoopVarDecl):
            self._reject(
                f"'{target.name}' is a loop variable and cannot be assigned to",
                target,
            )


class _ProgramChecker(_Checker):
    """An empty file is a legal program -- it builds an empty circuit -- but it
    is much more often a file that is not finished yet than one that is meant
    to be empty."""

    def visit_Program(self, node: Program) -> None:
        self.generic_visit(node)
        if not node.statements:
            self._warn("this program is empty, so the circuit it builds is too", node)


class _DeclarationChecker(_Checker):
    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        self.generic_visit(node)

        size = qubit_count(node.declared_type)
        if size is not None and size < 1:
            self._reject(
                f"'{node.name}' needs at least one qubit; a register with none holds no value",
                node,
            )
            return
        if size is None or not isinstance(node.initializer, Expression):
            return

        value = self._value(node.initializer)
        if isinstance(value, bool) and isinstance(node.declared_type, QIntType):
            # `qbool q = 1` stays legal, because 0 and 1 are what one qubit holds
            self._reject(
                f"'{node.name}' is a qint, so it holds a number, not a boolean",
                node.initializer,
            )
            return
        if not isinstance(value, int):
            return

        largest = 2**size - 1
        if not 0 <= value <= largest:
            self._reject(
                f"{value} does not fit in '{node.name}', which holds {size} "
                f"qubit(s) (allowed: 0..{largest})",
                node.initializer,
            )

    def visit_ParamArrayDecl(self, node: ParamArrayDecl) -> None:
        self.generic_visit(node)
        if node.size < 1:
            self._reject(
                f"'{node.name}' needs at least one element; no index is in "
                "range for an empty array",
                node,
            )


# Below this, a probability/amplitude list is treated as "meant to already be
# normalized": floating point noise from ordinary decimal literals is silently
# normalized away rather than warned about.
NORMALIZATION_TOLERANCE = 1e-9


class _ProbListChecker(_Checker):
    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        self.generic_visit(node)

        if not isinstance(node.initializer, ProbList):
            return

        probabilities = node.initializer.probabilities
        if not probabilities:
            # An empty list means equal superposition; nothing to validate.
            return

        size = qubit_count(node.declared_type)
        expected = 2**size
        if len(probabilities) != expected:
            self._reject(
                f"'{node.name}' needs {expected} probabilities (2^{size}), "
                f"got {len(probabilities)}",
                node.initializer,
            )
            return

        for index, probability in enumerate(probabilities):
            if not 0 <= probability <= 1:
                self._reject(
                    f"the probability {probability} for value {index} of "
                    f"'{node.name}' is out of range (must be between 0 and 1)",
                    node.initializer,
                )
                return

        total = sum(probabilities)
        if total <= 0:
            self._reject(
                f"the probabilities for '{node.name}' cannot sum to zero", node.initializer
            )
        elif abs(total - 1.0) > NORMALIZATION_TOLERANCE:
            self._warn(
                f"the probabilities for '{node.name}' sum to {total}, not 1; "
                "the compiler will normalize them",
                node.initializer,
            )


class _AmplitudeListChecker(_Checker):
    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        self.generic_visit(node)

        if not isinstance(node.initializer, AmplitudeList):
            return

        elements = node.initializer.elements
        if not elements:
            # An empty list means equal superposition; nothing to validate.
            return

        size = qubit_count(node.declared_type)
        if size is None:
            return
        expected = 2**size
        if len(elements) != expected:
            self._reject(
                f"'{node.name}' needs {expected} amplitudes (2^{size}), got {len(elements)}",
                node.initializer,
            )
            return

        amplitudes: list[complex | float | int] = []
        for index, element in enumerate(elements):
            value = self._amplitude_value(element, index, node.name)
            if value is None:
                return
            amplitudes.append(value)

        norm_squared = sum(abs(value) ** 2 for value in amplitudes)
        if norm_squared == 0:
            self._reject(f"the amplitudes for '{node.name}' cannot all be zero", node.initializer)
        elif abs(norm_squared - 1.0) > NORMALIZATION_TOLERANCE:
            self._warn(
                f"the amplitudes for '{node.name}' have squared norm {norm_squared}, "
                "not 1; the compiler will normalize them",
                node.initializer,
            )

    def _amplitude_value(
        self, element: Expression, index: int, name: str
    ) -> complex | float | int | None:
        try:
            value = const_value(element)
        except ConstEvalError as exc:
            self._reject(str(exc), element)
            return None
        if value is None:
            self._reject(
                f"the amplitude for value {index} of '{name}' is not a compile-time constant",
                element,
            )
            return None
        if type(value) is bool:
            self._reject(
                f"the amplitude for value {index} of '{name}' cannot be a boolean",
                element,
            )
            return None
        return value


class _ForChecker(_Checker):
    """The parts of a loop header that mean the same before and after
    substitution. What the iteration count actually is belongs to unrolling
    instead: a nested loop's range may name the variable of the loop around it
    (`range(i)`), which is only a constant once that one is unrolled."""

    def visit_For(self, node: For) -> None:
        self.generic_visit(node)

        if not isinstance(node.binding.declared_type, IntType):
            self._reject(
                f"a loop variable counts iterations, so it is an int; "
                f"'{node.binding.name}' cannot be another type",
                node.binding,
            )

        iterable = node.iterable
        if not (isinstance(iterable, Call) and iterable.callee.name == RANGE):
            self._reject(f"a for loop iterates over {RANGE}(...)", iterable)
            return

        if not 1 <= len(iterable.args) <= 3:
            self._reject(
                f"{RANGE}() takes a stop, a start and a stop, or a start, a stop "
                f"and a step -- got {len(iterable.args)} argument(s)",
                iterable,
            )
            return


class _ArithmeticChecker(_Checker):
    """`+=`/`-=` and a quantum-times-quantum initializer are not general
    assignment -- both describe a fixed circuit shape over physical qubits,
    checked here the same way a qif condition is checked, not as ordinary
    expression typing."""

    def visit_AugAssign(self, node: AugAssign) -> None:
        self.generic_visit(node)

        target = node.target
        if isinstance(target, Literal):
            return
        if not isinstance(target, Name):
            self._reject(
                "arithmetic assignment targets a whole quantum variable, not a single qubit",
                target,
            )
            return

        # Anything but a quantum target is ordinary assignment, which
        # _AugAssignChecker reports on; only a quantum one is the adder.
        if not _is_quantum_target(target):
            return

        if node.op == "*=":
            self._reject(
                f"'{target.name}' is a quantum variable, and a product needs a "
                "register of its own to land in; declare one, as in "
                "'qint<n> r = a * b'",
                node,
            )
            return
        if node.op == "/=":
            self._reject(
                f"'{target.name}' is a quantum variable, and division is not "
                "reversible, so it has no circuit",
                node,
            )
            return

        if self._require_quantum(target) is not None:
            self._check_value_shape(node.value, target)

    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        self.generic_visit(node)

        initializer = node.initializer
        if isinstance(initializer, BinaryOp) and initializer.op == "*":
            # A fresh declaration cannot alias an operand by name, so there is
            # no self-reference to check here, unlike AugAssign -- but the two
            # operands can still alias each other, which is checked below.
            operands = self._check_multiply_operands(initializer)
            if operands is not None:
                self._check_product_width(node, operands)

    def _check_product_width(self, node: QuantumDecl, operands: tuple[Name, Name]) -> None:
        """A product is at least as wide as its widest factor. The multiplier
        pads both factors to their common width and writes into the target, and
        Qiskit's HRS refuses a target narrower than that"""
        size = qubit_count(node.declared_type)
        widths = [(_quantum_size_of(operand), operand) for operand in operands]
        if size is None or any(width is None for width, _ in widths):
            return
        widest, operand = max(widths, key=lambda pair: pair[0] or 0)
        if widest is None or size >= widest:
            return
        self._reject(
            f"a product needs at least as many qubits as its widest operand; "
            f"'{node.name}' holds {size}, but '{operand.name}' holds {widest}",
            node,
        )

    def _check_value_shape(self, value: Expression, target: Name) -> None:
        if isinstance(value, Name):
            if isinstance(value.resolved_symbol, ParamDecl):
                self._reject_param_addend(value)
            elif self._require_quantum(value) is not None:
                self._reject_self_reference(target, (value,))
            return

        if isinstance(value, Index) and isinstance(value.base.resolved_symbol, ParamArrayDecl):
            self._reject_param_addend(value)
            return

        if isinstance(value, BinaryOp) and value.op == "*":
            operands = self._check_multiply_operands(value)
            if operands is not None:
                self._reject_self_reference(target, operands)
            return

        if self._int(value) is None:
            self._reject(
                "this arithmetic assignment shape is not implemented yet; "
                "only adding/subtracting a quantum variable, a compile-time "
                "constant, or the product of two quantum variables is "
                "supported",
                value,
            )

    def _reject_param_addend(self, value: Expression) -> None:
        self._reject(
            "adding a runtime parameter to a quantum variable is not "
            "implemented yet; only a compile-time constant, another quantum "
            "variable, or the product of two quantum variables is "
            "supported here",
            value,
        )

    def _reject_self_reference(self, target: Name, operands: tuple[Name, ...]) -> None:
        # The adder/multiplier needs the addend to hold its original value
        # for the whole operation (an uncompute step, where there is one,
        # re-reads it) -- if the target is also an operand, its value has
        # already changed underneath that assumption.
        for operand in operands:
            if operand.name == target.name:
                self._reject(
                    f"'{target.name}' cannot appear on both sides of an "
                    "arithmetic assignment -- it is being changed, so it "
                    "cannot also be read as an unchanged operand",
                    operand,
                )
                return

    def _check_multiply_operands(self, value: BinaryOp) -> tuple[Name, Name] | None:
        operands: list[Name] = []
        for operand in (value.left, value.right):
            if not isinstance(operand, Name):
                self._reject(
                    "the product on the right of an arithmetic assignment "
                    "must multiply two whole quantum variables",
                    operand,
                )
                return None
            if self._require_quantum(operand) is None:
                return None
            operands.append(operand)

        left, right = operands
        if left.name == right.name:
            self._reject(
                f"a product cannot use '{left.name}' for both operands -- the "
                "same qubits would enter the multiplier twice",
                right,
            )
            return None
        return left, right

    def _require_quantum(self, name: Name) -> Type | None:
        declared = _declared_type(name)
        if declared is None:
            self._reject(
                f"'{name.name}' is not a quantum variable; arithmetic "
                "assignment applies only to quantum variables",
                name,
            )
        return declared


class _ProcessCallChecker(_Checker):
    """Everything about a `process` call that is a property of the call rather
    than of the body: arity, the aliasing ban, and being called as a statement.
    Runs before expansion, since afterwards the call is gone."""

    def __init__(self, bag: DiagnosticBag) -> None:
        super().__init__(bag)
        self.statement_call: Expression | None = None

    def visit_ExprStatement(self, node: ExprStatement) -> None:
        previous = self.statement_call
        self.statement_call = node.expr
        self.generic_visit(node)
        self.statement_call = previous

    def visit_Call(self, node: Call) -> None:
        self.generic_visit(node)

        name = node.callee.name
        symbol = node.callee.resolved_symbol
        if symbol is None or isinstance(symbol, BuiltinDecl):
            # Undefined (already reported) or a builtin, which _CallChecker owns.
            return

        if not isinstance(symbol, ProcessDef):
            self._reject(f"'{name}' is not a gate, a function or a process", node)
            return

        if node is not self.statement_call:
            self._reject(
                f"'{name}' is a process and returns no value, so it can only "
                "be called as a statement",
                node,
            )
            return

        expected = len(symbol.params)
        if len(node.args) != expected:
            self._reject(f"'{name}' takes {expected} argument(s), got {len(node.args)}", node)
            return

        self._check_no_aliasing(name, node)

    def _check_no_aliasing(self, name: str, node: Call) -> None:
        """A process is inlined, so passing the same variable twice would put
        it on both sides of whatever the body does to its parameters."""
        seen: set[tuple[str, int]] = set()
        aliased: set[tuple[str, int]] = set()
        for argument in node.args:
            bits = _touched_bits(argument) or set()
            aliased |= seen & bits
            seen |= bits
        if aliased:
            names = ", ".join(sorted({bit_name for bit_name, _ in aliased}))
            self._reject(
                f"'{name}' cannot be given the same qubit(s) in more than one "
                f"argument (here: {names})",
                node,
            )


class _CallChecker(_Checker):
    """Arity, argument kinds and broadcast shape for calls to builtins."""

    def __init__(self, bag: DiagnosticBag) -> None:
        super().__init__(bag)
        self.statement_call: Expression | None = None

    def visit_ExprStatement(self, node: ExprStatement) -> None:
        previous = self.statement_call
        self.statement_call = node.expr
        self.generic_visit(node)
        self.statement_call = previous

    def visit_Call(self, node: Call) -> None:
        self.generic_visit(node)

        symbol = node.callee.resolved_symbol
        if not isinstance(symbol, BuiltinDecl):
            # Not a builtin: _ProcessCallChecker already ran and reported
            # whatever this is, before expansion took the valid calls away.
            return

        name = node.callee.name
        signature = symbol.signature
        if signature is None:
            self._reject(f"'{name}' is only valid as the iterable of a for loop", node)
            return

        expected = len(signature.args)
        if len(node.args) != expected:
            self._reject(f"'{name}' takes {expected} argument(s), got {len(node.args)}", node)
            return

        widths: list[int] = []
        qubit_args: list[Expression] = []
        for kind, argument in zip(signature.args, node.args, strict=True):
            width = self._check_argument(name, kind, argument)
            if width is not None and kind is ArgKind.QUBITS:
                widths.append(width)
                qubit_args.append(argument)

        # Qiskit's rule, measured: equal sizes pair up and a single bit spreads
        # over a register, but two registers of different sizes have no pairing.
        registers = {width for width in widths if width > 1}
        if len(registers) > 1:
            sizes = ", ".join(str(width) for width in sorted(registers))
            self._reject(f"'{name}' cannot combine operands of sizes {sizes}", node)

        self._check_no_aliasing(name, qubit_args, node)

        node.inferred_type = signature.returns
        self._check_position(name, signature, node)

    def _check_no_aliasing(self, name: str, qubit_args: list[Expression], node: Call) -> None:
        seen: set[tuple[str, int]] = set()
        aliased: set[tuple[str, int]] = set()
        for argument in qubit_args:
            bits = _touched_bits(argument) or set()
            aliased |= seen & bits
            seen |= bits
        if aliased:
            names = ", ".join(sorted({bit_name for bit_name, _ in aliased}))
            self._reject(
                f"'{name}' cannot use the same qubit(s) in more than one argument (here: {names})",
                node,
            )

    def _check_argument(self, name: str, kind: ArgKind, argument: Expression) -> int | None:
        operand = _quantum_operand(argument)

        if kind is ArgKind.ANGLE:
            if operand is not None:
                self._reject(f"'{name}' expects a number here, not a qubit", argument)
            return None

        if operand is None:
            self._reject(f"'{name}' expects a quantum variable here", argument)
            return None

        declared, indexed = operand
        if kind is ArgKind.QVAR and indexed:
            self._reject(
                f"'{name}' takes a whole quantum variable, not a single qubit",
                argument,
            )
            return None

        return 1 if indexed else qubit_count(declared)

    def _check_position(self, name: str, signature, node: Call) -> None:
        in_statement = node is self.statement_call
        if in_statement and signature.returns is not None:
            self._reject(f"the result of '{name}' must be assigned", node)
        elif not in_statement and signature.returns is None:
            self._reject(f"'{name}' does not return a value", node)


# Comparisons other than "==" need a different circuit shape than a single
# ctrl_state pattern (an inequality has no single matching bit-pattern) and
# are deferred; listed explicitly so they get a clear message, not a generic one.
_UNSUPPORTED_QIF_COMPARISONS = frozenset({"<", ">", "<=", ">="})


class _QIfConditionChecker(_Checker):
    """A qif condition is not a general boolean expression that happens to
    need a quantum ingredient -- it is exclusively a description of a
    ctrl_state pattern over physical qubits. A classical value has no such
    pattern, so every clause, individually, must resolve to a quantum
    variable; that single rule covers both a purely classical condition and
    one that mixes classical and quantum clauses -- there is no second rule."""

    def visit_QIf(self, node: QIf) -> None:
        self.generic_visit(node)

        condition = node.condition
        if isinstance(condition, UnaryOp) and condition.op == "!":
            self._check_and_chain(condition.operand)
        else:
            self._check_and_chain(condition)

        self._check_no_overlap(node)

    def _check_and_chain(self, condition: Expression) -> None:
        if isinstance(condition, BinaryOp) and condition.op == "&&":
            self._check_and_chain(condition.left)
            self._check_clause(condition.right)
            return
        if isinstance(condition, BinaryOp) and condition.op == "||":
            self._reject("'||' in a qif condition is not implemented yet", condition)
            return
        self._check_clause(condition)

    def _check_clause(self, clause: Expression) -> None:
        # `!=` is sugar for a negated equality test -- same rules as `!(a==c)`.
        if isinstance(clause, BinaryOp) and clause.op == "!=":
            self._check_equality_clause(clause)
            return

        if isinstance(clause, UnaryOp) and clause.op == "!":
            inner = clause.operand
            if isinstance(inner, BinaryOp) and inner.op in ("&&", "||"):
                self._reject(
                    "a negated compound condition may only appear at the very "
                    "top of a qif condition; deeper nesting is not implemented yet",
                    clause,
                )
                return
            self._check_clause_shape(inner)
            return
        self._check_clause_shape(clause)

    def _check_clause_shape(self, clause: Expression) -> None:
        if isinstance(clause, Name):
            self._require_single_qubit(clause)
            return
        if isinstance(clause, Index):
            self._require_quantum(clause.base)
            return
        if isinstance(clause, BinaryOp) and clause.op in ("==", "!="):
            self._check_equality_clause(clause)
            return
        if isinstance(clause, BinaryOp) and clause.op in _UNSUPPORTED_QIF_COMPARISONS:
            self._reject(f"'{clause.op}' in a qif condition is not implemented yet", clause)
            return
        self._reject("this qif condition shape is not implemented yet", clause)

    def _check_equality_clause(self, clause: BinaryOp) -> None:
        name, literal = _split_equality(clause)
        if name is None:
            self._reject(
                "a qif equality test must compare a quantum variable to a constant",
                clause,
            )
            return

        declared = self._require_quantum(name)
        if declared is None:
            return

        size = qubit_count(declared)
        if size is None:
            # An unsized qint parameter; the lowering reports this on its own.
            return

        value = self._value(literal)
        if not isinstance(value, int):  # also accepts bool -- true/false fit in 1 qubit
            self._reject(
                "a qif equality test must compare against a compile-time constant",
                literal,
            )
            return

        largest = 2**size - 1
        if not 0 <= int(value) <= largest:
            self._reject(
                f"{int(value)} does not fit in '{name.name}', which holds {size} "
                f"qubit(s) (allowed: 0..{largest})",
                literal,
            )

    def _require_quantum(self, name: Name) -> Type | None:
        declared = _declared_type(name)
        if declared is None:
            self._reject(
                f"a qif condition may only reference quantum variables; "
                f"'{name.name}' is not one -- use if for a classical condition",
                name,
            )
        return declared

    def _require_single_qubit(self, name: Name) -> Type | None:
        declared = self._require_quantum(name)
        if declared is None:
            return None
        size = qubit_count(declared)
        if size is not None and size != 1:
            self._reject(
                f"'{name.name}' has {size} qubits; a bare qif condition needs "
                "exactly one -- index a specific bit, or compare the whole "
                "register with ==",
                name,
            )
            return None
        return declared

    def _check_no_overlap(self, node: QIf) -> None:
        condition_bits = _collect_bits(node.condition)

        body_bits: set[tuple[str, int]] = set()
        for statement in node.body.statements:
            if isinstance(statement, ExprStatement) and isinstance(statement.expr, Call):
                for argument in statement.expr.args:
                    body_bits |= _collect_bits(argument)

        overlap = condition_bits & body_bits
        if overlap:
            names = ", ".join(sorted({name for name, _ in overlap}))
            self._reject(
                f"a qif body may not modify a qubit its own condition tests (here: {names})",
                node.body,
            )


def _collect_bits(expression: Expression) -> set[tuple[str, int]]:
    if isinstance(expression, Name | Index):
        return _touched_bits(expression) or set()
    bits: set[tuple[str, int]] = set()
    for child in iter_child_nodes(expression):
        bits |= _collect_bits(child)
    return bits


def _touched_bits(expression: Expression) -> set[tuple[str, int]] | None:
    """The physical (variable, bit-index) pairs an expression refers to, used to
    detect the same qubit being touched twice (a qif body vs. its condition, or
    two arguments of the same gate call)."""
    if isinstance(expression, Index):
        declared = _declared_type(expression.base)
        if declared is None:
            return None
        try:
            index = const_int(expression.index)
        except ConstEvalError:
            return None
        if index is None:
            return None
        return {(expression.base.name, index)}

    if isinstance(expression, Name):
        declared = _declared_type(expression)
        if declared is None:
            return None
        size = qubit_count(declared)
        if size is None:
            return None
        return {(expression.name, bit) for bit in range(size)}

    return None


def _split_equality(clause: BinaryOp) -> tuple[Name | None, Expression]:
    """The quantum-variable side and the constant side of `==`/`!=`, in
    whichever order the programmer wrote them (`a == 2` or `2 == a`)."""
    if isinstance(clause.left, Name):
        return clause.left, clause.right
    if isinstance(clause.right, Name):
        return clause.right, clause.left
    return None, clause.right


def _quantum_operand(expression: Expression) -> tuple[Type, bool] | None:
    """The declared type of a qubit operand and whether it is indexed."""
    if isinstance(expression, Index):
        declared = _declared_type(expression.base)
        return (declared, True) if declared is not None else None
    if isinstance(expression, Name):
        declared = _declared_type(expression)
        return (declared, False) if declared is not None else None
    return None


def _declared_type(name: Name) -> Type | None:
    symbol = name.resolved_symbol
    if not isinstance(symbol, QuantumDecl | ProcParam):
        return None
    return symbol.declared_type if is_quantum(symbol.declared_type) else None


def _quantum_size_of(base: Name) -> int | None:
    symbol = base.resolved_symbol
    if not isinstance(symbol, QuantumDecl):
        return None
    return qubit_count(symbol.declared_type)


class _QIfBodyChecker(_Checker):
    """Runs after process expansion and loop unrolling, so a body that only
    ever contained a call to an empty process, or a loop that turned out to
    run zero times, is empty here too -- and an empty body is the one shape
    that makes the whole qif, condition included, generate nothing."""

    def visit_QIf(self, node: QIf) -> None:
        self.generic_visit(node)
        if not node.body.statements:
            self._warn("this qif body is empty, so nothing is conditional", node.body)


MEASURE = "measure"

COMPARISONS: frozenset[str] = frozenset({"==", "!=", "<", "<=", ">", ">="})


def _is_boolean(expression: Expression) -> bool:
    """Whether `expression` denotes a truth value. Structural rather than
    evaluated, so it still holds once a classical variable can be reassigned
    and its value is no longer its initializer."""
    if isinstance(expression, Literal):
        return type(expression.value) is bool
    if isinstance(expression, BinaryOp):
        return expression.op in COMPARISONS or expression.op in ("&&", "||")
    if isinstance(expression, UnaryOp):
        return expression.op == "!"
    if isinstance(expression, Name):
        symbol = expression.resolved_symbol
        return isinstance(symbol, ClassicalDecl | ParamDecl | RealtimeDecl) and isinstance(
            symbol.declared_type, BoolType
        )
    if isinstance(expression, Index):
        symbol = expression.base.resolved_symbol
        return isinstance(symbol, ParamArrayDecl) and isinstance(symbol.declared_type, BoolType)
    return False


def _is_measurement(expression: Expression) -> bool:
    return isinstance(expression, Call) and expression.callee.name == MEASURE


def _touches_realtime(expression: Expression) -> bool:
    return any(isinstance(symbol, RealtimeDecl) for symbol in _declarations_in(expression))


def _declarations_in(expression: Expression) -> list[Declaration]:
    found = []
    for node in _walk(expression):
        if isinstance(node, Name) and isinstance(node.resolved_symbol, Declaration):
            found.append(node.resolved_symbol)
    return found


def _walk(node: Node):
    yield node
    for child in iter_child_nodes(node):
        yield from _walk(child)


class _MeasurementDeclChecker(_Checker):
    """Inside a qif body the declaration form is beside the point -- a
    measurement cannot stand there at all -- so the rule that says so reports
    alone."""

    def __init__(self, bag: DiagnosticBag) -> None:
        super().__init__(bag)
        self._in_qif = False

    def visit_QIf(self, node: QIf) -> None:
        outer, self._in_qif = self._in_qif, True
        try:
            self.generic_visit(node)
        finally:
            self._in_qif = outer

    def visit_ClassicalDecl(self, node: ClassicalDecl) -> None:
        if _is_measurement(node.initializer) and not self._in_qif:
            self._reject(
                f"a measurement result is real-time; declare '{node.name}' as rt int<>",
                node,
            )


class _RealtimeDeclChecker(_Checker):
    def visit_RealtimeDecl(self, node: RealtimeDecl) -> None:
        self.generic_visit(node)

        if _is_measurement(node.initializer):
            self._check_measured_width(node)
            return

        self._warn(
            f"'{node.name}' is a real-time variable that is not a measurement "
            "result; Aer 0.17.2 crashes on reading one in a condition",
            node,
        )
        value = self._value(node.initializer)
        if value is None:
            self._reject(
                f"'{node.name}' must start from a measurement or a value known "
                "while the circuit is being built",
                node,
            )
            return
        if not isinstance(value, int):
            self._reject(
                f"a real-time variable holds a whole number or a truth value, "
                f"so '{node.name}' cannot start from {value!r}",
                node,
            )
            return
        if value < 0:
            self._reject(
                f"a real-time variable is unsigned, so '{node.name}' cannot start from {value}",
                node,
            )
            return
        needed = max(1, int(value).bit_length())
        if node.width is None:
            node.width = needed
        elif node.width < 1:
            self._reject(f"'{node.name}' must be at least one bit wide", node)
        elif needed > node.width:
            self._reject(
                f"{value} does not fit in the {node.width} bits of '{node.name}'",
                node,
            )

    def _check_measured_width(self, node: RealtimeDecl) -> None:
        call = node.initializer
        assert isinstance(call, Call)
        if len(call.args) != 1 or not isinstance(call.args[0], Name):
            return
        symbol = call.args[0].resolved_symbol
        if not isinstance(symbol, QuantumDecl):
            return
        measured = qubit_count(symbol.declared_type)
        if measured is None:
            return
        if node.width is None:
            node.width = measured
        elif node.width != measured:
            self._reject(
                f"'{symbol.name}' measures into {measured} bits, not the "
                f"{node.width} declared for '{node.name}'",
                node,
            )


class _RealtimeIfChecker(_Checker):
    """`rt if`. The branch is taken while the circuit runs, so its condition
    has to name something that exists then."""

    def visit_RealtimeIf(self, node: RealtimeIf) -> None:
        self.generic_visit(node)

        for symbol in _declarations_in(node.condition):
            if isinstance(symbol, QuantumDecl):
                self._reject(
                    f"'{symbol.name}' is a quantum variable, so it cannot be an "
                    "rt if condition; use qif",
                    node.condition,
                )
                return
            if isinstance(symbol, ParamDecl | ParamArrayDecl):
                self._reject(
                    f"'{symbol.name}' is a param, which is bound after the "
                    "circuit is built, so it cannot be an rt if condition",
                    node.condition,
                )
                return

        if not _is_boolean(node.condition):
            self._reject("an rt if condition must be a true/false value", node.condition)
            return

        if not _touches_realtime(node.condition):
            self._reject(
                "an rt if condition must read a real-time value; use if",
                node.condition,
            )
            return

        self._check_arithmetic(node.condition)
        self._check_range(node.condition)
        _check_no_declarations(self, node.body, "an rt if")
        if node.orelse is not None:
            _check_no_declarations(self, node.orelse, "an rt if")

        if not node.body.statements:
            self._warn("this rt if body is empty", node.body)
        if node.orelse is not None and not node.orelse.statements:
            self._warn("this else body is empty", node.orelse)

    def _check_arithmetic(self, expression: Expression) -> None:
        if isinstance(expression, BinaryOp):
            if expression.op in ("+", "-", "*", "/", "%", "**") and _touches_realtime(expression):
                self._warn(
                    "Aer 0.17.2 cannot run arithmetic on a real-time value",
                    expression,
                )
            self._check_arithmetic(expression.left)
            self._check_arithmetic(expression.right)
        elif isinstance(expression, UnaryOp):
            self._check_arithmetic(expression.operand)

    def _check_range(self, expression: Expression) -> None:
        """A real-time value of N bits holds 0..2**N-1, so a comparison against
        a constant outside that range has one answer for every run."""
        if not isinstance(expression, BinaryOp):
            if isinstance(expression, UnaryOp):
                self._check_range(expression.operand)
            return
        if expression.op in ("&&", "||"):
            self._check_range(expression.left)
            self._check_range(expression.right)
            return
        if expression.op not in COMPARISONS:
            return

        for side, other in (
            (expression.left, expression.right),
            (expression.right, expression.left),
        ):
            width = _realtime_width(side)
            if width is None:
                continue
            value = self._value(other)
            if type(value) is not int:
                continue
            answer = _constant_answer(expression.op, width, value, side is expression.left)
            if answer is not None:
                self._warn(
                    f"this comparison is always {str(answer).lower()}: a "
                    f"{width}-bit real-time value is between 0 and "
                    f"{2**width - 1}",
                    expression,
                )
            return


def _realtime_width(expression: Expression) -> int | None:
    if not isinstance(expression, Name):
        return None
    symbol = expression.resolved_symbol
    if isinstance(symbol, RealtimeDecl):
        return symbol.width
    return None


def _constant_answer(op: str, width: int, value: int, realtime_on_left: bool) -> bool | None:
    low, high = 0, 2**width - 1
    if not realtime_on_left:
        op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(op, op)
    outcomes = {_compare(op, candidate, value) for candidate in (low, high)}
    if len(outcomes) == 1 and _compare(op, (low + high) // 2, value) in outcomes:
        return outcomes.pop()
    return None


def _compare(op: str, left: int, right: int) -> bool:
    return {
        "==": left == right,
        "!=": left != right,
        "<": left < right,
        "<=": left <= right,
        ">": left > right,
        ">=": left >= right,
    }[op]


def _check_no_declarations(checker: _Checker, block: Block, construct: str) -> None:
    for statement in block.statements:
        if isinstance(statement, Declaration):
            checker._reject(
                f"'{statement.name}' cannot be declared inside {construct} body",
                statement,
            )


class _BeforeDeclarationChecker(_Checker):
    """Every classical variable is a Python variable in the generated file, so
    reading one above its own declaration would find nothing there."""

    def visit_Name(self, node: Name) -> None:
        self.generic_visit(node)
        value = node.effective_value
        if isinstance(value, UnknownValue) and value.reason == BEFORE_DECLARATION:
            self._reject(f"'{node.name}' is read before its declaration", node)


class _RealtimeBodyAssignChecker(_Checker):
    """A real-time body's Python runs when the circuit is built, so a build-time
    assignment inside one does not wait for the measurement."""

    CONSEQUENCE = {
        "rt if": "would happen whatever the measurement says",
        "rt while": "would happen once, not on every pass",
    }

    def __init__(self, bag: DiagnosticBag) -> None:
        super().__init__(bag)
        self._context: list[str] = []

    def visit_RealtimeIf(self, node: RealtimeIf) -> None:
        self._descend(node, "rt if")

    def visit_RealtimeWhile(self, node: RealtimeWhile) -> None:
        self._descend(node, "rt while")

    def _descend(self, node: Node, label: str) -> None:
        self._context.append(label)
        try:
            self.generic_visit(node)
        finally:
            self._context.pop()

    def visit_Assign(self, node: Assign) -> None:
        self.generic_visit(node)
        self._check(node)

    def visit_AugAssign(self, node: AugAssign) -> None:
        self.generic_visit(node)
        self._check(node)

    def _check(self, node: Assign | AugAssign) -> None:
        if not self._context:
            return
        target = node.target
        if not isinstance(target, Name) or not isinstance(target.resolved_symbol, ClassicalDecl):
            return
        construct = self._context[-1]
        self._reject(
            f"'{target.name}' is settled when the circuit is built, so assigning "
            f"to it inside an {construct} body {self.CONSEQUENCE[construct]}",
            node,
        )


class _IfChecker(_Checker):
    """An `if` is resolved while the circuit is being built, so its condition
    has to hold a value at that point: a quantum variable never does, and a
    `param` is only bound afterwards."""

    def visit_If(self, node: If) -> None:
        self.generic_visit(node)

        for symbol in _declarations_in(node.condition):
            if isinstance(symbol, QuantumDecl):
                self._reject(
                    f"'{symbol.name}' is a quantum variable, so it cannot be an "
                    "if condition; use qif",
                    node.condition,
                )
                return
            if isinstance(symbol, ParamDecl | ParamArrayDecl):
                self._reject(
                    f"'{symbol.name}' is a param, which has no value while the "
                    "circuit is being built, so it cannot be an if condition",
                    node.condition,
                )
                return
            if isinstance(symbol, RealtimeDecl):
                self._reject(
                    f"'{symbol.name}' is a real-time value, which has no value "
                    "while the circuit is being built; use rt if",
                    node.condition,
                )
                return

        if not _is_boolean(node.condition):
            self._reject("an if condition must be a true/false value", node.condition)
            return

        self._value(node.condition)

        _check_no_declarations(self, node.body, "an if")
        if node.orelse is not None:
            _check_no_declarations(self, node.orelse, "an if")

        if not node.body.statements:
            self._warn("this if body is empty", node.body)
        if node.orelse is not None and not node.orelse.statements:
            self._warn("this else body is empty", node.orelse)


ASSIGNABLE_NOT_YET = {
    LoopVarDecl: "a loop variable",
}


class _AssignChecker(_Checker):
    """Plain `=`. Which side of the four phases the target lives on decides
    what may be written to it and when."""

    def visit_Assign(self, node: Assign) -> None:
        self.generic_visit(node)
        self._check(node.target, node.value, node)

    def _check(self, target: Expression, value: Expression, node: Node) -> None:
        if isinstance(target, Literal):
            # Unrolling substituted a loop variable here, and assigning to one
            # is already reported where it still had a name.
            return
        if not isinstance(target, Name):
            self._reject("only a variable can be assigned to", target)
            return

        symbol = target.resolved_symbol
        if isinstance(symbol, QuantumDecl):
            self._reject(
                f"'{symbol.name}' is a quantum variable, which cannot be "
                "assigned to; use reset() or a gate",
                target,
            )
            return
        if isinstance(symbol, ParamDecl | ParamArrayDecl):
            self._reject(f"'{symbol.name}' is a param and is bound, not assigned", target)
            return
        if isinstance(symbol, LoopVarDecl):
            self._reject(f"'{symbol.name}' is a loop variable and cannot be assigned to", target)
            return
        if isinstance(symbol, ProcParam):
            if is_quantum(symbol.declared_type):
                return
            self._reject(
                f"assigning to the process parameter '{symbol.name}' is not "
                "implemented yet; this is a limitation of the compiler, not an "
                "error in the program",
                target,
            )
            return

        if isinstance(symbol, ClassicalDecl) and _touches_realtime(value):
            self._reject(
                f"'{symbol.name}' is settled while the circuit is built, so it "
                "cannot be given a real-time value",
                value,
            )
            return

        if isinstance(symbol, RealtimeDecl):
            _check_realtime_value(self, value)


class _AugAssignChecker(_AssignChecker):
    """`+=`/`-=`. On a quantum variable it describes a reversible adder, which
    `_ArithmeticChecker` handles; on a classical one it is ordinary
    assignment."""

    def visit_Assign(self, node: Assign) -> None:
        self.generic_visit(node)

    def visit_AugAssign(self, node: AugAssign) -> None:
        self.generic_visit(node)
        # An indexed target is _ArithmeticChecker's to reject: only it knows
        # that `+=` addresses a whole register, not one qubit.
        if not isinstance(node.target, Name) or _is_quantum_target(node.target):
            return
        # `m += 1` computes `m + 1`, so the target is part of the arithmetic
        # even though only the right-hand side is written down.
        combined = BinaryOp(
            span=node.span,
            op=node.op.removesuffix("="),
            left=node.target,
            right=node.value,
        )
        self._check(node.target, combined, node)


def _check_realtime_value(checker: _Checker, value: Expression) -> None:
    if _is_measurement(value):
        return
    if any(isinstance(other, QuantumDecl) for other in _declarations_in(value)):
        checker._reject("a quantum variable has no classical value; measure it first", value)
        return
    if _has_realtime_arithmetic(value):
        checker._warn("Aer 0.17.2 cannot run arithmetic on a real-time value", value)


def _is_quantum_target(target: Name) -> bool:
    symbol = target.resolved_symbol
    if isinstance(symbol, QuantumDecl):
        return True
    return isinstance(symbol, ProcParam) and is_quantum(symbol.declared_type)


def _has_realtime_arithmetic(expression: Expression) -> bool:
    if isinstance(expression, BinaryOp):
        if expression.op in ("+", "-", "*", "/", "%", "**") and _touches_realtime(expression):
            return True
        return _has_realtime_arithmetic(expression.left) or _has_realtime_arithmetic(
            expression.right
        )
    if isinstance(expression, UnaryOp):
        return _has_realtime_arithmetic(expression.operand)
    return False


class _RealtimeWhileChecker(_Checker):
    def visit_RealtimeWhile(self, node: RealtimeWhile) -> None:
        self.generic_visit(node)
        if not _check_realtime_condition(self, node.condition, "rt while"):
            return
        _check_no_declarations(self, node.body, "an rt while")
        if _never_settles(node.condition, node.body):
            self._reject(
                "this rt while loop never ends: its body writes nothing its "
                "condition reads, and it has no break",
                node.condition,
            )
        if not node.body.statements:
            self._warn("this rt while body is empty", node.body)


class _LoopControlChecker(_Checker):
    """An rt while is the only loop that reaches the circuit, so it is the only
    one a break or a continue can belong to; a for is unrolled long before."""

    def __init__(self, bag: DiagnosticBag) -> None:
        super().__init__(bag)
        self._depth = 0

    def visit_RealtimeWhile(self, node: RealtimeWhile) -> None:
        self._depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._depth -= 1

    def visit_Break(self, node: Break) -> None:
        self._check(node, "break")

    def visit_Continue(self, node: Continue) -> None:
        self._check(node, "continue")

    def _check(self, node: Node, keyword: str) -> None:
        if not self._depth:
            self._reject(f"'{keyword}' is only meaningful inside an rt while loop", node)


def _check_realtime_condition(checker: _Checker, condition: Expression, construct: str) -> bool:
    for symbol in _declarations_in(condition):
        if isinstance(symbol, QuantumDecl):
            checker._reject(
                f"'{symbol.name}' is a quantum variable, so it cannot be an "
                f"{construct} condition; use qif",
                condition,
            )
            return False
        if isinstance(symbol, ParamDecl | ParamArrayDecl):
            checker._reject(
                f"'{symbol.name}' is a param, which is bound after the circuit "
                f"is built, so it cannot be an {construct} condition",
                condition,
            )
            return False
    if not _is_boolean(condition):
        checker._reject(f"an {construct} condition must be a true/false value", condition)
        return False
    if not _touches_realtime(condition):
        checker._reject(
            f"an {construct} condition must read a real-time value; a loop "
            "settled when the circuit is built is a for",
            condition,
        )
        return False
    return True


def _never_settles(condition: Expression, body: Block) -> bool:
    """Whether nothing the body writes can change the condition's answer. A
    `break` leaves by another route, so its presence settles the question."""
    read = {id(symbol) for symbol in _declarations_in(condition)}
    written: set[int] = set()
    _collect_written(body.statements, written)
    if read & written:
        return False
    return not _contains_break(body.statements)


def _collect_written(statements: list[Statement], found: set[int]) -> None:
    for statement in statements:
        if isinstance(statement, Assign | AugAssign) and isinstance(statement.target, Name):
            symbol = statement.target.resolved_symbol
            if symbol is not None:
                found.add(id(symbol))
        for child in iter_child_nodes(statement):
            if isinstance(child, Block):
                _collect_written(child.statements, found)


def _contains_break(statements: list[Statement]) -> bool:
    for statement in statements:
        if isinstance(statement, Break):
            return True
        if isinstance(statement, RealtimeWhile):
            continue  # its break leaves that loop, not this one
        for child in iter_child_nodes(statement):
            if isinstance(child, Block) and _contains_break(child.statements):
                return True
    return False


class _ParamTypeChecker(_Checker):
    """No Qiskit gate-synthesis primitive ever consumes a `Parameter` except as
    an already-present, real-valued angle or time -- never as a complex value
    (`StatePreparation`, `UnitaryGate` and `PauliEvolutionGate` all
    reject a symbolic/complex input). A `param complex` would therefore never
    have a working consumer."""

    def visit_ParamDecl(self, node: ParamDecl) -> None:
        self._reject_if_complex(node.declared_type, node)

    def visit_ParamArrayDecl(self, node: ParamArrayDecl) -> None:
        self._reject_if_complex(node.declared_type, node)

    def _reject_if_complex(self, declared_type: Type, node: Node) -> None:
        if isinstance(declared_type, ComplexType):
            self._reject(
                "a param must be int, float, or bool; complex is not supported here",
                node,
            )


def _check_process_calls(ast: Program, bag: DiagnosticBag) -> None:
    _ProcessCallChecker(bag).visit(ast)


def _check_loops(ast: Program, bag: DiagnosticBag) -> None:
    _ForChecker(bag).visit(ast)
    _LoopVarAssignChecker(bag).visit(ast)


def _check_types(ast: Program, bag: DiagnosticBag) -> None:
    _ProgramChecker(bag).visit(ast)
    _DeclarationChecker(bag).visit(ast)
    _IndexChecker(bag).visit(ast)
    _CallChecker(bag).visit(ast)
    _ProbListChecker(bag).visit(ast)
    _AmplitudeListChecker(bag).visit(ast)
    _QIfConditionChecker(bag).visit(ast)
    _QIfBodyChecker(bag).visit(ast)
    _IfChecker(bag).visit(ast)
    _BeforeDeclarationChecker(bag).visit(ast)
    _RealtimeBodyAssignChecker(bag).visit(ast)
    _MeasurementDeclChecker(bag).visit(ast)
    _RealtimeDeclChecker(bag).visit(ast)
    _RealtimeIfChecker(bag).visit(ast)
    _AssignChecker(bag).visit(ast)
    _AugAssignChecker(bag).visit(ast)
    _RealtimeWhileChecker(bag).visit(ast)
    _LoopControlChecker(bag).visit(ast)
    _ArithmeticChecker(bag).visit(ast)
    _ParamTypeChecker(bag).visit(ast)


def _check_affine(ast: Program, bag: DiagnosticBag) -> None:
    pass


def _error(bag: DiagnosticBag, message: str, span: Span) -> None:
    bag.error(message, line=span.start_line, column=span.start_col)


__all__ = ["analyze"]
