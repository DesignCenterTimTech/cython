# cython: infer_types=True
# cython: language_level=3
# cython: auto_pickle=False

#
#   Tree visitor and transform framework
#

from __future__ import absolute_import, print_function

import sys
import inspect

from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future

import cython


cython.declare(_PRINTABLE=tuple)

if sys.version_info[0] >= 3:
    _PRINTABLE = (bytes, str, int, float)
else:
    _PRINTABLE = (str, unicode, long, int, float)


class TreeVisitor(object):
    """
    Base class for writing visitors for a Cython tree, contains utilities for
    recursing such trees using visitors. Each node is
    expected to have a child_attrs iterable containing the names of attributes
    containing child nodes or lists of child nodes. Lists are not considered
    part of the tree structure (i.e. contained nodes are considered direct
    children of the parent node).

    visit_children visits each of the children of a given node (see the visit_children
    documentation). When recursing the tree using visit_children, an attribute
    access_path is maintained which gives information about the current location
    in the tree as a stack of tuples: (parent_node, attrname, index), representing
    the node, attribute and optional list index that was taken in each step in the path to
    the current node.

    Example:

    >>> class SampleNode(object):
    ...     child_attrs = ["head", "body"]
    ...     def __init__(self, value, head=None, body=None):
    ...         self.value = value
    ...         self.head = head
    ...         self.body = body
    ...     def __repr__(self): return "SampleNode(%s)" % self.value
    ...
    >>> tree = SampleNode(0, SampleNode(1), [SampleNode(2), SampleNode(3)])
    >>> class MyVisitor(TreeVisitor):
    ...     def visit_SampleNode(self, node):
    ...         print("in %s %s" % (node.value, self.access_path))
    ...         self.visitchildren(node)
    ...         print("out %s" % node.value)
    ...
    >>> MyVisitor().visit(tree)
    in 0 []
    in 1 [(SampleNode(0), 'head', None)]
    out 1
    in 2 [(SampleNode(0), 'body', 0)]
    out 2
    in 3 [(SampleNode(0), 'body', 1)]
    out 3
    out 0
    """
    def __init__(self):
        super(TreeVisitor, self).__init__()
        self.dispatch_table = {}
        self.access_path = []

    def dump_node(self, node):
        ignored = list(node.child_attrs or []) + [
            u'child_attrs', u'pos', u'gil_message', u'cpp_message', u'subexprs']
        values = []
        pos = getattr(node, 'pos', None)
        if pos:
            source = pos[0]
            if source:
                import os.path
                source = os.path.basename(source.get_description())
            values.append(u'%s:%s:%s' % (source, pos[1], pos[2]))
        attribute_names = dir(node)
        for attr in attribute_names:
            if attr in ignored:
                continue
            if attr.startswith('_') or attr.endswith('_'):
                continue
            try:
                value = getattr(node, attr)
            except AttributeError:
                continue
            if value is None or value == 0:
                continue
            elif isinstance(value, list):
                value = u'[...]/%d' % len(value)
            elif not isinstance(value, _PRINTABLE):
                continue
            else:
                value = repr(value)
            values.append(u'%s = %s' % (attr, value))
        return u'%s(%s)' % (node.__class__.__name__, u',\n    '.join(values))

    def _find_node_path(self, stacktrace):
        import os.path
        last_traceback = stacktrace
        nodes = []
        while hasattr(stacktrace, 'tb_frame'):
            frame = stacktrace.tb_frame
            node = frame.f_locals.get(u'self')
            if isinstance(node, Nodes.Node):
                code = frame.f_code
                method_name = code.co_name
                pos = (os.path.basename(code.co_filename),
                       frame.f_lineno)
                nodes.append((node, method_name, pos))
                last_traceback = stacktrace
            stacktrace = stacktrace.tb_next
        return (last_traceback, nodes)

    def _raise_compiler_error(self, child, e):
        trace = ['']
        for parent, attribute, index in self.access_path:
            node = getattr(parent, attribute)
            if index is None:
                index = ''
            else:
                node = node[index]
                index = u'[%d]' % index
            trace.append(u'%s.%s%s = %s' % (
                parent.__class__.__name__, attribute, index,
                self.dump_node(node)))
        stacktrace, called_nodes = self._find_node_path(sys.exc_info()[2])
        last_node = child
        for node, method_name, pos in called_nodes:
            last_node = node
            trace.append(u"File '%s', line %d, in %s: %s" % (
                pos[0], pos[1], method_name, self.dump_node(node)))
        raise Errors.CompilerCrash(
            getattr(last_node, 'pos', None), self.__class__.__name__,
            u'\n'.join(trace), e, stacktrace)

    @cython.final
    def find_handler(self, obj):
        # to resolve, try entire hierarchy
        cls = type(obj)
        pattern = "visit_%s"
        mro = inspect.getmro(cls)
        for mro_cls in mro:
            handler_method = getattr(self, pattern % mro_cls.__name__, None)
            if handler_method is not None:
                return handler_method
        print(type(self), cls)
        if self.access_path:
            print(self.access_path)
            print(self.access_path[-1][0].pos)
            print(self.access_path[-1][0].__dict__)
        raise RuntimeError("Visitor %r does not accept object: %s" % (self, obj))

    def visit(self, obj):
        return self._visit(obj)

    @cython.final
    def _visit(self, obj):
        try:
            try:
                handler_method = self.dispatch_table[type(obj)]
            except KeyError:
                handler_method = self.find_handler(obj)
                self.dispatch_table[type(obj)] = handler_method
            return handler_method(obj)
        except Errors.CompileError:
            raise
        except Errors.AbortError:
            raise
        except Exception as e:
            if DebugFlags.debug_no_exception_intercept:
                raise
            self._raise_compiler_error(obj, e)

    @cython.final
    def _visitchild(self, child, parent, attrname, idx):
        self.access_path.append((parent, attrname, idx))
        result = self._visit(child)
        self.access_path.pop()
        return result

    def visitchildren(self, parent, attrs=None):
        return self._visitchildren(parent, attrs)

    @cython.final
    @cython.locals(idx=cython.Py_ssize_t)
    def _visitchildren(self, parent, attrs):
        """
        Visits the children of the given parent. If parent is None, returns
        immediately (returning None).

        The return value is a dictionary giving the results for each
        child (mapping the attribute name to either the return value
        or a list of return values (in the case of multiple children
        in an attribute)).
        """
        if parent is None: return None
        result = {}
        for attr in parent.child_attrs:
            if attrs is not None and attr not in attrs: continue
            child = getattr(parent, attr)
            if child is not None:
                if type(child) is list:
                    childretval = [self._visitchild(x, parent, attr, idx) for idx, x in enumerate(child)]
                else:
                    childretval = self._visitchild(child, parent, attr, None)
                    assert not isinstance(childretval, list), 'Cannot insert list here: %s in %r' % (attr, parent)
                result[attr] = childretval
        return result


class VisitorTransform(TreeVisitor):
    """
    A tree transform is a base class for visitors that wants to do stream
    processing of the structure (rather than attributes etc.) of a tree.

    It implements __call__ to simply visit the argument node.

    It requires the visitor methods to return the nodes which should take
    the place of the visited node in the result tree (which can be the same
    or one or more replacement). Specifically, if the return value from
    a visitor method is:

    - [] or None; the visited node will be removed (set to None if an attribute and
    removed if in a list)
    - A single node; the visited node will be replaced by the returned node.
    - A list of nodes; the visited nodes will be replaced by all the nodes in the
    list. This will only work if the node was already a member of a list; if it
    was not, an exception will be raised. (Typically you want to ensure that you
    are within a StatListNode or similar before doing this.)
    """
    def visitchildren(self, parent, attrs=None, exclude=None):
        # generic def entry point for calls from Python subclasses
        if exclude is not None:
            attrs = self._select_attrs(parent.child_attrs if attrs is None else attrs, exclude)
        return self._process_children(parent, attrs)

    @cython.final
    def _select_attrs(self, attrs, exclude):
        return [name for name in attrs if name not in exclude]

    @cython.final
    def _process_children(self, parent, attrs=None):
        # fast cdef entry point for calls from Cython subclasses
        result = self._visitchildren(parent, attrs)
        for attr, newnode in result.items():
            if type(newnode) is list:
                newnode = self._flatten_list(newnode)
            setattr(parent, attr, newnode)
        return result

    @cython.final
    def _flatten_list(self, orig_list):
        # Flatten the list one level and remove any None
        newlist = []
        for x in orig_list:
            if x is not None:
                if type(x) is list:
                    newlist.extend(x)
                else:
                    newlist.append(x)
        return newlist

    def recurse_to_children(self, node):
        self._process_children(node)
        return node

    def __call__(self, root):
        return self._visit(root)


class CythonTransform(VisitorTransform):
    """
    Certain common conventions and utilities for Cython transforms.

     - Sets up the context of the pipeline in self.context
     - Tracks directives in effect in self.current_directives
    """
    def __init__(self, context):
        super(CythonTransform, self).__init__()
        self.context = context

    def __call__(self, node):
        from . import ModuleNode
        if isinstance(node, ModuleNode.ModuleNode):
            self.current_directives = node.directives
        return super(CythonTransform, self).__call__(node)

    def visit_CompilerDirectivesNode(self, node):
        old = self.current_directives
        self.current_directives = node.directives
        self._process_children(node)
        self.current_directives = old
        return node

    def visit_Node(self, node):
        self._process_children(node)
        return node


class ScopeTrackingTransform(CythonTransform):
    # Keeps track of type of scopes
    #scope_type: can be either of 'module', 'function', 'cclass', 'pyclass', 'struct'
    #scope_node: the node that owns the current scope

    def visit_ModuleNode(self, node):
        self.scope_type = 'module'
        self.scope_node = node
        self._process_children(node)
        return node

    def visit_scope(self, node, scope_type):
        prev = self.scope_type, self.scope_node
        self.scope_type = scope_type
        self.scope_node = node
        self._process_children(node)
        self.scope_type, self.scope_node = prev
        return node

    def visit_CClassDefNode(self, node):
        return self.visit_scope(node, 'cclass')

    def visit_PyClassDefNode(self, node):
        return self.visit_scope(node, 'pyclass')

    def visit_FuncDefNode(self, node):
        return self.visit_scope(node, 'function')

    def visit_CStructOrUnionDefNode(self, node):
        return self.visit_scope(node, 'struct')


class EnvTransform(CythonTransform):
    """
    This transformation keeps a stack of the environments.
    """
    def __call__(self, root):
        self.env_stack = []
        self.enter_scope(root, root.scope)
        return super(EnvTransform, self).__call__(root)

    def current_env(self):
        return self.env_stack[-1][1]

    def current_scope_node(self):
        return self.env_stack[-1][0]

    def global_scope(self):
        return self.current_env().global_scope()

    def enter_scope(self, node, scope):
        self.env_stack.append((node, scope))

    def exit_scope(self):
        self.env_stack.pop()

    def visit_FuncDefNode(self, node):
        self.enter_scope(node, node.local_scope)
        self._process_children(node)
        self.exit_scope()
        return node

    def visit_GeneratorBodyDefNode(self, node):
        self._process_children(node)
        return node

    def visit_ClassDefNode(self, node):
        self.enter_scope(node, node.scope)
        self._process_children(node)
        self.exit_scope()
        return node

    def visit_CStructOrUnionDefNode(self, node):
        self.enter_scope(node, node.scope)
        self._process_children(node)
        self.exit_scope()
        return node

    def visit_ScopedExprNode(self, node):
        if node.expr_scope:
            self.enter_scope(node, node.expr_scope)
            self._process_children(node)
            self.exit_scope()
        else:
            self._process_children(node)
        return node

    def visit_CArgDeclNode(self, node):
        # default arguments are evaluated in the outer scope
        if node.default:
            attrs = [attr for attr in node.child_attrs if attr != 'default']
            self._process_children(node, attrs)
            self.enter_scope(node, self.current_env().outer_scope)
            self.visitchildren(node, ('default',))
            self.exit_scope()
        else:
            self._process_children(node)
        return node


class NodeRefCleanupMixin(object):
    """
    Clean up references to nodes that were replaced.

    NOTE: this implementation assumes that the replacement is
    done first, before hitting any further references during
    normal tree traversal.  This needs to be arranged by calling
    "self.visitchildren()" at a proper place in the transform
    and by ordering the "child_attrs" of nodes appropriately.
    """
    def __init__(self, *args):
        super(NodeRefCleanupMixin, self).__init__(*args)
        self._replacements = {}

    def visit_CloneNode(self, node):
        arg = node.arg
        if arg not in self._replacements:
            self.visitchildren(arg)
        node.arg = self._replacements.get(arg, arg)
        return node

    def visit_ResultRefNode(self, node):
        expr = node.expression
        if expr is None or expr not in self._replacements:
            self.visitchildren(node)
            expr = node.expression
        if expr is not None:
            node.expression = self._replacements.get(expr, expr)
        return node

    def replace(self, node, replacement):
        self._replacements[node] = replacement
        return replacement


find_special_method_for_binary_operator = {
    '<':  '__lt__',
    '<=': '__le__',
    '==': '__eq__',
    '!=': '__ne__',
    '>=': '__ge__',
    '>':  '__gt__',
    '+':  '__add__',
    '&':  '__and__',
    '/':  '__div__',
    '//': '__floordiv__',
    '<<': '__lshift__',
    '%':  '__mod__',
    '*':  '__mul__',
    '|':  '__or__',
    '**': '__pow__',
    '>>': '__rshift__',
    '-':  '__sub__',
    '^':  '__xor__',
    'in': '__contains__',
}.get


find_special_method_for_unary_operator = {
    'not': '__not__',
    '~':   '__inv__',
    '-':   '__neg__',
    '+':   '__pos__',
}.get


class MethodDispatcherTransform(EnvTransform):
    """
    Base class for transformations that want to intercept on specific
    builtin functions or methods of builtin types, including special
    methods triggered by Python operators.  Must run after declaration
    analysis when entries were assigned.

    Naming pattern for handler methods is as follows:

    * builtin functions: _handle_(general|simple|any)_function_NAME

    * builtin methods: _handle_(general|simple|any)_method_TYPENAME_METHODNAME
    """
    # only visit call nodes and Python operations
    def visit_GeneralCallNode(self, node):
        self._process_children(node)
        function = node.function
        if not function.type.is_pyobject:
            return node
        arg_tuple = node.positional_args
        if not isinstance(arg_tuple, ExprNodes.TupleNode):
            return node
        keyword_args = node.keyword_args
        if keyword_args and not isinstance(keyword_args, ExprNodes.DictNode):
            # can't handle **kwargs
            return node
        args = arg_tuple.args
        return self._dispatch_to_handler(node, function, args, keyword_args)

    def visit_SimpleCallNode(self, node):
        self._process_children(node)
        function = node.function
        if function.type.is_pyobject:
            arg_tuple = node.arg_tuple
            if not isinstance(arg_tuple, ExprNodes.TupleNode):
                return node
            args = arg_tuple.args
        else:
            args = node.args
        return self._dispatch_to_handler(node, function, args, None)

    def visit_PrimaryCmpNode(self, node):
        if node.cascade:
            # not currently handled below
            self._process_children(node)
            return node
        return self._visit_binop_node(node)

    def visit_BinopNode(self, node):
        return self._visit_binop_node(node)

    def _visit_binop_node(self, node):
        self._process_children(node)
        # FIXME: could special case 'not_in'
        special_method_name = find_special_method_for_binary_operator(node.operator)
        if special_method_name:
            operand1, operand2 = node.operand1, node.operand2
            if special_method_name == '__contains__':
                operand1, operand2 = operand2, operand1
            elif special_method_name == '__div__':
                if Future.division in self.current_env().global_scope().context.future_directives:
                    special_method_name = '__truediv__'
            obj_type = operand1.type
            if obj_type.is_builtin_type:
                type_name = obj_type.name
            else:
                type_name = "object"  # safety measure
            node = self._dispatch_to_method_handler(
                special_method_name, None, False, type_name,
                node, None, [operand1, operand2], None)
        return node

    def visit_UnopNode(self, node):
        self._process_children(node)
        special_method_name = find_special_method_for_unary_operator(node.operator)
        if special_method_name:
            operand = node.operand
            obj_type = operand.type
            if obj_type.is_builtin_type:
                type_name = obj_type.name
            else:
                type_name = "object"  # safety measure
            node = self._dispatch_to_method_handler(
                special_method_name, None, False, type_name,
                node, None, [operand], None)
        return node

    ### dispatch to specific handlers

    def _find_handler(self, match_name, has_kwargs):
        call_type = has_kwargs and 'general' or 'simple'
        handler = getattr(self, '_handle_%s_%s' % (call_type, match_name), None)
        if handler is None:
            handler = getattr(self, '_handle_any_%s' % match_name, None)
        return handler

    def _delegate_to_assigned_value(self, node, function, arg_list, kwargs):
        assignment = function.cf_state[0]
        value = assignment.rhs
        if value.is_name:
            if not value.entry or len(value.entry.cf_assignments) > 1:
                # the variable might have been reassigned => play safe
                return node
        elif value.is_attribute and value.obj.is_name:
            if not value.obj.entry or len(value.obj.entry.cf_assignments) > 1:
                # the underlying variable might have been reassigned => play safe
                return node
        else:
            return node
        return self._dispatch_to_handler(
            node, value, arg_list, kwargs)

    def _dispatch_to_handler(self, node, function, arg_list, kwargs):
        if function.is_name:
            # we only consider functions that are either builtin
            # Python functions or builtins that were already replaced
            # into a C function call (defined in the builtin scope)
            if not function.entry:
                return node
            entry = function.entry
            is_builtin = (
                entry.is_builtin or
                entry is self.current_env().builtin_scope().lookup_here(function.name))
            if not is_builtin:
                if function.cf_state and function.cf_state.is_single:
                    # we know the value of the variable
                    # => see if it's usable instead
                    return self._delegate_to_assigned_value(
                        node, function, arg_list, kwargs)
                if arg_list and entry.is_cmethod and entry.scope and entry.scope.parent_type.is_builtin_type:
                    if entry.scope.parent_type is arg_list[0].type:
                        # Optimised (unbound) method of a builtin type => try to "de-optimise".
                        return self._dispatch_to_method_handler(
                            entry.name, self_arg=None, is_unbound_method=True,
                            type_name=entry.scope.parent_type.name,
                            node=node, function=function, arg_list=arg_list, kwargs=kwargs)
                return node
            function_handler = self._find_handler(
                "function_%s" % function.name, kwargs)
            if function_handler is None:
                return self._handle_function(node, function.name, function, arg_list, kwargs)
            if kwargs:
                return function_handler(node, function, arg_list, kwargs)
            else:
                return function_handler(node, function, arg_list)
        elif function.is_attribute:
            attr_name = function.attribute
            if function.type.is_pyobject:
                self_arg = function.obj
            elif node.self and function.entry:
                entry = function.entry.as_variable
                if not entry or not entry.is_builtin:
                    return node
                # C implementation of a Python builtin method - see if we find further matches
                self_arg = node.self
                arg_list = arg_list[1:]  # drop CloneNode of self argument
            else:
                return node
            obj_type = self_arg.type
            is_unbound_method = False
            if obj_type.is_builtin_type:
                if obj_type is Builtin.type_type and self_arg.is_name and arg_list and arg_list[0].type.is_pyobject:
                    # calling an unbound method like 'list.append(L,x)'
                    # (ignoring 'type.mro()' here ...)
                    type_name = self_arg.name
                    self_arg = None
                    is_unbound_method = True
                else:
                    type_name = obj_type.name
            else:
                type_name = "object"  # safety measure
            return self._dispatch_to_method_handler(
                attr_name, self_arg, is_unbound_method, type_name,
                node, function, arg_list, kwargs)
        else:
            return node

    def _dispatch_to_method_handler(self, attr_name, self_arg,
                                    is_unbound_method, type_name,
                                    node, function, arg_list, kwargs):
        method_handler = self._find_handler(
            "method_%s_%s" % (type_name, attr_name), kwargs)
        if method_handler is None:
            if (attr_name in TypeSlots.method_name_to_slot
                    or attr_name == '__new__'):
                method_handler = self._find_handler(
                    "slot%s" % attr_name, kwargs)
            if method_handler is None:
                return self._handle_method(
                    node, type_name, attr_name, function,
                    arg_list, is_unbound_method, kwargs)
        if self_arg is not None:
            arg_list = [self_arg] + list(arg_list)
        if kwargs:
            result = method_handler(
                node, function, arg_list, is_unbound_method, kwargs)
        else:
            result = method_handler(
                node, function, arg_list, is_unbound_method)
        return result

    def _handle_function(self, node, function_name, function, arg_list, kwargs):
        """Fallback handler"""
        return node

    def _handle_method(self, node, type_name, attr_name, function,
                       arg_list, is_unbound_method, kwargs):
        """Fallback handler"""
        return node


class RecursiveNodeReplacer(VisitorTransform):
    """
    Recursively replace all occurrences of a node in a subtree by
    another node.
    """
    def __init__(self, orig_node, new_node):
        super(RecursiveNodeReplacer, self).__init__()
        self.orig_node, self.new_node = orig_node, new_node

    def visit_CloneNode(self, node):
        if node is self.orig_node:
            return self.new_node
        if node.arg is self.orig_node:
            node.arg = self.new_node
        return node

    def visit_Node(self, node):
        self._process_children(node)
        if node is self.orig_node:
            return self.new_node
        else:
            return node

def recursively_replace_node(tree, old_node, new_node):
    replace_in = RecursiveNodeReplacer(old_node, new_node)
    replace_in(tree)


class NodeFinder(TreeVisitor):
    """
    Find out if a node appears in a subtree.
    """
    def __init__(self, node):
        super(NodeFinder, self).__init__()
        self.node = node
        self.found = False

    def visit_Node(self, node):
        if self.found:
            pass  # short-circuit
        elif node is self.node:
            self.found = True
        else:
            self._visitchildren(node, None)

def tree_contains(tree, node):
    finder = NodeFinder(node)
    finder.visit(tree)
    return finder.found


# Utils
def replace_node(ptr, value):
    """Replaces a node. ptr is of the form used on the access path stack
    (parent, attrname, listidx|None)
    """
    parent, attrname, listidx = ptr
    if listidx is None:
        setattr(parent, attrname, value)
    else:
        getattr(parent, attrname)[listidx] = value


class PrintTree(TreeVisitor):
    """Prints a representation of the tree to standard output.
    Subclass and override repr_of to provide more information
    about nodes. """
    def __init__(self, start=None, end=None):
        TreeVisitor.__init__(self)
        self._indent = ""
        if start is not None or end is not None:
            self._line_range = (start or 0, end or 2**30)
        else:
            self._line_range = None

    def indent(self):
        self._indent += "  "

    def unindent(self):
        self._indent = self._indent[:-2]

    def __call__(self, tree, phase=None):
        print("Parse tree dump at phase '%s'" % phase)
        self.visit(tree)
        return tree

    # Don't do anything about process_list, the defaults gives
    # nice-looking name[idx] nodes which will visually appear
    # under the parent-node, not displaying the list itself in
    # the hierarchy.
    def visit_Node(self, node):
        self._print_node(node)
        self.indent()
        self.visitchildren(node)
        self.unindent()
        return node

    def visit_CloneNode(self, node):
        self._print_node(node)
        self.indent()
        line = node.pos[1]
        if self._line_range is None or self._line_range[0] <= line <= self._line_range[1]:
            print("%s- %s: %s" % (self._indent, 'arg', self.repr_of(node.arg)))
        self.indent()
        self.visitchildren(node.arg)
        self.unindent()
        self.unindent()
        return node

    def _print_node(self, node):
        line = node.pos[1]
        if self._line_range is None or self._line_range[0] <= line <= self._line_range[1]:
            if len(self.access_path) == 0:
                name = "(root)"
            else:
                parent, attr, idx = self.access_path[-1]
                if idx is not None:
                    name = "%s[%d]" % (attr, idx)
                else:
                    name = attr
            print("%s- %s: %s" % (self._indent, name, self.repr_of(node)))

    def repr_of(self, node):
        if node is None:
            return "(none)"
        else:
            result = node.__class__.__name__
            if isinstance(node, ExprNodes.NameNode):
                result += "(type=%s, name=\"%s\")" % (repr(node.type), node.name)
            elif isinstance(node, Nodes.DefNode):
                result += "(name=\"%s\")" % node.name
            elif isinstance(node, ExprNodes.ExprNode):
                t = node.type
                result += "(type=%s)" % repr(t)
            if node.pos:
                pos = node.pos
                path = pos[0].get_description()
                if '/' in path:
                    path = path.split('/')[-1]
                if '\\' in path:
                    path = path.split('\\')[-1]
                result += "(pos=(%s:%s:%s))" % (path, pos[1], pos[2])

            return result

""" My Modification start """
from . import ModuleNode
from . import UtilNodes
class PrintTreePy(PrintTree):
    """Prints a python representation of the tree to standard output."""
    def __call__(self, tree, phase=None):
        print("Making a Python file format from AST")
        print(self.print_Node(tree))
        return tree
    
    def print_children(self, parent, attrs = None, exclude = None):
        if parent is None: return None
        result = ""
        for attr in parent.child_attrs:
            if attrs is not None and attr not in attrs: continue
            if exclude is not None and attr in exclude: continue
            child = getattr(parent, attr)
            if child is not None:
                if type(child) is list:
                    for child_element in child:
                        result += self.print_Node(child_element)
                else:
                    result += self.print_Node(child)
        return result
        
    def print_UnknownNode(self, node):
        result = "\n# in %s() found %s\n" % (inspect.stack()[1][3], 
                                             type(node))
        return result
    
    def print_Node(self, node):
        result = ""
        if not node:
            result += "None"
        elif isinstance(node, ModuleNode.ModuleNode):
            result +=         self.print_ModuleNode(node)
        elif isinstance(node, ExprNodes.CmpNode):
            result +=        self.print_CmpNode(node)
        elif isinstance(node, Nodes.LoopNode):
            result +=    self.print_LoopNode(node)
        elif isinstance(node, ExprNodes.ExprNode):
            result +=        self.print_ExprNode(node)
        #elif isinstance(node, ExprNodes.ComprehensionAppendNode):
        #    result +=        self.print_ComprehensionAppendNode(node)
        #elif isinstance(node, UtilNodes.TempsBlockNode):
        #    result +=        self.print_TempsBlockNode(node)
        #elif isinstance(node, Nodes.CompilerDirectivesNode):
        #    result +=    self.print_CompilerDirectivesNode(node)
        elif isinstance(node, Nodes.StatListNode):
            result +=    self.print_StatListNode(node)
        elif isinstance(node, Nodes.StatNode):
            result +=    self.print_StatNode(node)
        elif isinstance(node, Nodes.CDeclaratorNode):
            result +=    self.print_CDeclaratorNode(node)
        elif isinstance(node, Nodes.CArgDeclNode):
            result +=    self.print_CArgDeclNode(node)
        #elif isinstance(node, Nodes.CBaseTypeNode):
        #    result +=    self.print_CBaseTypeNode(node)
        #elif isinstance(node, Nodes.CAnalysedBaseTypeNode):
        #    result +=    self.print_CAnalysedBaseTypeNode(node)
        #elif isinstance(node, Nodes.PyArgDeclNode):
        #    result +=    self.print_PyArgDeclNode(node)
        #elif isinstance(node, Nodes.DecoratorNode):
        #    result +=    self.print_DecoratorNode(node)
        elif isinstance(node, Nodes.IfClauseNode):
            result +=    self.print_IfClauseNode(node)
        #elif isinstance(node, Nodes.DictIterationNextNode):
        #    result +=    self.print_DictIterationNextNode(node)
        #elif isinstance(node, Nodes.SetIterationNextNode):
        #    result +=    self.print_SetIterationNextNode(node)
        #elif isinstance(node, Nodes.ExceptClauseNode):
        #    result +=    self.print_ExceptClauseNode(node)
        #elif isinstance(node, Nodes.ParallelNode):
        #    result +=    self.print_ParallelNode(node)
        else:
            result += self.print_UnknownNode(node)
        return result
#==============================================================================
    def print_ModuleNode(self, node):
        result = self.print_children(node)
        return result
#==============================================================================      
    def print_CmpNode(self, node):
        result = ""
        if isinstance(node, ExprNodes.PrimaryCmpNode):
            result += "%s %s %s%s" % (self.print_Node(node.operand1),
                                          node.operator,
                                          self.print_Node(node.operand2),
                                          self.print_CmpNode(node.cascade))
        elif isinstance(node, ExprNodes.CascadedCmpNode):
            result += " %s %s%s" % (node.operator,
                                        self.print_Node(node.operand2),
                                        self.print_CmpNode(node.cascade))
        return result
#==============================================================================      
    def print_LoopNode(self, node):
        result = ""
        if isinstance(node, Nodes.WhileStatNode):
            result += self.print_WhileStatNode(node)
        elif isinstance(node, Nodes._ForInStatNode):
            result += self.print__ForInStatNode(node)
        elif isinstance(node, Nodes.ForFromStatNode):
            result += self.print_ForFromStatNode(node)
        else:
            result += self.print_UnknownNode(node)
        
        return result
        
    def print_WhileStatNode(self, node):
        result = ""
        return result

    def print__ForInStatNode(self, node):
        result = ""
        if isinstance(node, Nodes.ForInStatNode):
            result += "%sfor %s in %s:\n" % (self._indent, 
                                            self.print_Node(node.target), 
                                            self.print_Node(node.iterator))
        #if isinstance(node, Nodes.AsyncForStatNode):
        #    result += 
        else:
            result += self.print_UnknownNode(node)
        self.indent()
        result += self.print_Node(node.body)
        self.unindent()
        return result

    def print_ForFromStatNode(self, node):
        result = ""
        return result
#==============================================================================    
    def print_ExprNode(self, node): # call subclasses
        result = ""
        if isinstance(node, ExprNodes.AtomicExprNode):
            result +=      self.print_AtomicExprNode(node)
        #elif isinstance(node, ExprNodes.BackquoteNode):
        #    result +=        self.print_BackquoteNode(node)
        elif isinstance(node, ExprNodes.ImportNode):
            result +=      self.print_ImportNode(node)
        elif isinstance(node, ExprNodes.IteratorNode):
            result +=        self.print_IteratorNode(node)
        #elif isinstance(node, ExprNodes.AsyncIteratorNode):
        #    result +=        self.print_AsyncIteratorNode(node)
        #elif isinstance(node, ExprNodes.WithExitCallNode):
        #    result +=        self.print_WithExitCallNode(node)
        #elif isinstance(node, ExprNodes.TempNode):
        #    result +=        self.print_TempNode(node)
        #elif isinstance(node, ExprNodes.RawCNameExprNode):
        #    result +=        self.print_RawCNameExprNode(node)
        #elif isinstance(node, ExprNodes.JoinedStrNode):
        #    result +=        self.print_JoinedStrNode(node)
        #elif isinstance(node, ExprNodes.FormattedValueNode):
        #    result +=        self.print_FormattedValueNode(node)
        #elif isinstance(node, ExprNodes._IndexingBaseNode):
        #    result +=        self.print__IndexingBaseNode(node)
        #elif isinstance(node, ExprNodes.MemoryCopyNode):
        #    result +=        self.print_MemoryCopyNode(node)
        #elif isinstance(node, ExprNodes.SliceIndexNode):
        #    result +=        self.print_SliceIndexNode(node)
        #elif isinstance(node, ExprNodes.SliceNode):
        #    result +=        self.print_SliceNode(node)
        elif isinstance(node, ExprNodes.CallNode):
            result +=        self.print_CallNode(node)
        #elif isinstance(node, ExprNodes.NumPyMethodCallNode):
        #    result +=        self.print_NumPyMethodCallNode(node)
        #elif isinstance(node, ExprNodes.PythonCapiFunctionNode):
        #    result +=        self.print_PythonCapiFunctionNode(node)
        #elif isinstance(node, ExprNodes.AsTupleNode):
        #    result +=        self.print_AsTupleNode(node)
        #elif isinstance(node, ExprNodes.MergedDictNode):
        #    result +=        self.print_MergedDictNode(node)
        #elif isinstance(node, ExprNodes.AttributeNode):
        #    result +=        self.print_AttributeNode(node)
        #elif isinstance(node, ExprNodes.StarredUnpackingNode):
        #    result +=        self.print_StarredUnpackingNode(node)
        elif isinstance(node, ExprNodes.SequenceNode):
            result +=        self.print_SequenceNode(node)
        #elif isinstance(node, ExprNodes.ScopedExprNode):
        #    result +=        self.print_ScopedExprNode(node)
        #elif isinstance(node, ExprNodes.InlinedGeneratorExpressionNode):
        #    result +=        self.print_InlinedGeneratorExpressionNode(node)
        #elif isinstance(node, ExprNodes.MergedSequenceNode):
        #    result +=        self.print_MergedSequenceNode(node)
        #elif isinstance(node, ExprNodes.SetNode):
        #    result +=        self.print_SetNode(node)
        elif isinstance(node, ExprNodes.DictNode):
            result +=        self.print_DictNode(node)
        elif isinstance(node, ExprNodes.DictItemNode):
            result +=        self.print_DictItemNode(node)
        #elif isinstance(node, ExprNodes.SortedDictKeysNode):
        #    result +=        self.print_SortedDictKeysNode(node)
        #elif isinstance(node, ExprNodes.Py3ClassNode):
        #    result +=        self.print_Py3ClassNode(node)
        #elif isinstance(node, ExprNodes.PyClassMetaclassNode):
        #    result +=        self.print_PyClassMetaclassNode(node)
        #elif isinstance(node, ExprNodes.ClassCellInjectorNode):
        #    result +=        self.print_ClassCellInjectorNode(node)
        #elif isinstance(node, ExprNodes.ClassCellNode):
        #    result +=        self.print_ClassCellNode(node)
        #elif isinstance(node, ExprNodes.CodeObjectNode):
        #    result +=        self.print_CodeObjectNode(node)
        #elif isinstance(node, ExprNodes.DefaultLiteralArgNode):
        #    result +=        self.print_DefaultLiteralArgNode(node)
        #elif isinstance(node, ExprNodes.DefaultNonLiteralArgNode):
        #    result +=        self.print_DefaultNonLiteralArgNode(node)
        #elif isinstance(node, ExprNodes.YieldExprNode):
        #    result +=        self.print_YieldExprNode(node)
        #elif isinstance(node, ExprNodes.UnopNode):
        #    result +=        self.print_UnopNode(node)
        #elif isinstance(node, ExprNodes.TypecastNode):
        #    result +=        self.print_TypecastNode(node)
        #elif isinstance(node, ExprNodes.CythonArrayNode):
        #    result +=        self.print_CythonArrayNode(node)
        #elif isinstance(node, ExprNodes.SizeofNode):
        #    result +=        self.print_SizeofNode(node)
        #elif isinstance(node, ExprNodes.TypeidNode):
        #    result +=        self.print_TypeidNode(node)
        #elif isinstance(node, ExprNodes.TypeofNode):
        #    result +=        self.print_TypeofNode(node)
        elif isinstance(node, ExprNodes.BinopNode):
            result +=        self.print_BinopNode(node)
        #elif isinstance(node, ExprNodes.BoolBinopNode):
        #    result +=        self.print_BoolBinopNode(node)
        #elif isinstance(node, ExprNodes.BoolBinopResultNode):
        #    result +=        self.print_BoolBinopResultNode(node)
        #elif isinstance(node, ExprNodes.CondExprNode):
        #    result +=        self.print_CondExprNode(node)
        #elif isinstance(node, ExprNodes.CoercionNode):
        #    result +=        self.print_CoercionNode(node)
        #elif isinstance(node, ExprNodes.ModuleRefNode):
        #    result +=        self.print_ModuleRefNode(node)
        #elif isinstance(node, ExprNodes.DocstringRefNode):
        #    result +=        self.print_DocstringRefNode(node)
        else:
            result += self.print_UnknownNode(node)
        return result
        
    def print_AtomicExprNode(self, node):
        result = ""
        if isinstance(node, ExprNodes.PyConstNode):
            result += '"%s"' % node.value
        elif isinstance(node, ExprNodes.ConstNode):
            if isinstance(node, ExprNodes.CharNode) or \
               isinstance(node, ExprNodes.UnicodeNode):
                result += '"%s"' % node.value
            else:
                result += "%s" % node.value
            
        #elif isinstance(node, ExprNodes.ImagNode):
        #    result += 
        #elif isinstance(node, ExprNodes.NewExprNode):
        #    result += 
        elif isinstance(node, ExprNodes.NameNode):
            result += "%s" % node.name 
        #elif isinstance(node, ExprNodes.NextNode):
        #    result += 
        #elif isinstance(node, ExprNodes.AsyncNextNode):
        #    result += 
        #elif isinstance(node, ExprNodes.ExcValueNode):
        #    result += 
        #elif isinstance(node, ExprNodes.ParallelThreadsAvailableNode):
        #    result += 
        #elif isinstance(node, ExprNodes.ParallelThreadIdNode):
        #    result += 
        #elif isinstance(node, ExprNodes.GlobalsExprNode):
        #    result += 
        #elif isinstance(node, ExprNodes.PyClassLocalsExprNode):
        #    result += 
        #elif isinstance(node, ExprNodes.TempRefNode):
        #    result += 
        #elif isinstance(node, ExprNodes.ResultRefNode):
        #    result += 
        else:
            result += self.print_UnknownNode(node)
        
        '''if hasattr(node, "name"):
            result = "%s" % node.name
        elif isinstance(node, ExprNodes.UnicodeNode):
            result = '"%s"' % node.value
        elif hasattr(node, "value"):
            result = "%s" % node.value
        else:
            result = "%s" % type(node)'''
        return result

    def print_BackquoteNode(self, node):
        result = ""
        return result

    def print_ImportNode(self, node):
        result = "__import__(%s, globals(), None, %s, %s)" % (self.print_Node(node.module_name),
                                                              self.print_Node(node.name_list),
                                                              node.level)
        return result

    def print_IteratorNode(self, node):
        result = "iter(%s)" % self.print_Node(node.sequence)
        return result

    def print_AsyncIteratorNode(self, node):
        result = ""
        return result

    def print_WithExitCallNode(self, node):
        result = ""
        return result

    def print_TempNode(self, node):
        result = ""
        return result

    def print_RawCNameExprNode(self, node):
        result = ""
        return result

    def print_JoinedStrNode(self, node):
        result = ""
        return result

    def print_FormattedValueNode(self, node):
        result = ""
        return result

    def print__IndexingBaseNode(self, node):
        result = ""
        return result

    def print_MemoryCopyNode(self, node):
        result = ""
        return result

    def print_SliceIndexNode(self, node):
        result = ""
        return result

    def print_SliceNode(self, node):
        result = ""
        return result

    def print_CallNode(self, node):
        result = ""
        if isinstance(node, ExprNodes.SimpleCallNode):
            arguments = []
            for arg in node.args:
                arguments.append(self.print_Node(arg))
            result = "%s(%s)" % (self.print_Node(node.function),
                                 ", ".join(arguments))
        #elif isinstance(node, ExprNodes.InlinedDefNodeCallNode):
        #    result += ""
        elif isinstance(node, ExprNodes.GeneralCallNode):
            result += self.print_GeneralCallNode(node)
        else:
            result += self.print_UnknownNode(node)
        return result

    def print_GeneralCallNode(self, node):
        arguments = []
        for arg in node.positional_args.args:
            arguments.append(self.print_Node(arg))
        for arg in node.keyword_args.key_value_pairs:
            arguments.append("%s = %s" % (arg.key.value,
                                          self.print_Node(arg.value)))
        
        result = "%s(%s)" % (self.print_Node(node.function),
                             ", ".join(arguments))
        return result

    def print_NumPyMethodCallNode(self, node):
        result = ""
        return result

    def print_PythonCapiFunctionNode(self, node):
        result = ""
        return result

    def print_AsTupleNode(self, node):
        result = ""
        return result

    def print_MergedDictNode(self, node):
        result = ""
        return result

    def print_AttributeNode(self, node):
        result = ""
        return result

    def print_StarredUnpackingNode(self, node):
        result = ""
        return result

    def print_SequenceNode(self, node):
        arguments = []
        for arg in node.args:
            arguments.append(self.print_Node(arg))
        if isinstance(node, ExprNodes.ListNode):
            result = "[%s]" % (", ".join(arguments))
        else: # tuple
            result = "(%s)" % (", ".join(arguments))
        return result

    def print_ScopedExprNode(self, node):
        result = ""
        return result

    def print_InlinedGeneratorExpressionNode(self, node):
        result = ""
        return result

    def print_MergedSequenceNode(self, node):
        result = ""
        return result

    def print_SetNode(self, node):
        result = ""
        return result

    def print_DictNode(self, node):
        arguments = []
        for dict_pair in node.key_value_pairs:
            arguments.append(self.print_Node(dict_pair))
        result = "{%s}" % ", ".join(arguments)
        return result

    def print_DictItemNode(self, node):
        result = "%s: %s" % (self.print_Node(node.key),
                             self.print_Node(node.value))
        return result

    def print_SortedDictKeysNode(self, node):
        result = ""
        return result

    def print_Py3ClassNode(self, node):
        result = ""
        return result

    def print_PyClassMetaclassNode(self, node):
        result = ""
        return result

    def print_ClassCellInjectorNode(self, node):
        result = ""
        return result

    def print_ClassCellNode(self, node):
        result = ""
        return result

    def print_CodeObjectNode(self, node):
        result = ""
        return result

    def print_DefaultLiteralArgNode(self, node):
        result = ""
        return result

    def print_DefaultNonLiteralArgNode(self, node):
        result = ""
        return result

    def print_YieldExprNode(self, node):
        result = ""
        return result

    def print_UnopNode(self, node):
        result = ""
        return result

    def print_TypecastNode(self, node):
        result = ""
        return result

    def print_CythonArrayNode(self, node):
        result = ""
        return result

    def print_SizeofNode(self, node):
        result = ""
        return result

    def print_TypeidNode(self, node):
        result = ""
        return result

    def print_TypeofNode(self, node):
        result = ""
        return result

    def print_BinopNode(self, node):
        result = "%s %s %s" % (self.print_Node(node.operand1), 
                               node.operator, 
                               self.print_Node(node.operand2))
        return result

    def print_BoolBinopNode(self, node):
        result = ""
        return result

    def print_BoolBinopResultNode(self, node):
        result = ""
        return result

    def print_CondExprNode(self, node):
        result = ""
        return result

    def print_CoercionNode(self, node):
        result = ""
        return result

    def print_ModuleRefNode(self, node):
        result = ""
        return result

    def print_DocstringRefNode(self, node):
        result = ""
        return result
#==============================================================================
    def print_ComprehensionAppendNode(self, node): # call subclasses?
        result = "<ComprehensionAppendNode not made>\n"
        return result
#==============================================================================
    def print_TempsBlockNode(self, node): # no subclasses
        result = "<TempsBlockNode not made>\n"
        return result
#==============================================================================
    def print_CompilerDirectivesNode(self, node): # no subclasses
        result = "<CompilerDirectivesNode>\n"
        return result
#==============================================================================
    def print_StatListNode(self, node):
        result = ""
        for stat in node.stats:
            result += self.print_Node(stat)
        if not node.stats:
            result += "%spass" % self._indent
        return result
#==============================================================================
    def print_StatNode(self, node): # call subclasses
        result = ""
        if    isinstance(node, Nodes.CDefExternNode):
            result +=     self.print_CDefExternNode(node)
        #elif isinstance(node, Nodes.CVarDefNode):
        #    result +=    self.print_CVarDefNode(node)
        #elif isinstance(node, Nodes.CStructOrUnionDefNode):
        #    result +=    self.print_CStructOrUnionDefNode(node)
        #elif isinstance(node, Nodes.CEnumDefNode):
        #    result +=    self.print_CEnumDefNode(node)
        #elif isinstance(node, Nodes.CEnumDefItemNode):
        #    result +=    self.print_CEnumDefItemNode(node)
        #elif isinstance(node, Nodes.CTypeDefNode):
        #    result +=    self.print_CTypeDefNode(node)
        #elif isinstance(node, Nodes.OverrideCheckNode):
        #    result +=    self.print_OverrideCheckNode(node)
        #elif isinstance(node, Nodes.PropertyNode):
        #    result +=    self.print_PropertyNode(node)
        #elif isinstance(node, Nodes.GlobalNode):
        #    result +=    self.print_GlobalNode(node)
        #elif isinstance(node, Nodes.NonlocalNode):
        #    result +=    self.print_NonlocalNode(node)
        elif isinstance(node, Nodes.ExprStatNode):
            result +=    self.print_ExprStatNode(node)
        elif isinstance(node, Nodes.AssignmentNode):
            result +=    self.print_AssignmentNode(node)
        #elif isinstance(node, Nodes.PrintStatNode):
        #    result +=    self.print_PrintStatNode(node)
        #elif isinstance(node, Nodes.ExecStatNode):
        #    result +=    self.print_ExecStatNode(node)
        #elif isinstance(node, Nodes.DelStatNode):
        #    result +=    self.print_DelStatNode(node)
        #elif isinstance(node, Nodes.PassStatNode):
        #    result +=    self.print_PassStatNode(node)
        #elif isinstance(node, Nodes.BreakStatNode):
        #    result +=    self.print_BreakStatNode(node)
        #elif isinstance(node, Nodes.ContinueStatNode):
        #    result +=    self.print_ContinueStatNode(node)
        elif isinstance(node, Nodes.ReturnStatNode):
            result +=    self.print_ReturnStatNode(node)
        #elif isinstance(node, Nodes.RaiseStatNode):
        #    result +=    self.print_RaiseStatNode(node)
        #elif isinstance(node, Nodes.ReraiseStatNode):
        #    result +=    self.print_ReraiseStatNode(node)
        #elif isinstance(node, Nodes.AssertStatNode):
        #    result +=    self.print_AssertStatNode(node)
        elif isinstance(node, Nodes.IfStatNode):
            result +=    self.print_IfStatNode(node)
        #elif isinstance(node, Nodes.SwitchCaseNode):
        #    result +=    self.print_SwitchCaseNode(node)
        #elif isinstance(node, Nodes.SwitchStatNode):
        #    result +=    self.print_SwitchStatNode(node)
        elif isinstance(node, Nodes.WithStatNode):
            result +=    self.print_WithStatNode(node)
        #elif isinstance(node, Nodes.TryExceptStatNode):
        #    result +=    self.print_TryExceptStatNode(node)
        #elif isinstance(node, Nodes.TryFinallyStatNode):
        #    result +=    self.print_TryFinallyStatNode(node)
        #elif isinstance(node, Nodes.GILExitNode):
        #    result +=    self.print_GILExitNode(node)
        elif isinstance(node, Nodes.CImportStatNode):
            result +=    self.print_CImportStatNode(node)
        elif isinstance(node, Nodes.FromCImportStatNode):
            result +=    self.print_FromCImportStatNode(node)
        elif isinstance(node, Nodes.FromImportStatNode):
            result +=    self.print_FromImportStatNode(node)
        #elif isinstance(node, Nodes.CnameDecoratorNode):
        #    result +=    self.print_CnameDecoratorNode(node)
        elif isinstance(node, Nodes.CFuncDefNode):
            result +=    self.print_CFuncDefNode(node)
        else:
            result += self.print_UnknownNode(node)
        return result

    def print_CDefExternNode(self, node):
        result = ""
        return result

    def print_CVarDefNode(self, node):
        result = ""
        return result

    def print_CStructOrUnionDefNode(self, node):
        result = ""
        return result

    def print_CEnumDefNode(self, node):
        result = ""
        return result

    def print_CEnumDefItemNode(self, node):
        result = ""
        return result

    def print_CTypeDefNode(self, node):
        result = ""
        return result

    def print_OverrideCheckNode(self, node):
        result = ""
        return result

    def print_PropertyNode(self, node):
        result = ""
        return result

    def print_GlobalNode(self, node):
        result = ""
        return result

    def print_NonlocalNode(self, node):
        result = ""
        return result

    def print_ExprStatNode(self, node):
        result = "%s%s\n" % (self._indent, 
                             self.print_Node(node.expr))
        return result

    def print_AssignmentNode(self, node):
        result = ""
        if isinstance(node, Nodes.SingleAssignmentNode):
            result += "%s%s = %s\n" % (self._indent,
                                       self.print_Node(node.lhs),
                                       self.print_Node(node.rhs))
        elif isinstance(node, Nodes.CascadedAssignmentNode):
            arguments = []
            for lhs in node.lhs_list:
                arguments.append(self.print_Node(lhs))
            result += "%s%s = %s\n" % (self._indent,
                                       " = ".join(arguments),
                                       self.print_Node(node.rhs))
        elif isinstance(node, Nodes.ParallelAssignmentNode):
            for stat in node.stats:
                result += self.print_Node(stat)
        elif isinstance(node, Nodes.InPlaceAssignmentNode):
            result += "%s%s %s= %s\n" % (self._indent,
                                         self.print_Node(node.lhs), 
                                         node.operator, 
                                         self.print_Node(node.rhs))
        #elif isinstance(node, Nodes.WithTargetAssignmentStatNode):
        #    result += ""
        else:
            result += self.print_UnknownNode(node)
        return result

    def print_PrintStatNode(self, node):
        result = ""
        return result

    def print_ExecStatNode(self, node):
        result = ""
        return result

    def print_DelStatNode(self, node):
        result = ""
        return result

    def print_PassStatNode(self, node):
        result = ""
        return result

    def print_BreakStatNode(self, node):
        result = ""
        return result

    def print_ContinueStatNode(self, node):
        result = ""
        return result

    def print_ReturnStatNode(self, node):
        result = "%sreturn %s\n" % (self._indent, self.print_Node(node.value))
        return result

    def print_RaiseStatNode(self, node):
        result = ""
        return result

    def print_ReraiseStatNode(self, node):
        result = ""
        return result

    def print_AssertStatNode(self, node):
        result = ""
        return result

    def print_IfStatNode(self, node):
        arguments = []
        for if_clause in node.if_clauses:
            arguments.append(self.print_Node(if_clause))
        if node.else_clause:
            result = "%sif %s\n" % (self._indent, 
                                    ("%selif " % self._indent).join(arguments))
            result += "%selse:\n" % self._indent
            self.indent()
            result += self.print_Node(node.else_clause)
            self.unindent()
        else:
            result = "%sif %s:%s" % (self._indent, 
                                     " , ".join(arguments))        
        return result

    def print_SwitchCaseNode(self, node):
        result = ""
        return result

    def print_SwitchStatNode(self, node):
        result = ""
        return result

    def print_WithStatNode(self, node):
        result = "%swith %s as %s:\n" % (self._indent,
                                              self.print_Node(node.manager), 
                                              self.print_Node(node.target))
        self.indent()
        result += "%s" % self.print_Node(node.body.body.body.stats[1])
        self.unindent() 
        return result

    def print_TryExceptStatNode(self, node):
        result = ""
        return result

    def print_TryFinallyStatNode(self, node):
        result = ""
        return result

    def print_GILExitNode(self, node):
        result = ""
        return result

    def print_CImportStatNode(self, node):
        if node.as_name:
            result = "import %s as %s\n" % (node.module_name, 
                                            node.as_name)
        else:
            result = "import %s\n" % (node.module_name)
        return result

    def print_FromCImportStatNode(self, node):
        for argument in node.imported_names:
            if argument[2]:
                result = "from %s import %s as %s\n" % (node.module_name, 
                                                        argument[1],
                                                        argument[2])
            else:
                result = "from %s import %s\n" % (node.module_name, 
                                                  argument[1])
        return result

    def print_FromImportStatNode(self, node):
        result = "%s\n" % self.print_Node(node.module)
        return result

    def print_CnameDecoratorNode(self, node):
        result = ""
        return result
        
    def print_CFuncDefNode(self, node):
        result = "%sdef %s\n" % (self._indent, 
                                 self.print_Node(node.declarator))
        self.indent()
        result += self.print_Node(node.body)
        self.unindent()
        return result
#==============================================================================
    def print_CDeclaratorNode(self, node): # call subclasses
        result = ""
        if   isinstance(node, Nodes.CNameDeclaratorNode):
            result +=    self.print_CNameDeclaratorNode(node)
        #elif isinstance(node, Nodes.CPtrDeclaratorNode):
        #    result +=    self.print_CPtrDeclaratorNode(node)
        #elif isinstance(node, Nodes.CReferenceDeclaratorNode):
        #    result +=    self.print_CReferenceDeclaratorNode(node)
        #elif isinstance(node, Nodes.CArrayDeclaratorNode):
        #    result +=    self.print_CArrayDeclaratorNode(node)
        elif isinstance(node, Nodes.CFuncDeclaratorNode):
            result +=    self.print_CFuncDeclaratorNode(node)
        #elif isinstance(node, Nodes.CConstDeclaratorNode):
        #    result +=    self.print_CConstDeclaratorNode(node)
        else:
            result += self.print_UnknownNode(node)
        return result

    def print_CNameDeclaratorNode(self, node):
        result = node.name
        return result
        
    def print_CPtrDeclaratorNode(self, node):
        result = ""
        return result
        
    def print_CReferenceDeclaratorNode(self, node):
        result = ""
        return result
        
    def print_CArrayDeclaratorNode(self, node):
        result = ""
        return result
 
    def print_CFuncDeclaratorNode(self, node):
        arguments = []
        for arg in node.args:
            arguments.append(self.print_Node(arg))
        result = "%s(%s):" % (self.print_Node(node.base), 
                              ", ".join(arguments))
        return result
        
    def print_CConstDeclaratorNode(self, node):
        result = ""
        return result
#==============================================================================
    def print_CArgDeclNode(self, node):
        if node.default:
            result = "%s = %s" % (self.print_Node(node.declarator),
                                  self.print_Node(node.default))
        else:
            result = "%s" % self.print_Node(node.declarator)
        return result
#==============================================================================
    def print_CBaseTypeNode(self, node): # call subclasses
        result = "<CBaseTypeNode not made>\n"
        return result
#==============================================================================
    def print_CAnalysedBaseTypeNode(self, node): # no subclasses
        result = "<CAnalysedBaseTypeNode not made>\n" # not used on our phase?
        return result
#==============================================================================
    def print_PyArgDeclNode(self, node): # no subclasses
        result = "<PyArgDeclNode not made>\n"
        return result
#==============================================================================
    def print_DecoratorNode(self, node): # no subclasses
        result = "<DecoratorNode not made>\n"
        return result
#==============================================================================
    def print_IfClauseNode(self, node): # no subclasses
        result = "%s:\n" % (self.print_Node(node.condition))
        self.indent()
        result += self.print_Node(node.body)
        self.unindent()
        return result
#==============================================================================
    def print_DictIterationNextNode(self, node): # no subclasses
        result = "<DictIterationNextNode not made>\n"
        return result
#==============================================================================
    def print_SetIterationNextNode(self, node): # no subclasses
        result = "<SetIterationNextNode not made>\n"
        return result
#==============================================================================
    def print_ExceptClauseNode(self, node): # no subclasses
        result = "<ExceptClauseNode not made>\n"
        return result
#==============================================================================
    def print_ParallelNode(self, node): # no subclasses
        result = "<ParallelNode not made>\n"
        return result
###======================================================================
from operator import attrgetter
class PrintSkipTree(PrintTree):
    _last_pos = 0
    _positions = []
    _text = []
    """ Makes .py code out of AST """

    class Position():
        def __init__(self, line, pos, indent, node):
            self.line = line
            self.pos = pos
            self.indent = indent
            self.node = node

    def __call__(self, tree, phase=None):
        print("Skip tree print")

        positions = self.fill_pos(tree)
        positions.sort(key = attrgetter("line", "pos", "indent"))
        positions.append(self.Position(positions[-1].line, -1, '', 0))
        self._positions = positions
        
        path = tree.pos[0].get_description()
        with open(path, 'r') as f:
            self._text = f.readlines()
        
        print("##################CODE FILE START##################")
        print(self.print_Node(tree))
        print("###################CODE FILE END###################")
        
        #for pos in self._positions:
        #    print(pos.line, pos.pos, pos.indent, pos.node)
        return tree

    def fill_pos(self, node):
        # add info about node pos
        positions = []
        if node is None: return None
        if node.pos and isinstance(node, Nodes.StatNode):
                pos = node.pos
                path = pos[0].get_description()
                if '/' in path:
                    path = path.split('/')[-1]
                if '\\' in path:
                    path = path.split('\\')[-1]
                positions.append(self.Position(pos[1] - 1, 0, self._indent, node))
        
        # add info about children pos
        self.indent()
        for attr in node.child_attrs:
            children = getattr(node, attr)
            if children is not None:
                if type(children) is list:
                    for child in children:
                        positions.extend(self.fill_pos(child))
                else:
                    positions.extend(self.fill_pos(children))
        self.unindent()
        
        return positions
        
    def print_Node(self, node):
        if node is None: return None
        result = ""
        
        if isinstance(node, Nodes.StatNode):
            if self.check_IfNotC(node):
                result += self.print_ByPos(node)
            else:
                result += self.print_CNode(node)
        
        for attr in node.child_attrs:
            children = getattr(node, attr)
            if children is not None:
                if type(children) is list:
                    for child in children:
                        result += self.print_Node(child)
                else:
                    result += self.print_Node(children)

        return result

    def print_CNode(self, node):
        result = ""
        return result

    def check_IfNotC(self, node):
        if node is None: return True
        
        s_type = str(type(node))
        s_type = s_type[s_type.rfind(".") + 1:-2]
        if s_type[0] == "C" and s_type[0:1] == s_type[0:1].upper():
            return False
            
        for attr in node.child_attrs:
            children = getattr(node, attr)
            if children is not None:
                if type(children) is list:
                    for child in children:
                        if not self.check_IfNotC(child): return False
                else:
                    if not self.check_IfNotC(children): return False
        
        #print(s_type)
        return True

    def get_Pos(self, node):
        length = len(self._positions)
        for i in range(length):
            if self._positions[i].node == node:
                cur_pos  = self._positions[i]
                next_pos = self._positions[i + 1]
                return cur_pos, next_pos
        return 0, 0

    def print_ByPos(self, node):
        cur_pos, next_pos = self.get_Pos(node)
        result = ""
        if cur_pos.line == next_pos.line:
            result += "" + self._text[cur_pos.line][cur_pos.pos:next_pos.pos]
        else:
            result += "" + self._text[cur_pos.line][cur_pos.pos:]
            for i in range(cur_pos.line + 1, next_pos.line):
                result += self._text[i]
            result += self._text[next_pos.line][:next_pos.pos]
        
        return result
""" My Modification end """

if __name__ == "__main__":
    import doctest
    doctest.testmod()
