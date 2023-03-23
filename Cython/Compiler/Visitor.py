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

# MIPT: Add new class for generating python code from pyrex code 
#       based on cython debug function for printing AST
from operator import attrgetter
from re import findall
import os
class PrintSkipTree(PrintTree):
    # MIPT: _positions - code markers, 
    #       _text - original pyrex code
    #       _structs - list of struct names from code
    #       _python_dir - python3-dev dir path
    _positions = []
    _text = []
    _structs = ["list", "dict", "tuple"]
    _python_dir = ""
    _source_root = "/usr/include"

    # MIPT: Get python3-dev dir path
    def get_Python_dir(self, path):
        for filename in os.listdir(path):
            if filename.startswith("python3."):
                result = "%s%s/" % (path, filename)
        return result
    
    def get_source(self, filename):
        for rootdir, dirs, files in os.walk(self._source_root):
            for file in files:       
                if file == filename:
                    return "%s/%s" % (rootdir, file)
        return 0
        
    # MIPT: Supporting class for markers to transfer code directly
    class Position():
        def __init__(self, line, pos, indent, node):
            self.line = line
            self.pos = pos
            self.indent = indent
            self.node = node

    def indent(self):
        self._indent += "    "

    def unindent(self):
        self._indent = self._indent[:-4]

    # MIPT: function for generating python code from pyrex source AST
    def __call__(self, tree, phase=None):
        print("# Python code print")
        
        self._python_dir = self.get_Python_dir("/usr/include/")
        
        # get source code file name
        path_in = tree.pos[0].get_description()
        if path_in.endswith(".py"):
            # dont change anything in .py file
            return tree

        # fill code markers
        positions = self.fill_pos(tree)
        positions.sort(key = attrgetter("line", "pos", "indent"))
        positions.append(self.Position(positions[-1].line, -1, '', 0))
        self._positions = positions
        
        try:
            f = open(path_in, 'r')
        except:
            # stdlib from pyrex Includes
            path_in = __file__[:__file__.rfind(".") - 16] + "Includes/" + path_in
            f = open(path_in, 'r')
        
        # get source code
        self._text = f.readlines()
        f.close()
        
        py_code = "import cython\n"
        py_code += self.print_Node(tree)
        
        # get output path name
        path_out = tree.pos[0].path_description
        # change /name.pxd->/m_name.pxd of a .pxd lib 
        # for no intersection with possible py files
        if path_out.endswith(".pxd"):
            path_out = path_out[:path_out.rfind("/") + 1] + "m_"\
                     + path_out[path_out.rfind("/") + 1:]
        path_out = path_out[:-3] + "py"
        
        # create dirs if needed
        if "/" in path_out:
            os.makedirs(os.path.dirname(path_out), exist_ok=True)
        
        with open(path_out, "w") as f:
            f.write(py_code)
            
        print("# Code written in file %s\n" % path_out)
        print(py_code)
        return tree

    # MIPT: fill code markers
    def fill_pos(self, node):
        # add info about node pos
        positions = []
        if node is None: return None
        if node.pos and isinstance(node, Nodes.StatNode) or \
                        isinstance(node, Nodes.StatListNode):
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
    
    # MIPT: Block of main generating(printing) functions
    
    # MIPT: main node print function
    def print_Node(self, node):
        if node is None: return None
        result = ""
        
        # print StatNode via markers if there are no C functions in it
        if isinstance(node, Nodes.StatNode):
            if self.check_IfNotC(node):
                result += self.print_StatByPos(node)
            else:
                result += self.print_CNode(node)
        
        # print specific expressions, when StatListNode is empty - 
        # for ex. comments or pass sections
        elif isinstance(node, Nodes.StatListNode) and not node.stats:
            s_stat = "%s%s\n" % (self._indent, 
                                 self.print_ExprByPos(node))
            if s_stat.count("'''") % 2 == 0: 
                result += s_stat
        # try the same for children nodes
        else:
            for attr in node.child_attrs:
                children = getattr(node, attr)
                if children is not None:
                    if type(children) is list:
                        for child in children:
                            result += self.print_Node(child)
                    else:
                        result += self.print_Node(children) 
        
        return result

    # MIPT: main function for printing node with C constructions
    def print_CNode(self, node):
        result = ""
        if   isinstance(node, Nodes.CDefExternNode):
            result += self.print_CDefExternNode(node)
        elif isinstance(node, Nodes.CDeclaratorNode):
            result += self.print_CDeclaratorNode(node)
        elif isinstance(node, Nodes.CBaseTypeNode):
            result += self.print_CBaseTypeNode(node)
        elif isinstance(node, Nodes.CVarDefNode):
            result += self.print_CVarDefNode(node)
        elif isinstance(node, Nodes.CStructOrUnionDefNode):
            result += self.print_CStructOrUnionDefNode(node)
        elif isinstance(node, Nodes.CEnumDefNode):
            result += self.print_CEnumDefNode(node)
        elif isinstance(node, Nodes.CTypeDefNode):
            result += self.print_CTypeDefNode(node)
        elif isinstance(node, Nodes.CFuncDefNode):
            result += self.print_CFuncDefNode(node)
        elif isinstance(node, Nodes.CClassDefNode):
            result += self.print_CClassDefNode(node)
        elif isinstance(node, Nodes.CImportStatNode):
            result += self.print_CImportStatNode(node)
        elif isinstance(node, Nodes.FromCImportStatNode):
            result += self.print_FromCImportStatNode(node)
        else:
            result += self.print_UnknownNode(node)
        return result
    
    # MIPT: Block of print functions needed for extern statements via ctypes library
    #       (when importing direct C libraries is needed)
    
    # MIPT: printing Extern node
    #       shared object created, then linked with ctypes
    def print_CDefExternNode(self, node):
        result = ""
        c_name = node.include_file
        if not c_name:
            # cdef extern from *:
            result += "%s\n" % self.print_Node(node.body)
            return result
        elif "<" in c_name:
            # C stdlib file extern
            c_name = c_name[1:-1]
            so_name = "%s.so" % (c_name[:c_name.rfind(".")])
            c_name = self.get_source(c_name)
            if not c_name:
                result += "# Couldn't find file %s" % node.include_file
                return result
        elif c_name == "Python.h":
            # Python.h specific case
            so_name = c_name
            c_name = "%sPython.h" % self._python_dir
        else:
            # regular file extern
            so_name = c_name[:c_name.rfind(".")] + ".so"
        
        os.system("cc -fPIC -shared -o %s %s" % (so_name, c_name))

        result += "exported_lib = ctypes.CDLL(%s)\n" % c_name
        for stat in node.body.stats:
            result += self.print_CTypes_Node(stat)
        
        return result

    # MIPT: main function for printing nodes in extern statement 
    #       only CVarDefNode behave differently in extern -> different print
    #       also var and enum declarations are stubs for now because of
    #       no implementation in ctypes
    def print_CTypes_Node(self, node):
        result = ""
        if isinstance(node, Nodes.CVarDefNode):
            result += self.print_CTypes_VarDefNode(node)
        elif isinstance(node, Nodes.CStructOrUnionDefNode):
            result += self.print_CTypes_StructOrUnionDefNode(node)
        elif isinstance(node, Nodes.CEnumDefNode):
            result += "# no implementation for enum in ctypes\n"
        else:
            result += self.print_CNode(node)
        return result

    # MIPT: prints struct or union constructions in extern statement
    def print_CTypes_StructOrUnionDefNode(self, node):
        arguments = []
        
        result = "# cython.%s\n" % (node.kind)
        result += "class %s(ctypes.Structure): pass\n" % (node.name)        
        self.indent()
        if node.attributes:
            for arg in node.attributes:
                type = self.print_Ctypes_FullType(arg)
                name = self.print_CVarDefNode(arg)
                name = name[:name.find(":")].strip()
                arguments.append('("%s", %s)' % (name, type))

        result += "%s._fields_ = [%s]\n\n" % (node.name,
                                             ", ".join(arguments))
        self.unindent()       
                                      
        self._structs.append(node.name)
        return result

    # MIPT: CVarDefNode print in extern statement
    #       need to print function declarations correctly with ctypes shell
    def print_CTypes_VarDefNode(self, node):
        result = ""
        base = node.declarators[0]
        is_func = False
        
        # check whether it is a function declaration
        while hasattr(base, "base"):
            if isinstance(base, Nodes.CFuncDeclaratorNode):
                is_func = True
                break    
            base = base.base
        
        if is_func:
            # base has Nodes.CFuncDeclaratorNode type
            full_type = self.print_Ctypes_FullType(node)
            func_name = base.base.name
            arguments = []
            for arg in base.args:
                arguments.append(self.print_Ctypes_FullType(arg))
                
            # print ctypes shell
            result += "exported_lib.%s.restype = %s\n" % (func_name, 
                                                          full_type)
            result += "exported_lib.%s.argtypes = [%s]\n" % (func_name, 
                                                             ", ". join(arguments))
            result += "%s = exported_lib.%s\n\n" % (func_name, func_name)
        else:
            result += self.print_CVarDefNode(node)
        return result
   
    # MIPT: function for correct variable type print in CTypes style
    def print_Ctypes_FullType(self, node):
        s_type = self.print_CBaseTypeNode(node.base_type, ctypes = True)
        if hasattr(node, "declarator"):
            base = node.declarator
        else:
            base = node.declarators[0]
        full_type = self.print_TypeTree(base, ctypes = True) % s_type
        return full_type

    # MIPT: Block of print functions for other kinds of C nodes
   
    # MIPT: base function for printing different declarations 
    #       as CDeclarator nodes chains
    #       see print_CFuncDeclaratorNode() to understand from_cvardef
    def print_CDeclaratorNode(self, node, s_type = "", from_cvardef = False):
        result = ""
        
        # analise whether there are no more c expressions, print it if true 
        if self.check_IfNotCChildren(node):
            s_expr = self.print_ExprByPos(node).strip()
            if "=" in s_expr: # initialisation
                result += "%s" % s_expr
            elif "[" in s_expr: # list declaration
                name, size = s_expr.split("[")
                if size[:-1]:
                    result += "%s = [None] * %s" % (name, size[:-1])
                else:
                    result += "%s" % name
            elif s_type: # declaration with annotation
                if s_type == s_expr:
                    result += "%s" % (s_expr)
                elif s_type in self._structs:
                    result += "%s = %s()" % (s_expr, s_type)
                else:
                    result += "%s : %s" % (s_expr, s_type)
            else: # simple declaration
                if   '"' in s_expr:
                    s_expr = s_expr[:s_expr.find('"')]
                elif "'" in s_expr:
                    s_expr = s_expr[:s_expr.find("'")]
                result += "%s" % s_expr
        
        # continue the chain otherwise
        elif isinstance(node, Nodes.CNameDeclaratorNode):
            result += "%s" % node.name
        elif isinstance(node, Nodes.CFuncDeclaratorNode):
            result += "%s" % self.print_CFuncDeclaratorNode(node, from_cvardef)   
        else:
            # CPtrDeclaratorNode
            # CReferenceDeclaratorNode
            # CArrayDeclaratorNode
            # CConstDeclaratorNode
            result += "%s" % self.print_CDeclaratorNode(node.base, s_type, from_cvardef)
                   
        return result
    
    # MIPT: prints function declarator with cython decorator for functions
    #       note: only function call, without function body
    #
    #       can be called by 2 possible parents: (see from_funcdef var)
    #       1) CFuncDefNode - cdef func(arguments):\n <body from sibling>
    #       2) CVarDefNode  - cdef func(arguments) <no body, just declaration>
    #       2nd version can be seen in 2.1) function/class declarations or in 
    #       2.2) extern statements, see print_CTypes_VarDefNode()
    def print_CFuncDeclaratorNode(self, node, from_cvardef = False):
        # in declarator chain get function name
        base = node.base
        while not hasattr(base, "name"):
            base = base.base
        
        if from_cvardef:
            # just declaration
            result = "%s : function" % (base.name)
        else:
            # used in function definition
            arguments = []
            for arg in node.args:
                arguments.append(self.print_CArgDeclNode(arg))
            
            result = "@cython.cfunc\n"
            if node.exception_value:
                result += "%s@cython.exceptval(%s)\n" % (self._indent, 
                                                         node.exception_value.value)
                
            result += "%sdef %s(%s)" % (self._indent,
                                        base.name,
                                        ", ".join(arguments))
        return result
    
    # MIPT: prints basic argument declaration: type                                
    def print_CArgDeclNode(self, node):  
        s_type = self.print_CBaseTypeNode(node.base_type)
        full_type = self.print_TypeTree(node.declarator) % s_type
        result = "%s" % (self.print_CDeclaratorNode(node.declarator, full_type))
        return result

    # MIPT: needed for type print, flag ctypes shows the way: cython or ctypes
    def print_CBaseTypeNode(self, node, ctypes = False):
        if   isinstance(node, Nodes.CSimpleBaseTypeNode):
            result = self.print_CSimpleBaseTypeNode(node, ctypes)
        elif isinstance(node, Nodes.CConstTypeNode):
            result = self.print_CBaseTypeNode(node.base_type, ctypes)
        else:
            result = self.print_UnknownNode(node)   
        return result

    # MIPT: prints type in a cython or ctypes way
    def print_CSimpleBaseTypeNode(self, node, ctypes = False):
        result = ""
        # prints library path if any
        if node.is_basic_c_type:
            if ctypes:
                result += "ctypes."
            else:
                result += "cython."
        for path in node.module_path:
            result += "%s." % path
            
        double_longness = ["double", "longdouble"]
        int_longness = ["int", "long", "longlong", "short"]
            
        # finalise type for special cases - unsigned or long ones
        s_type = node.name
        if s_type == "int":
            s_type = int_longness[node.longness]
        elif s_type == "double":
            s_type = double_longness[node.longness]
        if not node.signed and "size_t" not in s_type:
            s_type = "u" + s_type
            
        result += "%s" % s_type
        return result

    # MIPT: prints variable declration with base value or no
    #       assignments are printed without type, and others with type
    def print_CVarDefNode(self, node):
        # node.visibility: public, _protected, __private__ - not used
        s_type = self.print_CBaseTypeNode(node.base_type)
        
        result = self.print_Decorators(node)
        for declarator in node.declarators:
            full_type = self.print_TypeTree(declarator) % s_type
            s_stat = "%s%s\n" % (self._indent,
                                 self.print_CDeclaratorNode(declarator, full_type, from_cvardef = True))
            result += s_stat

        return result

    # MIPT: prints struct or union constructions as python classes
    def print_CStructOrUnionDefNode(self, node):
        arguments = []
        
        result = "# cython.%s\n" % (node.kind)
        result += "class %s():" % (node.name)
        self.indent()
        if node.attributes:
            for arg in node.attributes:
                arguments.append(self.print_CVarDefNode(arg)[:-1])
            result += "%s\n" % ("\n".join(arguments))
        else:
            result += " pass\n\n"
        self.unindent()

        self._structs.append(node.name)
        return result

    # MIPT: prints enum construction as python class, assigns values when needed
    def print_CEnumDefNode(self, node):
        self.indent()
        arguments = []
        # needed for correct value assignment cause no enum in python or pure cython
        for (i, item) in enumerate(node.items):
            s_item = self.print_CEnumDefItemNode(item)
            if "=" not in s_item:
                if i == 0:
                    s_item += " = 0"
                else:
                    s_prev_item = arguments[i - 1]
                    s_item += " %s + 1" % s_prev_item[s_prev_item.find("="):]
            arguments.append(s_item)
        self.unindent()
        result = "class %s():\n%s\n\n" % (node.name,
                                          "\n".join(arguments))
        return result

    # MIPT: prints one enum variable declaration
    def print_CEnumDefItemNode(self, node):
        result = "%s%s" % (self._indent,
                           self.print_ExprByPos(node))
        return result

    # MIPT: prints typedef construction in a way: new type = base type
    def print_CTypeDefNode(self, node):
        s_type = self.print_CBaseTypeNode(node.base_type)
        result = "%s%s = %s\n" % (self._indent,
                                  self.print_CDeclaratorNode(node.declarator),
                                  self.print_TypeTree(node.declarator) % s_type)
        return result

    # MIPT: prints full function definition:
    #       def f() -> type:
    #           body
    def print_CFuncDefNode(self, node):
        s_type = self.print_CBaseTypeNode(node.base_type) 
        
        result = self.print_Decorators(node)
        result += "%s%s -> %s:\n" %(self._indent,
                                    self.print_CDeclaratorNode(node.declarator, s_type),
                                    self.print_TypeTree(node.declarator) % s_type)
        self.indent()
        result += "%s\n" % (self.print_Node(node.body))
        self.unindent()
        
        return result

    # MIPT: prints class with cython decorator for classes
    def print_CClassDefNode(self, node):
        arguments = []
        for base in node.bases.args:
            arguments.append(base.name)
            
        result  = "%s@cython.cclass\n" % self._indent
        result += self.print_Decorators(node)
        result += "%sclass %s(%s):\n" % (self._indent,
                                        node.class_name,
                                        ", ".join(arguments))
        self.indent()
        result += "%s" % (self.print_Node(node.body))
        self.unindent()
        return result

    # MIPT: prints decorators taking them from source code
    def print_Decorators(self, node):
        decorators = []
        if node.decorators:
            for decorator in node.decorators:
                decorators.append(self.print_ExprByPos(decorator))
        if decorators:
            return "%s\n" % (("%s\n" % self._indent).join(decorators))
        else:
            return ""

    # MIPT: prints cimport constructions as followed:
    #       cimport file(.pxd) (as ...)
    def print_CImportStatNode(self, node):
        # change name(.pxd) to m_name(.pxd)
        # to call correct files (because .pxd are changed in __call__)
        module = node.module_name
        module = module[:module.rfind(".") + 1] + "m_" + module[module.rfind(".") + 1:]
        if node.as_name:
            result = "import %s as %s\n" % (module, 
                                            node.as_name)
        else:
            result = "import %s\n" % (module)
        return result

    # MIPT: prints from cimport constructions as followed:
    #       from file(.pxd) cimport functions (as ...)
    def print_FromCImportStatNode(self, node):
        # change name(.pxd) to m_name(.pxd)
        module = node.module_name
        module = "." * node.relative_level + \
                 module[:module.rfind(".") + 1] + "m_" + \
                 module[module.rfind(".") + 1:]
        result = ""
        for argument in node.imported_names:
            if argument[2]:
                result += "from %s import %s as %s\n" % (module, 
                                                        argument[1],
                                                        argument[2])
            else:
                result += "from %s import %s\n" % (module, 
                                                  argument[1])
        return result

    # MIPT: debug printing of possible unprocessed node
    def print_UnknownNode(self, node):
        result = "\n# in %s() found %s\n" % (inspect.stack()[1][3], 
                                             type(node))
        return result

    # MIPT: Block of utility functions

    # MIPT: checks the node and its children for having C-like nodes
    def check_IfNotC(self, node):
        if node is None: return True
        
        s_type = str(type(node))
        s_type = s_type[s_type.rfind(".") + 1:-2]
        if findall('C[A-Z]', s_type):
            return False
            
        return self.check_IfNotCChildren(node)

    # MIPT: checks node children for having C-like nodes
    def check_IfNotCChildren(self, node):
        if node is None: return True

        for attr in node.child_attrs:
            children = getattr(node, attr)
            if children is not None:
                if type(children) is list:
                    for child in children:
                        if not self.check_IfNotC(child): return False
                else:
                    if not self.check_IfNotC(children): return False
        return True

    # MIPT: gets from marker list _positions correct positions of current
    #       and next stat nodes for given stat node
    def get_Pos(self, node):
        for (i, position) in enumerate(self._positions):
            if position.node == node:
                cur_pos  = position
                for (j, position2) in enumerate(self._positions[i + 1:]):
                    if (position2.indent <= position.indent):
                        next_pos = position2
                        return cur_pos, next_pos
        return 0, 0

    # MIPT: transfers statnode from source code
    def print_StatByPos(self, node):
        # get borders of needed source code
        cur_pos, next_pos = self.get_Pos(node)
        result = ""
        if cur_pos.line == next_pos.line:
            result += "" + self._text[cur_pos.line][cur_pos.pos:next_pos.pos]
        else:
            result += "" + self._text[cur_pos.line][cur_pos.pos:]
            for i in range(cur_pos.line + 1, next_pos.line):
                result += self._text[i]
            result += self._text[next_pos.line][:next_pos.pos]
        
        result = result.replace(";", "")
        result = self.improve_Expr(node, result)
        return result
        
    # MIPT: transfers sourse code expression corresponded to given node
    #  def: expression is a part of code, which starts at a node pos or efore it,
    #       ends at the end of a line or ',', and has correct sequence of nested brackets
    def print_ExprByPos(self, node, is_improve = True):
        result = self.print_ExprByPosCore(node)
        if is_improve:
            result = self.improve_Expr(node, result)
        return result
        
    def print_ExprByPosCore(self, node):
        end_sym = [',', '\n']
        continue_sym = [',', '\\']
        brackets_sym = ['[', ']', '(', ')', '<', '>']
        brackets_op_sym = ['[', '(', '<']
        brackets_cl_sym = [']', ')', '>']
        brackets_cnt = [0, 0, 0]
        
        line = node.pos[1] - 1
        pos = node.pos[2] 
        
        # check for incorrect (too late) pos, if so -> go to start of expr
        # neede because for ex. functions pos is set on its argument list starting
        # at (...) but not function name name(...)
        first_char = self._text[line][pos]
        if first_char in  (brackets_sym + end_sym + ["="]):
            # go left till start of expr
            pos -= 1
            while (self._text[line][pos] not in end_sym + brackets_sym):
                pos -= 1
            pos += 1
        
        str_line = self._text[line][pos:]
        
        # for multiline expressions
        while str_line[-1] in continue_sym:
            line += 1
            str_line += self._text[line]
        
        for (index, char) in enumerate(str_line):
            if   char in brackets_op_sym:
                brackets_ind = brackets_op_sym.index(char)
                brackets_cnt[brackets_ind] += 1
            elif char in brackets_cl_sym:
                brackets_ind = brackets_cl_sym.index(char)
                if brackets_cnt[brackets_ind] == 0:
                    result = "%s" % (str_line[:index])
                    return result
                else:
                    brackets_cnt[brackets_ind] -= 1
            elif char in end_sym and brackets_cnt.count(0) == len(brackets_cnt):
                result = "%s" % (str_line[:index])
                return result
        
        result = "%s" % (str_line)
        return result
        
    # MIPT: specific line construction for correct cython/stypes types chain print
    #       used like self.print_TypeTree(node.declarator) % s_type
    #       where s_type is a base type node 
    def print_TypeTree(self, node, ctypes = False):
        result = ""
        
        if isinstance(node, Nodes.CNameDeclaratorNode):
            return "%s"
            
        next_step = self.print_TypeTree(node.base, ctypes)
        if isinstance(node, Nodes.CPtrDeclaratorNode):
            if ctypes:
                result += "ctypes.POINTER(%s)" % next_step
            else:
                result += "cython.pointer(%s)" % next_step
        elif isinstance(node, Nodes.CReferenceDeclaratorNode):
            if ctypes:
                result += "ctypes.addressof(%s)" % next_step
            else:
                result += "cython.address(%s)" % next_step
        elif isinstance(node, Nodes.CArrayDeclaratorNode):
            result += "%s[%s]" % (next_step, node.dimension)
        else:
            result = "%s" % next_step
        
        return result
        
    # MIPT: changes C-like constructions in expressions, see classes in isinstance()
    #       these constructions dont have C-like node type, so changed with this func
    def improve_Expr(self, node, expr):
        if node is None: return True
        line = node.pos[1] - 1
        pos = node.pos[2]
        
        # for each inline c expression 
        # fill line, expression position
        # expression pattern -> changed pattern
        
        # static cast expression <type>(...)
        if isinstance(node, ExprNodes.TypecastNode):
            # get typecast start
            expr_str = self._text[line][pos:]
            operand = self.print_ExprByPos(node.operand, is_improve = False)
            
            pattern = expr_str[:expr_str.find(operand) + len(operand)]
            s_type = self.print_CBaseTypeNode(node.base_type)
            changed = "cython.cast(%s, %s, typecheck= %s)" % \
                      (self.print_TypeTree(node.declarator) % s_type,
                       self.improve_Expr(node.operand, operand),
                       node.typecheck)
            #print(" expr: %s\n operand: %s\n %s -> %s" % (expr, operand, pattern, changed))
            expr = expr.replace(pattern, changed)
       
        # regular &... expression
        elif isinstance(node, ExprNodes.AmpersandNode):
            expr_str = self._text[line][pos:]
            # get operand &var
            pattern = "&" + findall("[\w|\[|\]]+", expr_str)[0]
            changed = "cython.address(%s)" % pattern[1:]
            expr = expr.replace(pattern, changed)
        
        # regular sizeof(type) expression
        elif isinstance(node, ExprNodes.SizeofTypeNode):
            # add borders of len 1 to replace sizeof() correctrly            
            expr_str = self._text[line][pos - 1:]
            start_pos = len("sizeof(") + 1
            # get operand till closing bracket
            brackets_cnt = 1
            for (end_pos, char) in enumerate(expr_str[start_pos:], start_pos):
                if   char == "(": brackets_cnt += 1
                elif char == ")": brackets_cnt -= 1
                if brackets_cnt == 0:
                    break
            end_pos += 1
            
            pattern = expr_str[:end_pos]
            s_type = self.print_CBaseTypeNode(node.base_type)
            changed = "%scython.sizeof(%s)" % \
                      (expr_str[:1],
                       self.print_TypeTree(node.declarator) % s_type)
            expr = expr.replace(pattern, changed)
        
        # regular sizeof(variable) expression
        elif isinstance(node, ExprNodes.SizeofVarNode):
            # add borders of len 1 to replace sizeof() correctrly            
            expr_str = self._text[line][pos - 1:]
            start_pos = len("sizeof(") + 1
            # get operand till closing bracket
            brackets_cnt = 1
            for (end_pos, char) in enumerate(expr_str[start_pos:], start_pos):
                if   char == "(": brackets_cnt += 1
                elif char == ")": brackets_cnt -= 1  
                if brackets_cnt == 0:
                    break
            end_pos += 1
                    
            pattern = expr_str[:end_pos]
            changed = expr_str[:1] + "cython." + pattern[1:]
            expr = expr.replace(pattern, changed)
        
        # just NULL  
        elif isinstance(node, ExprNodes.NullNode):
            # add borders of len 1 to replace NULL correctrly
            pattern = self._text[line][pos - 1:pos + 5]
            changed = pattern[:1] + "cython.NULL" + pattern[-1:]
            expr = expr.replace(pattern, changed, 1)
            
        # else try to improve children
        else:
            for attr in node.child_attrs:
                children = getattr(node, attr)
                if children is not None:
                    if type(children) is list:
                        for child in children:
                             expr = self.improve_Expr(child, expr)  
                    else:
                        expr = self.improve_Expr(children, expr)  
        
        return expr
# MIPT: end of class and all of modifications in Visitor.py

if __name__ == "__main__":
    import doctest
    doctest.testmod()
