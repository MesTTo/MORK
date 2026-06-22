namespace MorkFormal

abbrev SymbolId := Nat
abbrev VarSlot := Nat

inductive AtomKind where
  | symbol : SymbolId -> AtomKind
  | newVar : VarSlot -> AtomKind
  | varRef : VarSlot -> AtomKind
deriving DecidableEq, Repr

inductive Expr where
  | atom : AtomKind -> Expr
  | app : Expr -> List Expr -> Expr

inductive Crumb where
  | appHead : List Expr -> Crumb
  | appArg : Expr -> List Expr -> List Expr -> Crumb

abbrev Context := List Crumb

def plug : Context -> Expr -> Expr
  | [], focus => focus
  | Crumb.appHead args :: rest, focus => plug rest (Expr.app focus args)
  | Crumb.appArg head leftRev right :: rest, focus =>
      plug rest (Expr.app head (leftRev.reverse ++ (focus :: right)))

structure Zipper where
  focus : Expr
  ctx : Context

def current (z : Zipper) : Expr :=
  plug z.ctx z.focus

def replaceFocus (newFocus : Expr) (z : Zipper) : Zipper :=
  { z with focus := newFocus }

def downHead : Zipper -> Option Zipper
  | { focus := Expr.app head args, ctx := ctx } =>
      some { focus := head, ctx := Crumb.appHead args :: ctx }
  | _ => none

def up : Zipper -> Option Zipper
  | { focus := _, ctx := [] } => none
  | { focus := focus, ctx := Crumb.appHead args :: rest } =>
      some { focus := Expr.app focus args, ctx := rest }
  | { focus := focus, ctx := Crumb.appArg head leftRev right :: rest } =>
      some {
        focus := Expr.app head (leftRev.reverse ++ (focus :: right)),
        ctx := rest
      }

theorem downHead_preserves_current
    (head : Expr) (args : List Expr) (ctx : Context) :
    current { focus := head, ctx := Crumb.appHead args :: ctx } =
      current { focus := Expr.app head args, ctx := ctx } := rfl

theorem up_after_downHead
    (head : Expr) (args : List Expr) (ctx : Context) :
    up { focus := head, ctx := Crumb.appHead args :: ctx } =
      some { focus := Expr.app head args, ctx := ctx } := rfl

theorem downHead_roundTrip
    (head : Expr) (args : List Expr) (ctx : Context) :
    (downHead { focus := Expr.app head args, ctx := ctx } >>= up) =
      some { focus := Expr.app head args, ctx := ctx } := rfl

theorem appArg_preserves_current
    (head focus : Expr)
    (leftRev right : List Expr)
    (ctx : Context) :
    current {
      focus := focus,
      ctx := Crumb.appArg head leftRev right :: ctx
    } =
      current {
        focus := Expr.app head (leftRev.reverse ++ (focus :: right)),
        ctx := ctx
      } := rfl

theorem up_after_appArg
    (head focus : Expr)
    (leftRev right : List Expr)
    (ctx : Context) :
    up {
      focus := focus,
      ctx := Crumb.appArg head leftRev right :: ctx
    } =
      some {
        focus := Expr.app head (leftRev.reverse ++ (focus :: right)),
        ctx := ctx
      } := rfl

theorem replace_reconstructs (newFocus : Expr) (z : Zipper) :
    current (replaceFocus newFocus z) = plug z.ctx newFocus := by
  cases z
  rfl

theorem replace_head_context
    (oldFocus newFocus : Expr)
    (args : List Expr)
    (ctx : Context) :
    current (replaceFocus newFocus
      { focus := oldFocus, ctx := Crumb.appHead args :: ctx }) =
      current { focus := Expr.app newFocus args, ctx := ctx } := rfl

theorem replace_arg_context
    (head oldFocus newFocus : Expr)
    (leftRev right : List Expr)
    (ctx : Context) :
    current (replaceFocus newFocus {
      focus := oldFocus,
      ctx := Crumb.appArg head leftRev right :: ctx
    }) =
      current {
        focus := Expr.app head (leftRev.reverse ++ (newFocus :: right)),
        ctx := ctx
      } := rfl

def replaceHead (newHead : Expr) : Expr -> Expr
  | Expr.app _ args => Expr.app newHead args
  | expr => expr

def replaceArgs (newArgs : List Expr) : Expr -> Expr
  | Expr.app head _ => Expr.app head newArgs
  | expr => expr

theorem head_args_replacement_commutes
    (head newHead : Expr) (args newArgs : List Expr) :
    replaceArgs newArgs (replaceHead newHead (Expr.app head args)) =
      replaceHead newHead (replaceArgs newArgs (Expr.app head args)) := rfl

structure Edge where
  id : Nat
  label : Nat
  ports : List Nat
deriving DecidableEq, Repr

def Edge.arity (edge : Edge) : Nat :=
  edge.ports.length

abbrev Graph := List Edge

def replacementCompatible (old replacement : Edge) : Prop :=
  old.arity = replacement.arity

def replaceOneEdge (target : Nat) (replacement edge : Edge) : Edge :=
  if edge.id = target ∧ edge.arity = replacement.arity then
    { replacement with id := edge.id }
  else
    edge

def replaceEdges (target : Nat) (replacement : Edge) : Graph -> Graph
  | [] => []
  | edge :: rest =>
      replaceOneEdge target replacement edge ::
        replaceEdges target replacement rest

theorem replacement_edge_arity
    (edge replacement : Edge)
    (compatible : replacementCompatible edge replacement) :
    ({ replacement with id := edge.id } : Edge).arity = edge.arity := by
  simpa [replacementCompatible, Edge.arity] using compatible.symm

theorem replaceOneEdge_preserves_arity_when_compatible
    (edge replacement : Edge)
    (compatible : replacementCompatible edge replacement) :
    (replaceOneEdge edge.id replacement edge).arity = edge.arity := by
  have arityEq : edge.arity = replacement.arity := compatible
  simpa [replaceOneEdge, arityEq] using
    replacement_edge_arity edge replacement compatible

theorem replaceOneEdge_keeps_independent_edge
    (target : Nat) (replacement edge : Edge)
    (independent : edge.id ≠ target) :
    replaceOneEdge target replacement edge = edge := by
  simp [replaceOneEdge, independent]

theorem replace_two_distinct_edges_commute
    (left right leftReplacement rightReplacement : Edge)
    (distinct : left.id ≠ right.id)
    (leftCompatible : replacementCompatible left leftReplacement)
    (rightCompatible : replacementCompatible right rightReplacement) :
    replaceEdges right.id rightReplacement
        (replaceEdges left.id leftReplacement [left, right]) =
      replaceEdges left.id leftReplacement
        (replaceEdges right.id rightReplacement [left, right]) := by
  have leftArity : left.arity = leftReplacement.arity := leftCompatible
  have rightArity : right.arity = rightReplacement.arity := rightCompatible
  simp [
    replaceEdges,
    replaceOneEdge,
    leftArity,
    rightArity,
    distinct,
    Ne.symm distinct
  ]

end MorkFormal
