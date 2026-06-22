namespace MorkFormal

abbrev Byte := UInt8
abbrev Path := List Byte

structure ZipperContext where
  rootPrefix : Path
deriving DecidableEq, Repr

structure ZipperFocus where
  localPath : Path
deriving DecidableEq, Repr

def originPath (ctx : ZipperContext) (focus : ZipperFocus) : Path :=
  ctx.rootPrefix ++ focus.localPath

theorem origin_reset (ctx : ZipperContext) :
    originPath ctx { localPath := [] } = ctx.rootPrefix := by
  simp [originPath]

theorem origin_descend
    (ctx : ZipperContext) (focus : ZipperFocus) (suffix : Path) :
    originPath ctx { localPath := focus.localPath ++ suffix } =
      originPath ctx focus ++ suffix := by
  simp [originPath, List.append_assoc]

abbrev Assignment := Nat -> Nat
abbrev Factor := Assignment -> Prop

def Sat : List Factor -> Assignment -> Prop
  | [], _ => True
  | factor :: rest, assignment => factor assignment /\ Sat rest assignment

theorem sat_swap
    (left right : Factor) (rest : List Factor) (assignment : Assignment) :
    Sat (left :: right :: rest) assignment <->
      Sat (right :: left :: rest) assignment := by
  constructor
  · intro h
    exact And.intro h.right.left (And.intro h.left h.right.right)
  · intro h
    exact And.intro h.right.left (And.intro h.left h.right.right)

theorem sat_swap_preserves_solutions
    (left right : Factor) (rest : List Factor) :
    forall assignment : Assignment,
      Sat (left :: right :: rest) assignment <->
        Sat (right :: left :: rest) assignment := by
  intro assignment
  exact sat_swap left right rest assignment

end MorkFormal
