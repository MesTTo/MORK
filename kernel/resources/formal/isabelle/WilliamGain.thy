theory WilliamGain
  imports Main "HOL-Library.Sublist"
begin

text \<open>
  Byte accounting of the WILLIAM factoring (kernel/src/william.rs, factor_pattern).

  A store is modelled as a list of atoms, each a byte list; its size is the sum of
  atom lengths. Factoring a pattern shared by the atoms \<open>pattern @ s\<close> (one per suffix
  \<open>s\<close>) rewrites each to \<open>refh @ s\<close> and adds one definition atom \<open>defh @ pattern\<close>,
  where the two id headers have the same length R (REF_COST). The theorem is the
  exact saving, and the corollary relates it to the gain-index weight
  \<open>predicted = (C - 1) * L - C * R\<close>: the realized saving is \<open>predicted - R\<close>, the one
  header the marginal formula does not charge. The Rust tests pin the measured
  \<open>bytes_before - bytes_after\<close> to this identity; this theory proves the identity
  itself for every pattern, header pair, and suffix list.
\<close>

definition bytes :: "'a list list \<Rightarrow> nat" where
  "bytes atoms = (\<Sum>a\<leftarrow>atoms. length a)"

lemma bytes_Nil [simp]: "bytes [] = 0"
  by (simp add: bytes_def)

lemma bytes_Cons [simp]: "bytes (a # as) = length a + bytes as"
  by (simp add: bytes_def)

lemma bytes_map_prepend: "bytes (map ((@) p) ss) = length ss * length p + bytes ss"
  by (induct ss) (simp_all add: algebra_simps)

theorem factoring_accounting:
  fixes pattern refh defh :: "'a list" and suffixes :: "'a list list"
  defines C_def: "C \<equiv> int (length suffixes)"
      and L_def: "L \<equiv> int (length pattern)"
      and R_def: "R \<equiv> int (length refh)"
  assumes defh_len: "length defh = length refh"
  shows "int (bytes (map ((@) pattern) suffixes))
           - int (bytes (map ((@) refh) suffixes) + bytes [defh @ pattern])
         = (C - 1) * L - (C + 1) * R"
proof -
  have before: "int (bytes (map ((@) pattern) suffixes)) = C * L + int (bytes suffixes)"
    by (simp add: bytes_map_prepend C_def L_def)
  have after: "int (bytes (map ((@) refh) suffixes) + bytes [defh @ pattern])
                 = C * R + int (bytes suffixes) + R + L"
    by (simp add: bytes_map_prepend defh_len C_def R_def L_def)
  show ?thesis
    unfolding before after by (simp add: algebra_simps)
qed

corollary realized_is_predicted_minus_ref:
  fixes pattern refh defh :: "'a list" and suffixes :: "'a list list"
  assumes "length defh = length refh"
  shows "int (bytes (map ((@) pattern) suffixes))
           - int (bytes (map ((@) refh) suffixes) + bytes [defh @ pattern])
         = ((int (length suffixes) - 1) * int (length pattern)
              - int (length suffixes) * int (length refh))
           - int (length refh)"
  using factoring_accounting[OF assms] by (simp add: algebra_simps)

text \<open>
  The maximal top-k selection (weighted_paths.rs, top_k_maximal) walks candidates in
  descending weight and keeps one only when it neither extends nor is extended by an
  already-kept pattern. Modelling exactly that guard, the kept set is always a
  prefix-free antichain: along any root-to-leaf chain at most one pattern survives,
  which is the report dedup the whitepaper asks for.
\<close>

definition antichain :: "'a list list \<Rightarrow> bool" where
  "antichain ps = (\<forall>a\<in>set ps. \<forall>b\<in>set ps. a \<noteq> b \<longrightarrow> \<not> prefix a b)"

fun greedy :: "'a list list \<Rightarrow> 'a list list \<Rightarrow> 'a list list" where
  "greedy kept [] = kept"
| "greedy kept (c # cs) =
     (if \<exists>k\<in>set kept. prefix k c \<or> prefix c k then greedy kept cs
      else greedy (kept @ [c]) cs)"

lemma greedy_preserves_antichain:
  assumes "antichain kept"
  shows "antichain (greedy kept cs)"
  using assms
proof (induction cs arbitrary: kept)
  case Nil
  then show ?case by simp
next
  case (Cons c cs)
  show ?case
  proof (cases "\<exists>k\<in>set kept. prefix k c \<or> prefix c k")
    case True
    with Cons show ?thesis by simp
  next
    case False
    have step: "antichain (kept @ [c])"
      unfolding antichain_def
    proof (intro ballI impI)
      fix a b
      assume a: "a \<in> set (kept @ [c])" and b: "b \<in> set (kept @ [c])" and ne: "a \<noteq> b"
      show "\<not> prefix a b"
      proof
        assume p: "prefix a b"
        from a b consider
            "a \<in> set kept" "b \<in> set kept"
          | "a \<in> set kept" "b = c"
          | "a = c" "b \<in> set kept"
          | "a = c" "b = c"
          by auto
        then show False
        proof cases
          case 1
          with Cons.prems ne p show False unfolding antichain_def by blast
        next
          case 2
          with False p show False by blast
        next
          case 3
          with False p show False by blast
        next
          case 4
          with ne show False by simp
        qed
      qed
    qed
    from False step Cons.IH show ?thesis by simp
  qed
qed

corollary greedy_output_is_prefix_free: "antichain (greedy [] cs)"
  by (rule greedy_preserves_antichain) (simp add: antichain_def)

end
