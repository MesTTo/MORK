use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

type Subst = BTreeMap<String, Term>;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Term {
    Var(String),
    Fun(String, Vec<Term>),
}

impl Term {
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    pub fn sym(name: impl Into<String>) -> Self {
        Self::Fun(name.into(), Vec::new())
    }

    pub fn app(name: impl Into<String>, args: Vec<Term>) -> Self {
        Self::Fun(name.into(), args)
    }

    fn apply(&self, subst: &Subst) -> Self {
        match self {
            Self::Var(name) => subst
                .get(name)
                .map(|term| term.apply(subst))
                .unwrap_or_else(|| self.clone()),
            Self::Fun(name, args) => Self::Fun(
                name.clone(),
                args.iter().map(|arg| arg.apply(subst)).collect(),
            ),
        }
    }

    fn contains_var(&self, needle: &str, subst: &Subst) -> bool {
        match self.apply(subst) {
            Self::Var(name) => name == needle,
            Self::Fun(_, args) => args.iter().any(|arg| arg.contains_var(needle, subst)),
        }
    }

    fn collect_vars(&self, vars: &mut BTreeSet<String>) {
        match self {
            Self::Var(name) => {
                vars.insert(name.clone());
            }
            Self::Fun(_, args) => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
        }
    }

    pub fn is_ground(&self) -> bool {
        match self {
            Self::Var(_) => false,
            Self::Fun(_, args) => args.iter().all(Self::is_ground),
        }
    }

    fn non_variable_positions(&self) -> Vec<Vec<usize>> {
        let mut positions = Vec::new();
        self.collect_non_variable_positions(&mut Vec::new(), &mut positions);
        positions
    }

    fn collect_non_variable_positions(
        &self,
        path: &mut Vec<usize>,
        positions: &mut Vec<Vec<usize>>,
    ) {
        let Self::Fun(_, args) = self else {
            return;
        };
        positions.push(path.clone());
        for (index, arg) in args.iter().enumerate() {
            path.push(index);
            arg.collect_non_variable_positions(path, positions);
            path.pop();
        }
    }

    fn subterm(&self, position: &[usize]) -> &Self {
        if let Some((&head, tail)) = position.split_first() {
            let Self::Fun(_, args) = self else {
                unreachable!("positions are collected only through non-variable terms");
            };
            args[head].subterm(tail)
        } else {
            self
        }
    }

    fn replace(&self, position: &[usize], replacement: Self) -> Self {
        if let Some((&head, tail)) = position.split_first() {
            let Self::Fun(name, args) = self else {
                unreachable!("positions are collected only through non-variable terms");
            };
            let mut replaced_args = args.clone();
            replaced_args[head] = replaced_args[head].replace(tail, replacement);
            Self::Fun(name.clone(), replaced_args)
        } else {
            replacement
        }
    }

    pub fn to_metta(&self) -> String {
        match self {
            Self::Var(name) => format!("${name}"),
            Self::Fun(name, args) if args.is_empty() => name.clone(),
            Self::Fun(name, args) => {
                let rendered_args = args
                    .iter()
                    .map(Self::to_metta)
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("({name} {rendered_args})")
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rule {
    pub name: String,
    pub lhs: Term,
    pub rhs: Term,
}

impl Rule {
    pub fn new(name: impl Into<String>, lhs: Term, rhs: Term) -> Self {
        Self {
            name: name.into(),
            lhs,
            rhs,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateRule {
    pub name: String,
    pub patterns: Vec<Term>,
    pub remove: Vec<Term>,
    pub add: Vec<Term>,
}

impl StateRule {
    pub fn new(
        name: impl Into<String>,
        patterns: Vec<Term>,
        remove: Vec<Term>,
        add: Vec<Term>,
    ) -> Self {
        Self {
            name: name.into(),
            patterns,
            remove,
            add,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StateStep {
    pub rule: String,
    pub remove: Vec<Term>,
    pub add: Vec<Term>,
    pub after: Vec<Term>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdditiveSaturationRuleRound {
    pub rule: String,
    pub generated_steps: usize,
    pub generated_facts: usize,
    pub unique_generated_facts: usize,
    pub candidate_steps: usize,
    pub candidate_facts: usize,
    pub new_facts: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdditiveSaturationRound {
    pub round: usize,
    pub state_facts_before: usize,
    pub generated_steps: usize,
    pub generated_facts: usize,
    pub unique_generated_facts: usize,
    pub candidate_steps: usize,
    pub candidate_facts: usize,
    pub new_facts: usize,
    pub state_facts_after: usize,
    pub rules: Vec<AdditiveSaturationRuleRound>,
}

macro_rules! impl_additive_saturation_gap_methods {
    ($ty:ty) => {
        impl $ty {
            pub fn duplicate_generated_facts(&self) -> usize {
                self.generated_facts
                    .saturating_sub(self.unique_generated_facts)
            }

            pub fn already_known_generated_facts(&self) -> usize {
                self.unique_generated_facts.saturating_sub(self.new_facts)
            }

            pub fn redundant_generated_facts(&self) -> usize {
                self.generated_facts.saturating_sub(self.new_facts)
            }
        }
    };
}

impl_additive_saturation_gap_methods!(AdditiveSaturationRuleRound);
impl_additive_saturation_gap_methods!(AdditiveSaturationRound);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdditiveSaturationReport {
    pub state: BTreeSet<Term>,
    pub initial_facts: usize,
    pub final_facts: usize,
    pub derived_facts: usize,
    pub rounds_executed: usize,
    pub converged: bool,
    pub rounds: Vec<AdditiveSaturationRound>,
}

impl AdditiveSaturationReport {
    pub fn total_generated_steps(&self) -> usize {
        self.sum_rounds(|round| round.generated_steps)
    }

    pub fn total_generated_facts(&self) -> usize {
        self.sum_rounds(|round| round.generated_facts)
    }

    pub fn total_unique_generated_facts(&self) -> usize {
        self.sum_rounds(|round| round.unique_generated_facts)
    }

    pub fn total_candidate_steps(&self) -> usize {
        self.sum_rounds(|round| round.candidate_steps)
    }

    pub fn total_candidate_facts(&self) -> usize {
        self.sum_rounds(|round| round.candidate_facts)
    }

    pub fn total_new_facts(&self) -> usize {
        self.sum_rounds(|round| round.new_facts)
    }

    fn sum_rounds(&self, field: impl Fn(&AdditiveSaturationRound) -> usize) -> usize {
        self.rounds
            .iter()
            .map(field)
            .fold(0usize, usize::saturating_add)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AdditiveSaturationWorkSavings {
    pub generated_steps: usize,
    pub generated_facts: usize,
    pub unique_generated_facts: usize,
    pub candidate_steps: usize,
    pub candidate_facts: usize,
    pub new_facts: usize,
}

impl AdditiveSaturationWorkSavings {
    pub fn any(&self) -> bool {
        self.generated_steps > 0
            || self.generated_facts > 0
            || self.unique_generated_facts > 0
            || self.candidate_steps > 0
            || self.candidate_facts > 0
            || self.new_facts > 0
    }

    fn from_reports(
        baseline: &AdditiveSaturationReport,
        optimized: &AdditiveSaturationReport,
    ) -> Self {
        Self::from_counts(
            Self::report_totals(baseline),
            Self::report_totals(optimized),
        )
    }

    fn from_rounds(
        baseline: Option<&AdditiveSaturationRound>,
        optimized: Option<&AdditiveSaturationRound>,
    ) -> Self {
        Self::from_counts(Self::round_counts(baseline), Self::round_counts(optimized))
    }

    fn from_counts(baseline: Self, optimized: Self) -> Self {
        Self {
            generated_steps: baseline
                .generated_steps
                .saturating_sub(optimized.generated_steps),
            generated_facts: baseline
                .generated_facts
                .saturating_sub(optimized.generated_facts),
            unique_generated_facts: baseline
                .unique_generated_facts
                .saturating_sub(optimized.unique_generated_facts),
            candidate_steps: baseline
                .candidate_steps
                .saturating_sub(optimized.candidate_steps),
            candidate_facts: baseline
                .candidate_facts
                .saturating_sub(optimized.candidate_facts),
            new_facts: baseline.new_facts.saturating_sub(optimized.new_facts),
        }
    }

    fn report_totals(report: &AdditiveSaturationReport) -> Self {
        Self {
            generated_steps: report.total_generated_steps(),
            generated_facts: report.total_generated_facts(),
            unique_generated_facts: report.total_unique_generated_facts(),
            candidate_steps: report.total_candidate_steps(),
            candidate_facts: report.total_candidate_facts(),
            new_facts: report.total_new_facts(),
        }
    }

    fn round_counts(round: Option<&AdditiveSaturationRound>) -> Self {
        match round {
            Some(round) => Self {
                generated_steps: round.generated_steps,
                generated_facts: round.generated_facts,
                unique_generated_facts: round.unique_generated_facts,
                candidate_steps: round.candidate_steps,
                candidate_facts: round.candidate_facts,
                new_facts: round.new_facts,
            },
            None => Self::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdditiveSaturationRoundWorkSavings {
    pub round: usize,
    pub savings: AdditiveSaturationWorkSavings,
}

impl AdditiveSaturationRoundWorkSavings {
    pub fn any(&self) -> bool {
        self.savings.any()
    }

    fn from_reports(
        baseline: &AdditiveSaturationReport,
        optimized: &AdditiveSaturationReport,
    ) -> Vec<Self> {
        let round_count = baseline.rounds.len().max(optimized.rounds.len());

        (0..round_count)
            .map(|index| {
                let baseline_round = baseline.rounds.get(index);
                let optimized_round = optimized.rounds.get(index);
                let round = baseline_round
                    .or(optimized_round)
                    .map(|round| round.round)
                    .unwrap_or(index + 1);

                Self {
                    round,
                    savings: AdditiveSaturationWorkSavings::from_rounds(
                        baseline_round,
                        optimized_round,
                    ),
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdditiveSaturationComparisonReport {
    pub naive: AdditiveSaturationReport,
    pub semi_naive: AdditiveSaturationReport,
    pub semi_naive_savings: AdditiveSaturationWorkSavings,
    pub round_savings: Vec<AdditiveSaturationRoundWorkSavings>,
}

impl AdditiveSaturationComparisonReport {
    pub fn same_final_state(&self) -> bool {
        self.naive.state == self.semi_naive.state
    }

    pub fn same_convergence_status(&self) -> bool {
        self.naive.converged == self.semi_naive.converged
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CriticalPairWitness {
    pub outer_rule: String,
    pub inner_rule: String,
    pub position: Vec<usize>,
    pub peak: Term,
    pub left_normal: Term,
    pub right_normal: Term,
}

impl CriticalPairWitness {
    pub fn position_name(&self) -> String {
        if self.position.is_empty() {
            "root".to_owned()
        } else {
            format!(
                "p{}",
                self.position
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("_")
            )
        }
    }

    pub fn to_metta_atom(&self) -> String {
        format!(
            "(critical-pair {} {} {} {} {} {})",
            self.outer_rule,
            self.inner_rule,
            self.position_name(),
            self.peak.to_metta(),
            self.left_normal.to_metta(),
            self.right_normal.to_metta(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuleExtractionError {
    message: String,
}

impl RuleExtractionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RuleExtractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for RuleExtractionError {}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Sexpr {
    Atom(String),
    List(Vec<Sexpr>),
}

impl Sexpr {
    fn atom(&self) -> Option<&str> {
        let Self::Atom(atom) = self else {
            return None;
        };
        Some(atom)
    }

    fn to_rule_name(&self, fallback: usize) -> String {
        match self {
            Self::Atom(atom) if is_plain_metta_symbol(atom) => atom.clone(),
            _ => format!("rule-{fallback}"),
        }
    }
}

pub fn first_non_joinable_witness(rules: &[Rule], fuel: usize) -> Option<CriticalPairWitness> {
    for (outer_index, outer) in rules.iter().enumerate() {
        for (inner_index, inner) in rules.iter().enumerate() {
            for position in outer.lhs.non_variable_positions() {
                if outer_index == inner_index && position.is_empty() {
                    continue;
                }

                let mut subst = Subst::new();
                if !unify(outer.lhs.subterm(&position), &inner.lhs, &mut subst) {
                    continue;
                }

                let peak = outer.lhs.apply(&subst);
                let left_branch = outer.rhs.apply(&subst);
                let right_branch = peak.replace(&position, inner.rhs.apply(&subst));
                let grounded = ground_open_terms(&[peak, left_branch, right_branch]);

                let left_normal = normalize(grounded[1].clone(), rules, fuel);
                let right_normal = normalize(grounded[2].clone(), rules, fuel);
                if left_normal != right_normal {
                    return Some(CriticalPairWitness {
                        outer_rule: outer.name.clone(),
                        inner_rule: inner.name.clone(),
                        position,
                        peak: grounded[0].clone(),
                        left_normal,
                        right_normal,
                    });
                }
            }
        }
    }
    None
}

pub fn rules_from_mm2_program(program: &str) -> Result<Vec<Rule>, RuleExtractionError> {
    let sexprs = Parser::new(program).parse_all()?;
    Ok(sexprs
        .iter()
        .enumerate()
        .flat_map(|(index, sexpr)| rules_from_exec(index, sexpr))
        .collect())
}

pub fn state_rules_from_mm2_program(program: &str) -> Result<Vec<StateRule>, RuleExtractionError> {
    let sexprs = Parser::new(program).parse_all()?;
    Ok(sexprs
        .iter()
        .enumerate()
        .filter_map(|(index, sexpr)| state_rule_from_exec(index, sexpr))
        .collect())
}

pub fn ground_facts_from_mm2_program(program: &str) -> Result<BTreeSet<Term>, RuleExtractionError> {
    let sexprs = Parser::new(program).parse_all()?;
    Ok(sexprs
        .iter()
        .filter(|sexpr| !is_exec(sexpr))
        .filter_map(term_from_sexpr)
        .filter(Term::is_ground)
        .collect())
}

pub fn state_rule_successors(state: &BTreeSet<Term>, rule: &StateRule) -> Vec<StateStep> {
    collect_state_rule_substitutions(state, rule)
        .into_iter()
        .filter_map(|subst| {
            let remove = instantiate_ground_terms(&rule.remove, &subst)?;
            let add = instantiate_ground_terms(&rule.add, &subst)?;
            let mut after = state.clone();
            for term in &remove {
                after.remove(term);
            }
            for term in &add {
                after.insert(term.clone());
            }
            (after != *state).then(|| StateStep {
                rule: rule.name.clone(),
                remove,
                add,
                after: after.into_iter().collect(),
            })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub fn saturate_additive_state(
    initial: impl IntoIterator<Item = Term>,
    rules: &[StateRule],
    max_rounds: usize,
) -> BTreeSet<Term> {
    saturate_additive_state_report(initial, rules, max_rounds).state
}

pub fn saturate_additive_state_semi_naive(
    initial: impl IntoIterator<Item = Term>,
    rules: &[StateRule],
    max_rounds: usize,
) -> BTreeSet<Term> {
    saturate_additive_state_semi_naive_report(initial, rules, max_rounds).state
}

pub fn saturate_additive_state_report(
    initial: impl IntoIterator<Item = Term>,
    rules: &[StateRule],
    max_rounds: usize,
) -> AdditiveSaturationReport {
    saturate_additive_state_report_from_state(
        initial.into_iter().collect(),
        rules,
        max_rounds,
        |state, _frontier, rule| state_rule_add_candidates(state, rule),
    )
}

pub fn saturate_additive_state_semi_naive_report(
    initial: impl IntoIterator<Item = Term>,
    rules: &[StateRule],
    max_rounds: usize,
) -> AdditiveSaturationReport {
    saturate_additive_state_report_from_state(
        initial.into_iter().collect(),
        rules,
        max_rounds,
        state_rule_add_candidates_touching,
    )
}

pub fn compare_additive_saturation_reports(
    initial: impl IntoIterator<Item = Term>,
    rules: &[StateRule],
    max_rounds: usize,
) -> AdditiveSaturationComparisonReport {
    let initial = initial.into_iter().collect::<BTreeSet<_>>();
    let naive = saturate_additive_state_report(initial.clone(), rules, max_rounds);
    let semi_naive = saturate_additive_state_semi_naive_report(initial, rules, max_rounds);
    let semi_naive_savings = AdditiveSaturationWorkSavings::from_reports(&naive, &semi_naive);
    let round_savings = AdditiveSaturationRoundWorkSavings::from_reports(&naive, &semi_naive);

    AdditiveSaturationComparisonReport {
        naive,
        semi_naive,
        semi_naive_savings,
        round_savings,
    }
}

fn saturate_additive_state_report_from_state<F>(
    mut state: BTreeSet<Term>,
    rules: &[StateRule],
    max_rounds: usize,
    mut add_candidates: F,
) -> AdditiveSaturationReport
where
    F: FnMut(&BTreeSet<Term>, &BTreeSet<Term>, &StateRule) -> Vec<Vec<Term>>,
{
    let initial_facts = state.len();
    let mut frontier = state.clone();
    let mut rounds = Vec::new();
    let mut converged = false;

    for round in 1..=max_rounds {
        let state_facts_before = state.len();
        let mut next = state.clone();
        let mut new_terms = BTreeSet::new();
        let mut generated_steps = 0usize;
        let mut generated_facts = 0usize;
        let mut generated_terms = BTreeSet::new();
        let mut candidate_steps = 0usize;
        let mut candidate_facts = 0usize;
        let mut new_facts = 0usize;
        let mut rule_rounds = Vec::new();

        for rule in rules.iter().filter(|rule| rule.remove.is_empty()) {
            let mut rule_round = AdditiveSaturationRuleRound {
                rule: rule.name.clone(),
                generated_steps: 0,
                generated_facts: 0,
                unique_generated_facts: 0,
                candidate_steps: 0,
                candidate_facts: 0,
                new_facts: 0,
            };
            let mut rule_generated_terms = BTreeSet::new();
            for add_terms in add_candidates(&state, &frontier, rule) {
                let add_len = add_terms.len();
                generated_steps += 1;
                generated_facts = generated_facts.saturating_add(add_len);
                rule_round.generated_steps += 1;
                rule_round.generated_facts = rule_round.generated_facts.saturating_add(add_len);
                for term in &add_terms {
                    generated_terms.insert(term.clone());
                    rule_generated_terms.insert(term.clone());
                }

                let mut candidate_new_facts = 0usize;
                for term in add_terms {
                    if next.insert(term.clone()) {
                        new_terms.insert(term);
                        new_facts += 1;
                        candidate_new_facts += 1;
                    }
                }
                if candidate_new_facts > 0 {
                    candidate_steps += 1;
                    candidate_facts = candidate_facts.saturating_add(add_len);
                    rule_round.candidate_steps += 1;
                    rule_round.candidate_facts = rule_round.candidate_facts.saturating_add(add_len);
                    rule_round.new_facts += candidate_new_facts;
                }
            }
            rule_round.unique_generated_facts = rule_generated_terms.len();
            if rule_round.generated_steps > 0 {
                rule_rounds.push(rule_round);
            }
        }
        rounds.push(AdditiveSaturationRound {
            round,
            state_facts_before,
            generated_steps,
            generated_facts,
            unique_generated_facts: generated_terms.len(),
            candidate_steps,
            candidate_facts,
            new_facts,
            state_facts_after: next.len(),
            rules: rule_rounds,
        });
        if new_terms.is_empty() {
            converged = true;
            break;
        }
        state = next;
        frontier = new_terms;
    }

    let final_facts = state.len();
    AdditiveSaturationReport {
        state,
        initial_facts,
        final_facts,
        derived_facts: final_facts.saturating_sub(initial_facts),
        rounds_executed: rounds.len(),
        converged,
        rounds,
    }
}

fn unify(lhs: &Term, rhs: &Term, subst: &mut Subst) -> bool {
    let lhs = lhs.apply(subst);
    let rhs = rhs.apply(subst);
    match (lhs, rhs) {
        (Term::Var(name), term) | (term, Term::Var(name)) => bind_var(name, term, subst),
        (Term::Fun(lhs_name, lhs_args), Term::Fun(rhs_name, rhs_args)) => {
            lhs_name == rhs_name
                && lhs_args.len() == rhs_args.len()
                && lhs_args
                    .iter()
                    .zip(rhs_args.iter())
                    .all(|(lhs_arg, rhs_arg)| unify(lhs_arg, rhs_arg, subst))
        }
    }
}

fn bind_var(name: String, term: Term, subst: &mut Subst) -> bool {
    if term == Term::Var(name.clone()) {
        return true;
    }
    if term.contains_var(&name, subst) {
        return false;
    }
    subst.insert(name, term);
    true
}

fn rewrite_once(term: &Term, rules: &[Rule]) -> Option<Term> {
    for position in term.non_variable_positions() {
        for rule in rules {
            let mut subst = Subst::new();
            if unify(term.subterm(&position), &rule.lhs, &mut subst) {
                return Some(term.replace(&position, rule.rhs.apply(&subst)));
            }
        }
    }
    None
}

fn normalize(term: Term, rules: &[Rule], fuel: usize) -> Term {
    let mut current = term;
    for _ in 0..fuel {
        let Some(next) = rewrite_once(&current, rules) else {
            break;
        };
        current = next;
    }
    current
}

fn ground_open_terms(terms: &[Term]) -> Vec<Term> {
    let mut vars = BTreeSet::new();
    for term in terms {
        term.collect_vars(&mut vars);
    }

    let constants = ["a", "b", "c", "d", "e", "f", "h", "i"];
    let mut subst = Subst::new();
    for (index, var) in vars.into_iter().enumerate() {
        let term = constants
            .get(index)
            .map(|name| Term::sym(*name))
            .unwrap_or_else(|| Term::sym(format!("g{index}")));
        subst.insert(var, term);
    }

    terms.iter().map(|term| term.apply(&subst)).collect()
}

fn rules_from_exec(index: usize, sexpr: &Sexpr) -> Vec<Rule> {
    let Sexpr::List(items) = sexpr else {
        return Vec::new();
    };
    if items.len() != 4 || items[0].atom() != Some("exec") {
        return Vec::new();
    }

    let Some(patterns) = collect_group_args(&items[2], ",") else {
        return Vec::new();
    };
    if patterns.len() != 1 {
        return Vec::new();
    }
    let Some(lhs) = term_from_sexpr(patterns[0]) else {
        return Vec::new();
    };

    let rhs_terms = positive_templates(&items[3]);
    if rhs_terms.is_empty() {
        return Vec::new();
    }

    let base_name = items[1].to_rule_name(index);
    let multiple_outputs = rhs_terms.len() > 1;
    rhs_terms
        .into_iter()
        .enumerate()
        .map(|(output_index, rhs)| {
            let name = if multiple_outputs {
                format!("{base_name}-out-{output_index}")
            } else {
                base_name.clone()
            };
            Rule::new(name, lhs.clone(), rhs)
        })
        .collect()
}

fn state_rule_from_exec(index: usize, sexpr: &Sexpr) -> Option<StateRule> {
    let Sexpr::List(items) = sexpr else {
        return None;
    };
    if items.len() != 4 || items[0].atom() != Some("exec") {
        return None;
    }

    let patterns = collect_group_args(&items[2], ",")?
        .into_iter()
        .map(term_from_sexpr)
        .collect::<Option<Vec<_>>>()?;
    let (remove, add) = template_effects(&items[3])?;
    if patterns.is_empty() || remove.is_empty() && add.is_empty() {
        return None;
    }

    Some(StateRule::new(
        items[1].to_rule_name(index),
        patterns,
        remove,
        add,
    ))
}

fn is_exec(sexpr: &Sexpr) -> bool {
    matches!(
        sexpr,
        Sexpr::List(items) if items.first().and_then(Sexpr::atom) == Some("exec")
    )
}

fn collect_group_args<'a>(sexpr: &'a Sexpr, head: &str) -> Option<Vec<&'a Sexpr>> {
    match sexpr {
        Sexpr::List(items) if items.first().and_then(Sexpr::atom) == Some(head) => {
            Some(items[1..].iter().collect())
        }
        _ => None,
    }
}

fn positive_templates(sexpr: &Sexpr) -> Vec<Term> {
    template_args(sexpr)
        .iter()
        .filter_map(|arg| match arg {
            Sexpr::List(items) if items.first().and_then(Sexpr::atom) == Some("+") => (items.len()
                == 2)
                .then(|| term_from_sexpr(&items[1]))
                .flatten(),
            Sexpr::List(items) if items.first().and_then(Sexpr::atom) == Some("-") => None,
            other => term_from_sexpr(other),
        })
        .collect()
}

fn template_effects(sexpr: &Sexpr) -> Option<(Vec<Term>, Vec<Term>)> {
    let mut remove = Vec::new();
    let mut add = Vec::new();
    for arg in template_args(sexpr) {
        match arg {
            Sexpr::List(items) if items.first().and_then(Sexpr::atom) == Some("+") => {
                let [_, term] = items.as_slice() else {
                    return None;
                };
                add.push(term_from_sexpr(term)?);
            }
            Sexpr::List(items) if items.first().and_then(Sexpr::atom) == Some("-") => {
                let [_, term] = items.as_slice() else {
                    return None;
                };
                remove.push(term_from_sexpr(term)?);
            }
            other => add.push(term_from_sexpr(other)?),
        }
    }

    Some((remove, add))
}

fn template_args(sexpr: &Sexpr) -> &[Sexpr] {
    match sexpr {
        Sexpr::List(items)
            if matches!(items.first().and_then(Sexpr::atom), Some(",") | Some("O")) =>
        {
            &items[1..]
        }
        _ => std::slice::from_ref(sexpr),
    }
}

fn state_rule_add_candidates(state: &BTreeSet<Term>, rule: &StateRule) -> Vec<Vec<Term>> {
    collect_state_rule_substitutions(state, rule)
        .into_iter()
        .filter_map(|subst| instantiate_ground_terms(&rule.add, &subst))
        .collect()
}

fn state_rule_add_candidates_touching(
    state: &BTreeSet<Term>,
    frontier: &BTreeSet<Term>,
    rule: &StateRule,
) -> Vec<Vec<Term>> {
    collect_state_rule_substitutions_touching(state, frontier, rule)
        .into_iter()
        .filter_map(|subst| instantiate_ground_terms(&rule.add, &subst))
        .collect()
}

fn collect_state_rule_substitutions(state: &BTreeSet<Term>, rule: &StateRule) -> Vec<Subst> {
    collect_state_rule_substitutions_filtered(state, None, rule)
}

fn collect_state_rule_substitutions_touching(
    state: &BTreeSet<Term>,
    frontier: &BTreeSet<Term>,
    rule: &StateRule,
) -> Vec<Subst> {
    collect_state_rule_substitutions_filtered(state, Some(frontier), rule)
}

fn collect_state_rule_substitutions_filtered(
    state: &BTreeSet<Term>,
    frontier: Option<&BTreeSet<Term>>,
    rule: &StateRule,
) -> Vec<Subst> {
    if rule.patterns.is_empty() {
        return Vec::new();
    }

    let facts = state.iter().collect::<Vec<_>>();
    let mut substitutions = Vec::new();
    collect_state_matches(
        &rule.patterns,
        &facts,
        frontier,
        false,
        0,
        &Subst::new(),
        &mut substitutions,
    );
    substitutions
}

fn collect_state_matches(
    patterns: &[Term],
    facts: &[&Term],
    frontier: Option<&BTreeSet<Term>>,
    touched_frontier: bool,
    pattern_index: usize,
    subst: &Subst,
    matches: &mut Vec<Subst>,
) {
    let Some(pattern) = patterns.get(pattern_index) else {
        if frontier.is_none() || touched_frontier {
            matches.push(subst.clone());
        }
        return;
    };

    for fact in facts {
        let mut next = subst.clone();
        if unify(pattern, fact, &mut next) {
            let next_touched = touched_frontier
                || frontier
                    .map(|frontier| frontier.contains(*fact))
                    .unwrap_or(false);
            collect_state_matches(
                patterns,
                facts,
                frontier,
                next_touched,
                pattern_index + 1,
                &next,
                matches,
            );
        }
    }
}

fn instantiate_ground_terms(terms: &[Term], subst: &Subst) -> Option<Vec<Term>> {
    terms
        .iter()
        .map(|term| {
            let instantiated = term.apply(subst);
            instantiated.is_ground().then_some(instantiated)
        })
        .collect()
}

fn term_from_sexpr(sexpr: &Sexpr) -> Option<Term> {
    match sexpr {
        Sexpr::Atom(atom) if let Some(var) = atom.strip_prefix('$') => {
            Some(Term::var(var.to_owned()))
        }
        Sexpr::Atom(atom) => Some(Term::sym(atom.clone())),
        Sexpr::List(items) => {
            let (head, args) = items.split_first()?;
            let head = head.atom()?;
            Some(Term::app(
                head.to_owned(),
                args.iter()
                    .map(term_from_sexpr)
                    .collect::<Option<Vec<_>>>()?,
            ))
        }
    }
}

fn is_plain_metta_symbol(atom: &str) -> bool {
    !atom.is_empty()
        && !atom.starts_with('$')
        && atom
            .bytes()
            .all(|b| !b.is_ascii_whitespace() && !matches!(b, b'(' | b')' | b'"' | b';'))
}

struct Parser<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            offset: 0,
        }
    }

    fn parse_all(&mut self) -> Result<Vec<Sexpr>, RuleExtractionError> {
        let mut out = Vec::new();
        loop {
            self.skip_ws_and_comments();
            if self.offset == self.input.len() {
                return Ok(out);
            }
            out.push(self.parse_expr()?);
        }
    }

    fn parse_expr(&mut self) -> Result<Sexpr, RuleExtractionError> {
        self.skip_ws_and_comments();
        match self.peek() {
            Some(b'(') => self.parse_list(),
            Some(b')') => Err(RuleExtractionError::new(format!(
                "unexpected ')' at byte {}",
                self.offset
            ))),
            Some(_) => self.parse_atom().map(Sexpr::Atom),
            None => Err(RuleExtractionError::new("unexpected end of input")),
        }
    }

    fn parse_list(&mut self) -> Result<Sexpr, RuleExtractionError> {
        self.offset += 1;
        let mut items = Vec::new();
        loop {
            self.skip_ws_and_comments();
            match self.peek() {
                Some(b')') => {
                    self.offset += 1;
                    return Ok(Sexpr::List(items));
                }
                Some(_) => items.push(self.parse_expr()?),
                None => return Err(RuleExtractionError::new("unclosed list")),
            }
        }
    }

    fn parse_atom(&mut self) -> Result<String, RuleExtractionError> {
        if self.peek() == Some(b'"') {
            return self.parse_quoted_atom();
        }

        let start = self.offset;
        while let Some(byte) = self.peek() {
            if byte.is_ascii_whitespace() || matches!(byte, b'(' | b')' | b';') {
                break;
            }
            self.offset += 1;
        }
        if self.offset == start {
            return Err(RuleExtractionError::new(format!(
                "expected atom at byte {}",
                self.offset
            )));
        }
        String::from_utf8(self.input[start..self.offset].to_vec())
            .map_err(|_| RuleExtractionError::new("atom is not valid UTF-8"))
    }

    fn parse_quoted_atom(&mut self) -> Result<String, RuleExtractionError> {
        let start = self.offset;
        self.offset += 1;
        while let Some(byte) = self.peek() {
            self.offset += 1;
            match byte {
                b'"' => {
                    return String::from_utf8(self.input[start..self.offset].to_vec())
                        .map_err(|_| RuleExtractionError::new("quoted atom is not valid UTF-8"));
                }
                b'\\' => {
                    if self.peek().is_some() {
                        self.offset += 1;
                    }
                }
                _ => {}
            }
        }
        Err(RuleExtractionError::new("unterminated quoted atom"))
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
                self.offset += 1;
            }
            if self.peek() != Some(b';') {
                break;
            }
            while let Some(byte) = self.peek() {
                self.offset += 1;
                if byte == b'\n' {
                    break;
                }
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.offset).copied()
    }
}
