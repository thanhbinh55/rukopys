"""
Custom Kaggle evaluation metric for Ukrainian Handwritten Text Recognition.

Score = 0.15 * Detection_F1 + 0.05 * ClassAcc + 0.30 * (1 - CER) + 0.50 * (1 - PageCER)

Text is normalized before CER comparison:
  - Cyrillic/Latin lookalike characters → Cyrillic
  - All dash types → hyphen-minus
  - Whitespace collapse, strip
  - Quote/apostrophe normalization
  - Strikethrough markers ~~text~~ removed
  - Formula: Unicode super/subscripts → ^/_ notation, single-char braces removed
  - Tables: whitespace around pipes stripped

Submission format: CSV with columns `image` and `regions`.
The `regions` column contains a JSON-encoded list of region objects:
    [{"bbox": [x1, y1, x2, y2], "type": "handwritten", "text": "..."}]

Required fields per region: bbox, type, text.
Region types: handwritten, printed, formula, table, annotation, image, graph.
If an image has no regions, use an empty list: []

Note: `language` and `legibility` are GT-only attributes — participants do not need to predict them.

Recent changes
--------------
- PageCER is now computed symmetrically: prediction regions that match a
  non-scorable GT region (illegible / language=other / image / graph)
  are excluded from `pred_page`, in the same way GT non-scorable regions
  are excluded from `gt_page`. Previously, a prediction's text for an
  illegible GT region inflated `pred_page` while the GT side excluded it,
  preventing theoretically-perfect scores on pages with such regions.
  Detection F1, classification accuracy and per-region CER are unchanged.

>>> import pandas as pd
>>> row_id_column_name = "image"
>>> solution = pd.DataFrame({
...     "image": ["test.jpg"],
...     "regions": ['[{"bbox":[50,100,850,130],"type":"handwritten","language":"uk","legibility":"legible","text":"Доброго ранку"},{"bbox":[50,150,870,180],"type":"handwritten","language":"uk","legibility":"legible","text":"Сьогодні гарна погода"},{"bbox":[50,250,650,270],"type":"printed","language":"uk","legibility":"legible","text":"Завдання 1"},{"bbox":[700,50,950,250],"type":"image","language":"uk","legibility":"legible","text":""},{"bbox":[900,950,980,990],"type":"annotation","language":"uk","legibility":"legible","text":"5"}]'],
... })
>>> submission = pd.DataFrame({
...     "image": ["test.jpg"],
...     "regions": ['[{"bbox":[52,101,852,131],"type":"handwritten","language":"uk","legibility":"legible","text":"Доброго ранку"},{"bbox":[50,148,860,184],"type":"handwritten","language":"uk","legibility":"legible","text":"Сьогодні гарна пагода"},{"bbox":[48,249,649,272],"type":"handwritten","language":"uk","legibility":"legible","text":"Завдання 1"},{"bbox":[710,60,945,248],"type":"image","language":"uk","legibility":"legible","text":""}]'],
... })
>>> round(score(solution, submission, row_id_column_name), 4)
0.9348
"""

import json
import re
import pandas as pd


class ParticipantVisibleError(Exception):
    pass


# ── Text normalization ────────────────────────────────────────

# ── LaTeX command → Unicode symbol mapping ────────────────────
# Applied BEFORE Cyrillic/Latin conversion (so \pi doesn't become \рі)

_LATEX_SYMBOLS = {
    # Greek letters (common in formulas)
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
    r'\theta': 'θ', r'\vartheta': 'ϑ', r'\iota': 'ι', r'\kappa': 'κ',
    r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
    r'\pi': 'π', r'\rho': 'ρ', r'\sigma': 'σ', r'\tau': 'τ',
    r'\upsilon': 'υ', r'\phi': 'φ', r'\varphi': 'φ', r'\chi': 'χ',
    r'\psi': 'ψ', r'\omega': 'ω',
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Phi': 'Φ',
    r'\Psi': 'Ψ', r'\Omega': 'Ω',
    # Operators & relations
    r'\cdot': '·', r'\times': '×', r'\div': '÷', r'\pm': '±', r'\mp': '∓',
    r'\circ': '∘', r'\bullet': '•', r'\star': '⋆',
    r'\leq': '≤', r'\le': '≤', r'\geq': '≥', r'\ge': '≥',
    r'\neq': '≠', r'\ne': '≠', r'\approx': '≈', r'\equiv': '≡',
    r'\sim': '∼', r'\propto': '∝',
    # Arrows
    r'\rightarrow': '→', r'\to': '→', r'\leftarrow': '←',
    r'\leftrightarrow': '↔', r'\Rightarrow': '⇒', r'\Leftarrow': '⇐',
    r'\Leftrightarrow': '⇔', r'\implies': '⇒', r'\iff': '⇔',
    # Set theory
    r'\cap': '∩', r'\cup': '∪', r'\subset': '⊂', r'\supset': '⊃',
    r'\subseteq': '⊆', r'\supseteq': '⊇', r'\in': '∈', r'\notin': '∉',
    r'\emptyset': '∅', r'\varnothing': '∅',
    r'\oplus': '⊕', r'\otimes': '⊗',
    # Misc
    r'\infty': '∞', r'\partial': '∂', r'\nabla': '∇',
    r'\forall': '∀', r'\exists': '∃', r'\neg': '¬',
    r'\sqrt': '√', r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
    r'\ldots': '…', r'\dots': '…', r'\cdots': '⋯',
    # Geometry / logic symbols
    r'\therefore': '∴', r'\because': '∵',
    r'\perp': '⊥', r'\angle': '∠', r'\parallel': '∥',
    r'\square': '□', r'\Box': '□', r'\triangle': '△',
    # Up/down arrows (often used as products of reaction)
    r'\uparrow': '↑', r'\downarrow': '↓', r'\Uparrow': '⇑', r'\Downarrow': '⇓',
    # Logical operators
    r'\vee': '∨', r'\wedge': '∧', r'\lor': '∨', r'\land': '∧',
    # Set operations
    r'\setminus': '∖', r'\backslash': '\\',
    # Vertical bar variants
    r'\mid': '|', r'\lvert': '|', r'\rvert': '|', r'\Vert': '‖',
    # Floor/ceil delimiters
    r'\lfloor': '⌊', r'\rfloor': '⌋', r'\lceil': '⌈', r'\rceil': '⌉',
    # Marvosym/wasysym symbols (used in genetics for sex notation)
    r'\male': '♂', r'\female': '♀',
}

# LaTeX named functions: \sin, \cos, \ln, \lim, ... → strip backslash + trailing space
# Without trailing space "\\sin\\alpha" would normalize to "sinα" while "sin α" → "sin α"
# (mismatched). Trailing space collapses with following whitespace via _MULTI_SPACE.
_LATEX_FUNCTION_NAMES = (
    'arcsin','arccos','arctan','arcctg','arccot','arcsec','arccsc',
    'sinh','cosh','tanh','coth',
    'sin','cos','tan','cot','sec','csc','ctg',
    'liminf','limsup',
    'log','ln','lg','exp','lim','sup','inf','min','max','det','dim','gcd','lcm','mod',
    'arg','deg','hom','ker',
)
# `(?![A-Za-z])` (not `\b`) so that `\lim_{x→0}` and `\sin{x}` still match: the
# char after the command name may be `_`/`^`/`{` which `\b` would block.
_LATEX_FUNCTIONS_RE = re.compile(r'\\(' + '|'.join(_LATEX_FUNCTION_NAMES) + r')(?![A-Za-z])')

# \xrightarrow{label} / \xleftarrow{label} → arrow (label discarded)
# Some forms have optional bracketed below-label: \xrightarrow[below]{above}
_XARROW_RIGHT = re.compile(r'\\xrightarrow\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}')
_XARROW_LEFT = re.compile(r'\\xleftarrow\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}')

# Sizing commands (visual hints, no semantic value) → strip
_LATEX_SIZING = re.compile(r'\\(?:big|Big|bigg|Bigg)[lr]?\b')

# Math styles \mathrm{}, \mathbf{}, ..., \operatorname{} → strip wrapper
_MATH_STYLE = re.compile(r'\\(?:mathrm|mathbf|mathit|mathbb|mathcal|mathfrak|mathsf|mathtt|operatorname|boldsymbol|pmb)\s*\{([^{}]*)\}')

# Cancel/overset/underset: cancellation marks → keep base content
# \cancel{18} → 18 (the struck-through value is what we want to read back)
# \overset{a}{b} → b (b is the main symbol; a is a decoration like an oxidation state)
# \underset{a}{b} → b (same logic)
_CANCEL = re.compile(r'\\cancel\s*\{([^{}]*)\}')
_OVERSET = re.compile(r'\\overset\s*\{[^{}]*\}\s*\{([^{}]*)\}')
_UNDERSET = re.compile(r'\\underset\s*\{[^{}]*\}\s*\{([^{}]*)\}')
# Sort by length descending so \rightarrow matches before \right
_LATEX_COMMANDS_RE = re.compile(
    '|'.join(re.escape(k) for k in sorted(_LATEX_SYMBOLS.keys(), key=len, reverse=True))
)

# \text{...} → content (strip LaTeX text wrapper)
_LATEX_TEXT_WRAPPER = re.compile(r'\\text\{([^}]*)\}')
# \left( \right) → ( )
_LATEX_LEFT_RIGHT = re.compile(r'\\(left|right)\s*([()|\[\]{}.])')
# LaTeX spacing commands → space or nothing
_LATEX_SPACING = re.compile(r'\\[,;:!]|\\quad|\\qquad|\\hspace\{[^}]*\}')

# Multiplication sign normalization: * and · → · (middle dot)
_MULT_SIGNS = re.compile(r'[*∗⋅]')  # asterisk, combining asterisk, dot operator

# Latin → Cyrillic lookalike mapping (lowercase + uppercase)
_LATIN_TO_CYRILLIC = {
    'a': 'а', 'c': 'с', 'e': 'е', 'i': 'і', 'o': 'о',
    'p': 'р', 'x': 'х', 'y': 'у',
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т', 'X': 'Х',
}

# Unicode superscript/subscript → ASCII
_SUPERSCRIPTS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ', '0123456789+-=()n')
_SUBSCRIPTS = str.maketrans('₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎', '0123456789+-=()')

# All dash-like characters → hyphen-minus
_DASHES = re.compile(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]')

# Filler dashes/underscores (3+ repeated) → normalized form
_FILLERS = re.compile(r'[_\-]{3,}')

# Strikethrough: ~~old~~{new} → new (correction replaces struck-through text)
_STRIKETHROUGH_CORRECTION = re.compile(r'~~.*?~~\{(.*?)\}')
# Strikethrough: ~~text~~ → text (standalone, no correction)
_STRIKETHROUGH = re.compile(r'~~(.*?)~~')

# Multiple whitespace → single space
_MULTI_SPACE = re.compile(r'[ \t\u00A0\u2000-\u200B\u3000]+')

# Spaces inside brackets: "( text )" → "(text)"
_SPACE_IN_PARENS = re.compile(r'\(\s+')
_SPACE_IN_PARENS_R = re.compile(r'\s+\)')

# Space immediately before "(" or before sub/superscript markers — strips
# the trailing space introduced by `\sin ` / `\lim ` etc. so:
#   "\sin(x)" → "sin (x)" → "sin(x)" (matches plain "sin(x)")
#   "\lim_{x→0}" → "lim _{x→0}" → "lim_{x→0}"
_SPACE_BEFORE_PAREN = re.compile(r' +\(')
_SPACE_BEFORE_SUBSUPER = re.compile(r' +([_^])')

# LaTeX braces: x_{3} → x_3, x^{2} → x^2, S_{повн} → S_повн
_LATEX_BRACE = re.compile(r'([_^])\{([^}]+)\}')

# LaTeX table environments → PSV
# Covers: array, tabular, matrix/pmatrix/bmatrix/vmatrix/Vmatrix/smallmatrix,
# aligned/align/alignat/gathered/cases (multi-row alignment envs that share
# the same \\ row-separator + & column-separator syntax).
_LATEX_TABLE_ENV = re.compile(r'\\begin\{(?:array|tabular|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix|aligned|align|alignat|gathered|cases|split)\*?\}(?:\{[^}]*\})?\s*')
_LATEX_TABLE_ENV_END = re.compile(r'\s*\\end\{(?:array|tabular|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix|aligned|align|alignat|gathered|cases|split)\*?\}')
_LATEX_TABLE_ROW_SEP = re.compile(r'\s*\\\\\s*')
_LATEX_TABLE_COL_SEP = re.compile(r'\s*&\s*')

# Table horizontal line decorations are visual-only, no semantic value.
# `\hline` is a no-arg command; `\cline{2-4}` takes a span argument.
_TABLE_LINES = re.compile(r'\\hline\b|\\cline\s*\{[^{}]*\}')

# `\phantom{x}` renders invisible — no semantic value, strip.
_PHANTOM = re.compile(r'\\phantom\s*\{[^{}]*\}')

# `\underline{x}` → x  (visual underline; same treatment as `\bar` etc.)
_UNDERLINE = re.compile(r'\\underline\s*\{([^{}]*)\}')

# Student-style row separator inside plain parens: `(a b \n c d)` means a 2D
# matrix (one row per `\n`). Normalize to multiline PSV like the existing
# `(a; b)` rule does for column matrices.
# Match an opening paren, body containing at least one literal `\n`, closing paren.
_PAREN_NEWLINE_ROWS = re.compile(r'\(([^()]*\\n[^()]*)\)')

# Quotes normalization
_QUOTES_DOUBLE = re.compile(r'["\u201C\u201D\u201E\u00AB\u00BB\u2033]')
_QUOTES_SINGLE = re.compile(r"['\u2018\u2019\u02BC\u0027\u2032]")

# Pipe-separated values: strip whitespace around pipes
_PSV_PIPE = re.compile(r'\s*\|\s*')

# \frac{a}{b} → a/b  (applied iteratively for nested fractions)
_FRAC = re.compile(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}')

# \sqrt{...} → √...  (after \sqrt → √ symbol mapping, strip the trailing argument braces)
_SQRT_BRACES = re.compile(r'√\s*\{([^{}]*)\}')

# Arrow decorations: \overrightarrow{X} → →X, \overleftarrow{X} → ←X, \vec{X} → →X
_ARROW_RIGHT_DECOR = re.compile(r'\\(?:overrightarrow|vec)\s*\{([^{}]*)\}')
_ARROW_LEFT_DECOR = re.compile(r'\\overleftarrow\s*\{([^{}]*)\}')

# Decorators: \bar{x}, \hat{x}, \overline{x}, \widetilde{x}, \widehat{x}, \dot{x}, \ddot{x} → strip wrapper
_DECORATOR = re.compile(r'\\(?:bar|hat|overline|widetilde|widehat|dot|ddot)\s*\{([^{}]*)\}')

# Combining diacritical marks (U+0300-036F) + symbol combining (U+20D0-20FF, includes \vec arrow ⃗)
# Strip after decorators to symmetrize: \bar{x} (stripped to x) ↔ x̄ (Unicode combining macron stripped to x)
_COMBINING_MARKS = re.compile(r'[̀-ͯ⃐-⃿]')

# Plain (a; b; c) column matrix → PSV (a\nb\nc)  — only when paren content is purely semicolon-separated
_PAREN_SEMI = re.compile(r'\(([^();]+(?:\s*;\s*[^();]+)+)\)')

# Plain pipe determinant: | a b | | c d | | e f | (2+ pipe-bounded segments) → PSV
_PIPE_MATRIX = re.compile(r'\|\s*([^|\n]+?)\s*\|(?:\s*\|\s*[^|\n]+?\s*\|)+')


def _normalize_text(text: str, region_type: str = "handwritten") -> str:
    """
    Normalize text before CER comparison.
    Applied identically to both GT and prediction.

    >>> _normalize_text("Доброго ранку")
    'Доброго ранку'
    >>> _normalize_text("cocна")  # Latin 'c','o' → Cyrillic
    'сосна'
    >>> _normalize_text("тире — довге")  # em-dash → hyphen
    'тире - довге'
    >>> _normalize_text("~~закреслено~~ слово")
    'закреслено слово'
    >>> _normalize_text("x_{3} + y^{2}", region_type="formula")
    'х_3 + у^2'
    >>> _normalize_text("A | B | C", region_type="table")
    'А|В|С'
    >>> _normalize_text("x² + y₃", region_type="formula")
    'х^2 + у_3'
    >>> _normalize_text('«Привіт»')
    '"Привіт"'
    >>> _normalize_text("( дужки )")
    '(дужки)'
    >>> _normalize_text("_____")
    '___'
    >>> _normalize_text("cat") == _normalize_text("сat")  # Latin/Cyrillic forgiven
    True
    >>> _normalize_text("π r^2", region_type="formula")  # Unicode π unchanged
    'π r^2'
    >>> _normalize_text("2 * 3 = 6", region_type="formula")  # * → ·
    '2 · 3 = 6'
    >>> _normalize_text("x² + y₃", region_type="formula")  # Unicode super/sub
    'х^2 + у_3'
    >>> _normalize_text("H_{2}SO_{4}", region_type="formula")  # single-char braces
    'Н_2SО_4'
    >>> _normalize_text(r"\\frac{1}{2}", region_type="formula")  # frac → plain
    '1/2'
    >>> _normalize_text(r"a/b", region_type="formula") == _normalize_text(r"\\frac{a}{b}", region_type="formula")
    True
    >>> _normalize_text(r"\\sqrt{169}", region_type="formula")
    '√169'
    >>> _normalize_text(r"\\bar{x}", region_type="formula")
    'х'
    >>> _normalize_text("(1; -1)", region_type="formula")  # column matrix → PSV
    '1\\n-1'
    >>> _normalize_text("| 3 2 | | -1 1 |", region_type="formula")  # pipe determinant → PSV
    '3 2\\n-1 1'
    >>> _normalize_text(r"\\overrightarrow{AB}", region_type="formula")  # arrow notation
    '→АВ'
    >>> _normalize_text(r"\\vec{a}", region_type="formula")  # vec also → arrow
    '→а'
    >>> _normalize_text(r"\\therefore \\overrightarrow{AB} \\perp \\overrightarrow{AC}", region_type="formula")
    '∴ →АВ ⊥ →АС'
    >>> _normalize_text("x̄", region_type="formula") == _normalize_text(r"\\bar{x}", region_type="formula")
    True
    """
    if not text:
        return ""

    # 1. Strikethrough: ~~old~~{new} → new (must come before plain ~~)
    text = _STRIKETHROUGH_CORRECTION.sub(r'\1', text)
    text = _STRIKETHROUGH.sub(r'\1', text)

    # 2. LaTeX normalization (BEFORE Cyrillic conversion — so \pi doesn't become \рі)
    if region_type in ("formula", "table"):
        # LaTeX table environments → PSV (must be before symbol conversion)
        text = _LATEX_TABLE_ENV.sub('', text)
        text = _LATEX_TABLE_ENV_END.sub('', text)
        text = _LATEX_TABLE_ROW_SEP.sub('\n', text)
        text = _LATEX_TABLE_COL_SEP.sub('|', text)
        # Visual-only table decorations: \hline, \cline{2-4}, \phantom{x}
        text = _TABLE_LINES.sub('', text)
        text = _PHANTOM.sub('', text)
        # \underline{x} → x (treat like \bar)
        text = _UNDERLINE.sub(r'\1', text)
        # \text{...} → content
        text = _LATEX_TEXT_WRAPPER.sub(r'\1', text)
        # \left( \right) → ( )
        text = _LATEX_LEFT_RIGHT.sub(r'\2', text)
        # Sizing hints \big, \Bigg, \bigl, \biggr, … → strip
        text = _LATEX_SIZING.sub('', text)
        # Math styles \mathrm{}, \mathbf{}, ..., \operatorname{} → strip wrapper (iterate for nested)
        prev = None
        while text != prev:
            prev = text
            text = _MATH_STYLE.sub(r'\1', text)
        # \xrightarrow[below]{above} / \xleftarrow → → / ←
        text = _XARROW_RIGHT.sub('→', text)
        text = _XARROW_LEFT.sub('←', text)
        # \cancel{x} → x; \overset{a}{b} / \underset{a}{b} → b
        # Iterate because these can be nested (e.g. \overset{\overset{...}{|}}{X})
        prev = None
        while text != prev:
            prev = text
            text = _CANCEL.sub(r'\1', text)
            text = _OVERSET.sub(r'\1', text)
            text = _UNDERSET.sub(r'\1', text)
        # LaTeX spacing → single space
        text = _LATEX_SPACING.sub(' ', text)
        # \frac{a}{b} → a/b  (iterate for nested fractions)
        prev = None
        while text != prev:
            prev = text
            text = _FRAC.sub(r'\1/\2', text)
        # Arrow decorations: \overrightarrow{X}/\vec{X} → →X, \overleftarrow{X} → ←X
        text = _ARROW_RIGHT_DECOR.sub(r'→\1', text)
        text = _ARROW_LEFT_DECOR.sub(r'←\1', text)
        # Decorators: \bar{x}, \hat{x}, \overline{x} → strip wrapper
        text = _DECORATOR.sub(r'\1', text)
        # Named functions: \sin, \cos, \log, \lim … → "sin ", "cos ", … (trailing space
        # ensures "\\sin\\alpha" → "sin α" matches "sin α"; collapsed later)
        text = _LATEX_FUNCTIONS_RE.sub(r'\1 ', text)
        # LaTeX symbols → Unicode (\pi → π, \cdot → ·, etc.)
        text = _LATEX_COMMANDS_RE.sub(lambda m: _LATEX_SYMBOLS[m.group()], text)
        # \sqrt argument braces: √{169} → √169  (after \sqrt → √ mapping above)
        text = _SQRT_BRACES.sub(r'√\1', text)
        # Strip Unicode combining marks (symmetric with decorator wrapper-stripping above)
        # e.g. x̄ (x + U+0304 macron) → x; a⃗ (a + U+20D7 right arrow) → a
        text = _COMBINING_MARKS.sub('', text)
        # Multiplication signs: * ∗ ⋅ → · (middle dot)
        text = _MULT_SIGNS.sub('·', text)
        # Unicode superscripts → ^N
        converted = []
        for ch in text:
            if ch in '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ':
                converted.append('^' + ch.translate(_SUPERSCRIPTS))
            elif ch in '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎':
                converted.append('_' + ch.translate(_SUBSCRIPTS))
            else:
                converted.append(ch)
        text = ''.join(converted)
        # Braces: x_{3} → x_3, S_{повн} → S_повн
        text = _LATEX_BRACE.sub(r'\1\2', text)
        # Plain (a; b; c) column matrix → PSV
        text = _PAREN_SEMI.sub(lambda m: '\n'.join(x.strip() for x in m.group(1).split(';')), text)
        # Plain `(a b \n c d)` 2D matrix (student notation) → multiline rows
        text = _PAREN_NEWLINE_ROWS.sub(lambda m: '\n'.join(x.strip() for x in m.group(1).split('\\n')), text)
        # Plain pipe determinant: | a b | | c d | | e f | → PSV (each row on its own line)
        text = _PIPE_MATRIX.sub(lambda m: '\n'.join(re.findall(r'\|\s*([^|\n]+?)\s*\|', m.group(0))), text)

    # 3. Cyrillic/Latin lookalikes → Cyrillic
    text = ''.join(_LATIN_TO_CYRILLIC.get(ch, ch) for ch in text)

    # 4. Dashes → hyphen-minus
    text = _DASHES.sub('-', text)

    # 5. Filler dashes/underscores → 3 chars
    text = _FILLERS.sub('___', text)

    # 6. Quotes
    text = _QUOTES_DOUBLE.sub('"', text)
    text = _QUOTES_SINGLE.sub("'", text)

    # 7. Whitespace: NBSP and exotic spaces → regular space, collapse multiples
    text = _MULTI_SPACE.sub(' ', text)

    # 8. Spaces inside parentheses
    text = _SPACE_IN_PARENS.sub('(', text)
    text = _SPACE_IN_PARENS_R.sub(')', text)
    # 8b. Strip trailing space introduced by `\sin ` / `\lim ` etc. when
    # followed by "(" or sub/superscript marker.
    text = _SPACE_BEFORE_PAREN.sub('(', text)
    text = _SPACE_BEFORE_SUBSUPER.sub(r'\1', text)

    # 9. Table-specific: PSV cleanup (LaTeX table already converted in step 2)
    if region_type == "table":
        text = _PSV_PIPE.sub('|', text)
        lines = text.split('\n')
        lines = [line.strip('|').strip() for line in lines]
        text = '\n'.join(line for line in lines if line)

    # 10. Strip leading/trailing whitespace
    text = text.strip()

    return text


# ── Levenshtein distance (pure Python, no external deps) ──────

def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (c1 != c2),
            ))
        prev = curr
    return prev[-1]


# ── IoU ───────────────────────────────────────────────────────

def _compute_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


# ── Greedy IoU matching ───────────────────────────────────────

def _greedy_match(gt_regions, pred_regions, threshold=0.5):
    pairs = []
    for gi, g in enumerate(gt_regions):
        for pi, p in enumerate(pred_regions):
            iou = _compute_iou(g["bbox"], p["bbox"])
            if iou >= threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda x: -x[0])

    matched = []
    used_gt, used_pred = set(), set()
    for iou, gi, pi in pairs:
        if gi not in used_gt and pi not in used_pred:
            matched.append((gi, pi))
            used_gt.add(gi)
            used_pred.add(pi)

    unmatched_gt = [i for i in range(len(gt_regions)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_regions)) if i not in used_pred]
    return matched, unmatched_gt, unmatched_pred


# ── Scorable check ────────────────────────────────────────────

def _is_scorable(region):
    if region.get("type", "handwritten") in ("image", "graph"):
        return False
    if region.get("language", "uk") == "other":
        return False
    if region.get("legibility", "legible") == "illegible":
        return False
    return True


# ── Page text builder ─────────────────────────────────────────

def _build_page_text(regions, normalize=False, drop_indices=None):
    """Build concatenated page text from regions.

    `drop_indices` (optional): set of region indices to additionally exclude
    beyond the standard _is_scorable filter. Used on the prediction side to
    drop pred regions that match a non-scorable GT region (otherwise their
    text inflates pred_page asymmetrically vs gt_page).
    """
    drop = drop_indices or set()
    scorable = [
        r for i, r in enumerate(regions)
        if _is_scorable(r) and i not in drop
    ]
    # Bucketed reading order: cluster regions whose center_y is within ~15px
    # (half a typical handwritten line height), then left-to-right by center_x.
    # Stabilizes ordering when detection splits one GT line into multiple bboxes
    # with slightly different top_y values, which would otherwise scramble the
    # page-level text concatenation.
    scorable.sort(key=lambda r: (
        ((r["bbox"][1] + r["bbox"][3]) / 2) // 15,
        (r["bbox"][0] + r["bbox"][2]) / 2,
    ))
    if normalize:
        return "\n".join(
            _normalize_text(r.get("text", ""), r.get("type", "handwritten"))
            for r in scorable
        )
    return "\n".join(r.get("text", "") for r in scorable)


# ── Parse regions JSON ────────────────────────────────────────

VALID_TYPES = {"handwritten", "printed", "formula", "table", "annotation", "image", "graph"}


def _parse_regions(regions_str, image_name, is_submission=True):
    label = "Submission" if is_submission else "Solution"
    if pd.isna(regions_str) or regions_str == "":
        return []
    try:
        regions = json.loads(regions_str)
    except (json.JSONDecodeError, TypeError) as e:
        raise ParticipantVisibleError(
            f'{label} for image "{image_name}": invalid JSON in regions column. Error: {e}'
        )
    if not isinstance(regions, list):
        raise ParticipantVisibleError(
            f'{label} for image "{image_name}": regions must be a JSON list, got {type(regions).__name__}'
        )
    for i, r in enumerate(regions):
        if not isinstance(r, dict):
            raise ParticipantVisibleError(
                f'{label} for image "{image_name}", region {i}: must be a JSON object'
            )
        if "bbox" not in r:
            raise ParticipantVisibleError(
                f'{label} for image "{image_name}", region {i}: missing "bbox" field'
            )
        bbox = r["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ParticipantVisibleError(
                f'{label} for image "{image_name}", region {i}: bbox must be [x1, y1, x2, y2]'
            )
        rtype = r.get("type", "handwritten")
        if is_submission and rtype not in VALID_TYPES:
            raise ParticipantVisibleError(
                f'{label} for image "{image_name}", region {i}: invalid type "{rtype}". '
                f'Must be one of: {", ".join(sorted(VALID_TYPES))}'
            )
        # Defaults
        r.setdefault("type", "handwritten")
        r.setdefault("language", "uk")
        r.setdefault("legibility", "legible")
        r.setdefault("text", "")
    return regions


# ── Main scoring function ────────────────────────────────────

def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    w_det: float = 0.15,
    w_cls: float = 0.05,
    w_cer: float = 0.30,
    w_page: float = 0.50,
) -> float:
    """
    Ukrainian Handwritten Text Recognition (HTR) Competition Metric.

    Evaluates end-to-end document understanding: region detection,
    classification, and text transcription.

    Score = 0.15 * Detection_F1 + 0.05 * ClassAcc + 0.30 * (1-CER) + 0.50 * (1-PageCER)

    Components:
      - Detection F1 (0.15): type-agnostic bbox matching at IoU >= 0.5
      - Classification Accuracy (0.05): correct region type among IoU-matched pairs
      - CER (0.30): per-region Character Error Rate on matched scorable regions
      - Page CER (0.50): full-page text comparison, agnostic to bbox granularity

    Score range: 0.0 to 1.0 (higher is better).

    Submission: CSV with columns `image` and `regions`.
    `regions` is a JSON list of detected regions per image:
        [{"bbox": [x1,y1,x2,y2], "type": "handwritten", "text": "..."}]

    Region types: handwritten, printed, formula, table, annotation, image, graph.
    Use [] for images with no detections. All test images must be present.

    Text normalization (applied to both GT and predictions before CER):
      - Cyrillic/Latin lookalikes unified (Latin c → Cyrillic с)
      - Dash types unified (em/en-dash → hyphen)
      - Whitespace collapsed, quotes normalized
      - Strikethrough markers removed: ~~old~~{new} → new
      - Formula: x_{3} → x_3, x² → x^2
      - Table: whitespace around pipes stripped

    Regions excluded from CER (GT attributes, not required from participants):
      type=image/graph, language=other, legibility=illegible

    >>> import pandas as pd
    >>> sol = pd.DataFrame({"image": ["a.jpg"], "regions": ['[{"bbox":[0,0,100,50],"type":"handwritten","text":"hello"}]']})
    >>> sub = pd.DataFrame({"image": ["a.jpg"], "regions": ['[{"bbox":[0,0,100,50],"type":"handwritten","text":"hello"}]']})
    >>> score(sol, sub, "image")
    1.0
    """
    # Validate columns
    if "regions" not in submission.columns:
        raise ParticipantVisibleError(
            'Submission must have a "regions" column containing JSON-encoded region predictions.'
        )
    if "regions" not in solution.columns:
        raise ParticipantVisibleError('Solution is missing "regions" column.')

    # Check all solution images are in submission
    sol_images = set(solution[row_id_column_name])
    sub_images = set(submission[row_id_column_name])
    missing = sol_images - sub_images
    if missing:
        examples = sorted(missing)[:5]
        raise ParticipantVisibleError(
            f'Submission is missing {len(missing)} image(s). Examples: {examples}. '
            f'Include all test images, even those with no predictions (use empty regions: []).'
        )

    # Build lookup
    sub_lookup = {}
    for _, row in submission.iterrows():
        img = row[row_id_column_name]
        sub_lookup[img] = _parse_regions(row["regions"], img, is_submission=True)

    # Accumulators
    all_det_tp = 0
    all_det_fp = 0
    all_det_fn = 0
    all_class_correct = 0
    all_class_total = 0
    all_cer_values = []
    all_page_cers = []

    for _, row in solution.iterrows():
        img = row[row_id_column_name]
        gt = _parse_regions(row["regions"], img, is_submission=False)
        pred = sub_lookup.get(img, [])

        # Match by IoU
        matched, unmatched_gt, unmatched_pred = _greedy_match(gt, pred, threshold=0.5)

        # Detection F1 (type-agnostic: measures bbox quality only)
        all_det_tp += len(matched)
        all_det_fp += len(unmatched_pred)
        all_det_fn += len(unmatched_gt)

        # Classification Accuracy
        for gi, pi in matched:
            all_class_total += 1
            if gt[gi]["type"] == pred[pi]["type"]:
                all_class_correct += 1

        # CER (per-region, with text normalization)
        for gi, pi in matched:
            if _is_scorable(gt[gi]):
                rtype = gt[gi].get("type", "handwritten")
                gt_text = _normalize_text(gt[gi].get("text", ""), rtype)
                pred_text = _normalize_text(pred[pi].get("text", ""), rtype)
                cer_i = _levenshtein(pred_text, gt_text) / max(len(gt_text), 1)
                all_cer_values.append(cer_i)

        # Page CER (with text normalization).
        # Drop pred regions matched to a non-scorable GT region — otherwise
        # their text would inflate pred_page while GT side excludes them.
        pred_drop = {pi for gi, pi in matched if not _is_scorable(gt[gi])}
        gt_page = _build_page_text(gt, normalize=True)
        pred_page = _build_page_text(pred, normalize=True, drop_indices=pred_drop)
        if len(gt_page) > 0:
            page_cer_i = _levenshtein(pred_page, gt_page) / len(gt_page)
            all_page_cers.append(page_cer_i)

    # Aggregate
    det_prec = all_det_tp / max(all_det_tp + all_det_fp, 1)
    det_rec = all_det_tp / max(all_det_tp + all_det_fn, 1)
    det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-9)

    class_acc = all_class_correct / max(all_class_total, 1)

    # Default CER = 1.0 (worst) when no regions matched — prevents free score for empty submissions
    cer = sum(all_cer_values) / len(all_cer_values) if all_cer_values else 1.0
    page_cer = sum(all_page_cers) / len(all_page_cers) if all_page_cers else 1.0

    final_score = (
        w_det * det_f1
        + w_cls * class_acc
        + w_cer * max(0.0, 1.0 - cer)
        + w_page * max(0.0, 1.0 - page_cer)
    )

    return float(final_score)

def score_detailed(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    w_det: float = 0.15,
    w_cls: float = 0.05,
    w_cer: float = 0.30,
    w_page: float = 0.50,
) -> dict:
    """
    Same metric as `score()` but returns a dict with all component scores.

    Use this locally to debug your submission — you will see which component
    (detection, classification, per-region CER, or page CER) is dragging
    the score down.

    Not used by Kaggle — Kaggle calls `score()` which returns a single float.
    """
    if "regions" not in submission.columns:
        raise ParticipantVisibleError(
            'Submission must have a "regions" column containing JSON-encoded region predictions.'
        )
    if "regions" not in solution.columns:
        raise ParticipantVisibleError('Solution is missing "regions" column.')

    sol_images = set(solution[row_id_column_name])
    sub_images = set(submission[row_id_column_name])
    missing = sol_images - sub_images
    if missing:
        examples = sorted(missing)[:5]
        raise ParticipantVisibleError(
            f'Submission is missing {len(missing)} image(s). Examples: {examples}.'
        )

    sub_lookup = {
        row[row_id_column_name]: _parse_regions(row["regions"], row[row_id_column_name], is_submission=True)
        for _, row in submission.iterrows()
    }

    all_det_tp = all_det_fp = all_det_fn = 0
    all_class_correct = all_class_total = 0
    all_cer_values, all_page_cers = [], []

    for _, row in solution.iterrows():
        img = row[row_id_column_name]
        gt = _parse_regions(row["regions"], img, is_submission=False)
        pred = sub_lookup.get(img, [])
        matched, unmatched_gt, unmatched_pred = _greedy_match(gt, pred, threshold=0.5)

        all_det_tp += len(matched)
        all_det_fp += len(unmatched_pred)
        all_det_fn += len(unmatched_gt)

        for gi, pi in matched:
            all_class_total += 1
            if gt[gi]["type"] == pred[pi]["type"]:
                all_class_correct += 1

        for gi, pi in matched:
            if _is_scorable(gt[gi]):
                rtype = gt[gi].get("type", "handwritten")
                gt_text = _normalize_text(gt[gi].get("text", ""), rtype)
                pred_text = _normalize_text(pred[pi].get("text", ""), rtype)
                cer_i = _levenshtein(pred_text, gt_text) / max(len(gt_text), 1)
                all_cer_values.append(cer_i)

        # Drop pred regions matched to non-scorable GT (avoid asymmetric pred_page inflation)
        pred_drop = {pi for gi, pi in matched if not _is_scorable(gt[gi])}
        gt_page = _build_page_text(gt, normalize=True)
        pred_page = _build_page_text(pred, normalize=True, drop_indices=pred_drop)
        if len(gt_page) > 0:
            all_page_cers.append(_levenshtein(pred_page, gt_page) / len(gt_page))

    det_prec = all_det_tp / max(all_det_tp + all_det_fp, 1)
    det_rec = all_det_tp / max(all_det_tp + all_det_fn, 1)
    det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-9)
    class_acc = all_class_correct / max(all_class_total, 1)
    cer = sum(all_cer_values) / len(all_cer_values) if all_cer_values else 1.0
    page_cer = sum(all_page_cers) / len(all_page_cers) if all_page_cers else 1.0

    composite = (
        w_det * det_f1
        + w_cls * class_acc
        + w_cer * max(0.0, 1.0 - cer)
        + w_page * max(0.0, 1.0 - page_cer)
    )

    return {
        "composite_score": float(composite),
        "detection_f1": float(det_f1),
        "detection_precision": float(det_prec),
        "detection_recall": float(det_rec),
        "classification_accuracy": float(class_acc),
        "region_cer": float(cer),
        "page_cer": float(page_cer),
        "n_images": int(len(solution)),
        "n_matched_regions": int(all_det_tp),
        "n_false_positives": int(all_det_fp),
        "n_false_negatives": int(all_det_fn),
    }


# ── CLI for local debugging ───────────────────────────────────
# NOTE: kept as a function (not a top-level __main__ block) so that
# importing this file in a notebook / Kaggle metric uploader does not
# trigger argparse on a kernel-launcher cmdline.

def _cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="RUKOPYS scoring — run locally to see component breakdown",
    )
    parser.add_argument("--solution", required=True, help="Path to ground-truth CSV (image, regions)")
    parser.add_argument("--submission", required=True, help="Path to your submission CSV (image, regions)")
    parser.add_argument("--row-id", default="image", help="Row ID column name (default: image)")
    args = parser.parse_args()

    sol = pd.read_csv(args.solution)
    sub = pd.read_csv(args.submission)

    r = score_detailed(sol, sub, args.row_id)

    print()
    print(f"  Images evaluated       : {r['n_images']}")
    print(f"  Matched regions (IoU≥.5): {r['n_matched_regions']}")
    print(f"  False positives        : {r['n_false_positives']}")
    print(f"  False negatives        : {r['n_false_negatives']}")
    print()
    print(f"  Detection F1           : {r['detection_f1']:.4f}   (precision {r['detection_precision']:.3f} / recall {r['detection_recall']:.3f})")
    print(f"  Classification accuracy: {r['classification_accuracy']:.4f}")
    print(f"  Region CER             : {r['region_cer']:.4f}   → score {1-r['region_cer']:.4f}")
    print(f"  Page CER               : {r['page_cer']:.4f}   → score {1-r['page_cer']:.4f}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Composite score        : {r['composite_score']:.4f}")
    print()


if __name__ == "__main__":
    import sys
    # Only auto-run CLI if --solution arg present; skips Jupyter/Colab
    # where __name__ == "__main__" but sys.argv is a kernel launcher.
    if any(a == "--solution" for a in sys.argv[1:]):
        _cli_main()
