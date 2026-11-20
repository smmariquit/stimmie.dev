#!/usr/bin/env python3
# scripts/get14-drills.py

# Quick drills for /get14/drills: small arithmetic, tracing skills, facts to
# memorize and language traps. Every answer comes from a real run.
# Needs python3, node and gcc.
#
# Play in a terminal:  python3 scripts/get14-drills.py [count] [topics]
# Rebuild the bank:    python3 scripts/get14-drills.py --dump src/data/get14-drills.json
#                      npx biome format --write src/data/get14-drills.json
#
# Topics: mod, cdiv, bits, prec, base, py, js, c, ascii, memo, traps
# Groups: math, skills, memory, traps

import contextlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import time

R = random.randint
WORDS = ["python", "banana", "coding", "chamber", "quarter", "reprise", "decibel"]
LABEL = {"python": "Python", "js": "JS", "c": "C", "any": "any"}
TMP = tempfile.mkdtemp(prefix="get14-drills-")


def nums(n=4):
    return random.sample(range(1, 10), n)


# C and JS integer math rounds toward zero.
def trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def trunc_mod(a, b):
    return a - b * trunc_div(a, b)


assert (trunc_div(-7, 2), trunc_mod(-7, 2), trunc_mod(7, -3)) == (-3, -1, 1)


def run_py(code):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})
    return out.getvalue().strip()


def run_js(code):
    done = subprocess.run(["node", "-e", code], capture_output=True, text=True)
    return done.stdout.strip()


def run_c(body):
    src = os.path.join(TMP, "q.c")
    exe = os.path.join(TMP, "q")
    with open(src, "w") as f:
        f.write(
            "#include <limits.h>\n#include <stdio.h>\n#include <string.h>\n"
            f"int main(void) {{ {body} return 0; }}\n"
        )
    subprocess.run(["gcc", "-w", "-o", exe, src], check=True)
    return subprocess.run([exe], capture_output=True, text=True).stdout.strip()


RUN = {"python": run_py, "js": run_js, "c": run_c}


# ── math ─────────────────────────────────────────────────────────────


def mod():
    a, b = R(-20, 40), random.choice([2, 3, 4, 5, 6, 7, 8, 9, -3, -4])
    op = random.choice(["%", "//"])
    return "python", f"{a} {op} {b}", None, str(eval(f"{a} {op} {b}"))


def cdiv():
    a, b = R(-30, 40), random.choice([2, 3, 4, 5, 6, 7, -3, -4])
    if random.random() < 0.5:
        return "c", f"{a} / {b}   (both int)", None, str(trunc_div(a, b))
    return random.choice(["c", "js"]), f"{a} % {b}", None, str(trunc_mod(a, b))


def bits():
    kind = random.choice(["&", "|", "^", "<<", ">>", "~"])
    if kind == "<<":
        a, b = R(1, 13), R(1, 4)
    elif kind == ">>":
        a, b = R(8, 120), R(1, 3)
    elif kind == "~":
        a = R(0, 20)
        return "any", f"~{a}", None, str(~a)
    else:
        a, b = R(1, 15), R(1, 15)
    return "any", f"{a} {kind} {b}", None, str(eval(f"{a} {kind} {b}"))


def prec():
    a, b, c_ = R(2, 12), R(2, 9), R(2, 7)
    expr = random.choice(
        [
            f"{a} + {b} * {c_}",
            f"{a} - {b} % {c_}",
            f"({a} + {b}) // {c_}",
            f"{a} * {b} // {c_}",
            f"{a} % {b} * {c_}",
            f"{a} // {b} * {b} + {a} % {b}",
            f"2 ** {R(1, 3)} ** 2",
            f"-{R(2, 6)} ** 2",
            f"{a} + {b} << 1",
            f"{a} & {b} + 1",
        ]
    )
    return "python", expr, None, str(eval(expr))


def base():
    kind = R(1, 4)
    n = R(5, 63)
    if kind == 1:
        return "any", f"{bin(n)} in decimal", None, str(n)
    if kind == 2:
        n = R(10, 255)
        return "any", f"0x{n:X} in decimal", None, str(n)
    if kind == 3:
        return "any", f"{n} in binary (no 0b)", None, f"{n:b}"
    return "any", f"0o{n:o} in decimal", None, str(n)


# ── skills ───────────────────────────────────────────────────────────


def py():
    w = random.choice(WORDS)
    a = nums()
    i, j, k, n = R(0, 2), R(3, 5), R(1, 3), R(4, 7)
    x, y = (repr(v) for v in random.sample([0, 5, "hi", [], [0], None, 3, 0.0], 2))
    ch = random.choice(w)
    alias = "b = a is the same list, so both names see the change."
    neg = "A negative index counts from the end, and -1 is the last item."
    code, rule = random.choice(
        [
            (f's = "{w}"; print(s[{i}:{j}])', "s[a:b] starts at index a and stops before index b."),
            (f's = "{w}"; print(s[::-1])', "A step of -1 walks backward, so it reverses."),
            (f's = "{w}"; print(s[::2])', "A step of 2 takes every other character, starting at index 0."),
            (f's = "{w}"; print(s[{i}::2])', "The slice starts at the first number, then jumps by the step."),
            (f's = "{w}"; print(s[-{k}:])', neg),
            (f's = "{w}"; print(s[:-{k}])', neg),
            (f"a = {a}; print(a[-{k}], a[1:3])", neg),
            (f"print(list(range({i}, {j + 6}, {k + 1})))", "range stops before the end value."),
            (f"print(list(range({j + 4}, {i}, -{k + 1})))", "A negative step counts down and still stops before the end value."),
            (f"print(len(range({i}, {j + 6}, {k + 1})))", "Count the values, and remember range stops before the end."),
            (f"a = {a}; b = a; b.append(9); print(len(a))", alias),
            (f"a = {a}; b = a[:]; b.append(9); print(len(a))", "a[:] makes a copy, so a doesn't change."),
            (f"a = {a}; b = a; b = b + [9]; print(len(a))", "b + [9] builds a new list, so a doesn't change."),
            (f"a = {a}; x = a.pop(0); print(x, a)", "pop(0) removes and returns the first item."),
            (f"a = {a}; x = a.pop(); print(x, len(a))", "pop() removes and returns the last item."),
            (f"a = {a}; a.insert(1, 0); print(a)", "insert(1, 0) puts 0 at index 1 and pushes the rest right."),
            (f"a = {a}; print(sorted(a)[1], a[1])", "sorted returns a new list and leaves a alone."),
            (f'print("{w}".count("{ch}"), "{w}".find("{ch}"))', "find returns the first index, or -1 when it's missing."),
            (f'print("ab" * {k + 1})', "A string times n repeats it n times."),
            (f'print("-".join("{w[:3]}"))', "join puts the separator between the items."),
            (f'print("{w[:2]},{w[2:4]},{w[4:]}".split(",")[{i}])', "split makes a list, then the index picks one piece."),
            (f"print({x} or {y})", "or returns the first truthy value, or the last value."),
            (f"print({x} and {y})", "and returns the first falsy value, or the last value."),
            (f"print([v * {k + 1} for v in range({n}) if v % 2])", "The if keeps only odd v here."),
            (f"print(sum(v for v in range({n}) if v % 2 == 0))", "range stops before the end, and the if keeps even v."),
            (f'd = {{"a": {k}}}; d["b"] = d.get("b", 0) + {n}; print(d["b"], len(d))', "get returns the default when the key is missing."),
            (f"x = {n}; x += x // 2; x *= 2; print(x)", "Update x one statement at a time and write each value down."),
            (f"f = lambda n: 1 if n == 0 else n * f(n - 1); print(f({k + 2}))", "Trace the calls down to the base case, then multiply back up."),
            (f"print(int(-{k}.7), -{k}.7 // 1)", "int() cuts toward zero, and // rounds down."),
            (f"print({2 * n + 1} / 2, {2 * n + 1} // 2)", "/ always gives a float, and // rounds down."),
            (f"print(str({k}) + str({n}), {k} + {n})", "+ joins strings and adds numbers."),
            (f"a = {a}; a[1], a[2] = a[2], a[1]; print(a)", "The right side is read first, then both are assigned."),
            (f"print(list(enumerate('{w[:2]}'))[1])", "enumerate gives (index, item) pairs."),
            (f"print(max({a}) - min({a}), len(set({a + a[:2]})))", "A set drops duplicates."),
            (f"a = {a}; print([v for v in a if v > {n}], a[::2])", "The comprehension filters, and a[::2] takes indexes 0 and 2."),
            (f"s = 0\nfor v in {a}:\n    if v % 2: continue\n    s += v\nprint(s)", "continue skips the rest of the loop body for odd v."),
            (f"n = {n * 3}\nwhile n > {k + 1}:\n    n //= 2\nprint(n)", "Halve n until the test fails, and // rounds down."),
        ]
    )
    return "python", code, rule, None


def js():
    w = random.choice(WORDS)
    a = nums()
    arr = ", ".join(map(str, a))
    x, y, z, n = R(1, 9), R(1, 9), R(1, 9), R(3, 6)
    odd = 2 * x + 1
    math_ops = "*, - and / turn strings into numbers."
    code, rule = random.choice(
        [
            (f'console.log("{x}" + {y})', "With +, a string turns the other side into a string."),
            (f'console.log({x} + {y} + "{z}")', "+ runs left to right, so the numbers add first and then join the string."),
            (f'console.log("{x}" + {y} + {z})', "+ runs left to right, so after the string everything gets joined."),
            (f'console.log("{x * 2}" * {y})', math_ops),
            (f'console.log("{x + 10}" - {y})', math_ops),
            (f"console.log({odd} / 2)", "JS has no integer division, so / keeps the decimal."),
            (f"console.log(Math.floor(-{odd} / 2), Math.trunc(-{odd} / 2))", "floor goes down, and trunc goes toward zero."),
            (f'console.log(parseInt("{x}{y}px"), Number("{x}{y}px"))', "parseInt reads digits until a non-digit. Number needs the whole string to be a number."),
            (f"const a = [{arr}]; console.log(a.shift(), a.length)", "shift removes and returns the first item."),
            (f"const a = [{arr}]; console.log(a.pop(), a.length)", "pop removes and returns the last item."),
            (f"const a = [{arr}]; console.log(a.push({x}))", "push returns the new length, not the array."),
            (f"const a = [{arr}]; console.log(a.unshift({x}), a[1])", "unshift adds to the front and returns the new length."),
            (f"const a = [{arr}]; console.log(a.slice(1, 3))", "slice(start, end) copies from start and stops before end."),
            (f"const a = [{arr}]; a.splice(1, 2); console.log(a)", "splice(start, count) removes count items in place."),
            (f"console.log([{x}, 10, {y}, 1].sort())", "sort() with no function compares items as strings, so 10 comes before 2."),
            (f"console.log([{arr}].map(v => v * 2))", "map makes a new array with the function applied to each item."),
            (f"console.log([{arr}].filter(v => v % 2))", "filter keeps items where the function is truthy, and odd % 2 is 1."),
            (f"console.log([{arr}].reduce((s, v) => s + v, 0))", "reduce carries s from item to item, starting at 0."),
            (f"console.log([{arr}].indexOf({a[2]}), [{arr}].indexOf(0))", "indexOf returns the position, or -1 when it's missing."),
            (f'console.log([{arr}].join("-"))', "join puts the separator between the items."),
            (f"console.log(0 || {x}, 0 ?? {x})", "|| skips any falsy value. ?? only skips null and undefined."),
            (f"console.log(0 && {x}, {y} && {x})", "&& returns the first falsy value, or the last value."),
            (f"for (var i = 0; i < {n}; i++) {{}} console.log(i)", "var isn't scoped to the loop, and the loop ends when i reaches the limit."),
            (f"let x = {x}; {{ let x = {y}; }} console.log(x)", "let inside a block is a different variable."),
            (f"const f = [1, 2, 3].map(i => () => i * {x}); console.log(f[1]())", "Each arrow function keeps its own i."),
            (f"const o = {{ a: {x} }}; const p = o; p.a = {y}; console.log(o.a)", "p = o points to the same object."),
            (f"let i = {x}; console.log(i++ + ++i, i)", "i++ gives the old value, and ++i gives the new one."),
            (f'const s = "{w}"; console.log(s.length, s[1], s.slice(-2))', "slice(-2) takes the last two characters."),
            (f'console.log([..."{w[:4]}"].reverse().join(""))', "The spread splits the string into characters."),
            (f'console.log(String({x}) + {y}, Number("{x}") + {y})', "String joins, and Number adds."),
            (f"const [p, , q = {z}] = [{x}, {y}]; console.log(p, q)", "Destructuring goes by position, and a default fills a missing spot."),
            (f"const add = a => b => a + b; console.log(add({x})({y}))", "add(a) returns a second function that still remembers a."),
            (f"const f = (a, b = {z}) => a * b; console.log(f({x}), f({x}, {y}))", "The default is used only when the argument is missing."),
            (f"const f = x => {{ x * 2 }}; console.log(f({x}))", "Braces need an explicit return, so this gives undefined."),
            (f"for (var i = 0; i < {n}; i++) setTimeout(() => console.log(i), 0);", "The callbacks run after the loop, and they all share one var i."),
            (f"for (let i = 0; i < 3; i++) setTimeout(() => console.log(i * {x}), 0);", "let gives each turn of the loop its own i."),
            (f'setTimeout(() => console.log("a"), 0); console.log("b");', "A setTimeout callback waits until the current code is done."),
            (f"const o = {{ a: {x}, b: {y} }}; console.log(Object.keys(o).length, o.c)", "A missing property is undefined, not an error."),
        ]
    )
    return "js", code, rule, None


def c():
    w = random.choice(WORDS)
    a = nums()
    arr = ", ".join(map(str, a))
    x, y, k, n = R(2, 9), R(2, 9), R(1, 3), R(11, 987)
    odd = 2 * x + 1
    code, rule = random.choice(
        [
            (f'int i = {x}; int v = i++ + {y}; printf("%d %d", v, i);', "i++ gives the old value first, then i goes up by 1."),
            (f'int i = {x}; int v = ++i * 2; printf("%d %d", v, i);', "++i adds 1 first, then gives the new value."),
            (f'int i = {x}; int v = i-- - 1; printf("%d %d", v, i);', "i-- gives the old value first, then i goes down by 1."),
            (f'int a[] = {{{arr}}}; int *p = a + 1; printf("%d", *(p + 1));', "p + 1 moves one element forward."),
            (f'int a[] = {{{arr}}}; int *p = a; p++; printf("%d", *p + 1);', "* binds before +, so this is (the item p points at) + 1."),
            (f'int a[] = {{{arr}}}; int *p = a + 2; printf("%d %d", p[-1], p[1]);', "p[i] is *(p + i), so a negative index looks behind p."),
            (f'int a[] = {{{arr}}}; int *p = a; int v = *p++; printf("%d %d", v, *p);', "*p++ reads the item, then moves p forward."),
            (f'int v = {x}; int *p = &v; *p += {y}; printf("%d", v);', "*p is v itself, so the change shows in v."),
            (f'printf("%d", !{x} + ({y} && 0) + ({k} || 0));', "Logic operators give 1 for true and 0 for false."),
            (f'printf("%.1f", (float){odd} / 2);', "The cast happens first, so this is float division."),
            (f'printf("%.1f", (float)({odd} / 2));', "The int division happens first, then the cast."),
            (f'int c = 0; for (int i = {k}; i < {k + 9}; i += {x}) c++; printf("%d", c);', "List the values i takes, and stop before the limit."),
            (f'int i = 0, s = 0; while (i++ < {k + 2}) s += i; printf("%d %d", s, i);', "The test uses the old i, then i goes up before the body runs."),
            (f'char s[] = "{w}"; printf("%d %c", (int)strlen(s), s[{k}]);', "strlen doesn't count the ending null character."),
            (f'char s[] = "{w}"; printf("%s", s + {k});', "s + n starts the string at index n."),
            (f'char s[] = "{w}"; s[{k + 1}] = 0; printf("%s", s);', "A 0 in the array ends the string there."),
            (f'int v = {x}; v = v > {y} ? v - {y} : v + {y}; printf("%d", v);', "The ternary picks the first value when the test is true."),
            (f'int v = {k}, c = 0; switch (v) {{ case 1: c += 1; case 2: c += 2; break; case 3: c += 3; }} printf("%d", c);', "Without break, a case falls into the next one."),
            (f'int v = {n}, s = 0; while (v) {{ s += v % 10; v /= 10; }} printf("%d", s);', "% 10 takes the last digit, and / 10 drops it."),
            (f'int v = {n}, r = 0; while (v) {{ r = r * 10 + v % 10; v /= 10; }} printf("%d", r);', "This builds the number with its digits reversed."),
            (f'unsigned char u = 250; u += {x + 6}; printf("%d", u);', "An unsigned char wraps around at 256."),
            (f'int a[] = {{{arr}}}; int s = 0; for (int i = 0; i < 4; i += 2) s += a[i]; printf("%d", s);', "i is 0 and then 2, so only those two items count."),
            (f'int v = {x}; if (v = 0) v = 5; else v += {y}; printf("%d", v);', "A single = assigns 0, and 0 is false, so the else runs."),
            (f'int v = {x * 4}; v >>= 1; v |= 1; v <<= 2; printf("%d", v);', "Update v one statement at a time and write each value down."),
            (f'int a = {x}, b = {y}; int *p = &a, *q = &b; *p = *q; *q += 1; printf("%d %d", a, b);', "*p = *q copies the value, so a no longer follows b."),
            (f'int a[] = {{{arr}}}; int *p = a + 3; printf("%d", (int)(p - a) + *p);', "Subtracting two pointers gives the number of elements between them."),
            (f'int i; for (i = 0; i < {x}; i++) if (i == {k + 1}) break; printf("%d", i);', "break leaves the loop before i++ runs again."),
            (f'int c = 0; for (int i = 0; i < 3; i++) for (int j = i; j < 3; j++) c += {k}; printf("%d", c);', "The inner loop runs 3, then 2, then 1 times."),
        ]
    )
    return "c", code, rule, None


# ── memory ───────────────────────────────────────────────────────────


def ascii_():
    letter = random.choice("abcdefghijklmnopqrstuvwxy")
    kind = R(1, 5)
    if kind == 1:
        ch = random.choice([letter, letter.upper(), str(R(0, 9))])
        return "python", f"ord('{ch}')", None, str(ord(ch))
    if kind == 2:
        n = random.choice([R(65, 90), R(97, 122), R(48, 57)])
        return "python", f"chr({n})", None, chr(n)
    if kind == 3:
        k = R(1, 122 - ord(letter))
        return "c", f"printf(\"%c\", '{letter}' + {k});", None, chr(ord(letter) + k)
    if kind == 4:
        d = R(0, 9)
        return "c", f"'{d}' - '0'", None, str(d)
    return "c", f"'{letter}' - 'a'", None, str(ord(letter) - 97)


FACTS = [
    ("python", 'print(bool(""), bool("0"), bool(" "))', "Only the empty string is falsy. Any character makes it truthy."),
    ("python", "print(bool([]), bool([0]), bool(0.0))", "Empty containers and zero are falsy. A list with anything in it is truthy."),
    ("python", "print(bool(None), bool({}), bool((0,)))", "None and empty containers are falsy. A tuple with one item is truthy."),
    ("python", "print(type(3 / 1).__name__, type(3 // 1).__name__)", "/ always gives a float, and // of two ints gives an int."),
    ("python", "print(type((1)).__name__, type((1,)).__name__)", "Parentheses alone don't make a tuple. The comma does."),
    ("python", "print(type({}).__name__, type({1}).__name__)", "{} is an empty dict, and {1} is a set."),
    ("python", "print([3, 1, 2].sort())", "list.sort() sorts in place and returns None."),
    ("python", "print([].append(1))", "append changes the list and returns None."),
    ("python", 'print(len("a\\nb"))', "An escape like \\n is one character."),
    ("python", 'print("abc".upper(), "ABC".lower().capitalize())', "capitalize makes only the first letter uppercase."),
    ("python", 'print(len("a b  c".split()), len("a b  c".split(" ")))', 'split() with no argument skips repeated spaces. split(" ") keeps an empty piece.'),
    ("python", "print(10 // 3, 10 % 3, divmod(10, 3))", "divmod gives the quotient and the remainder as a tuple."),
    ("python", "print(True + True, True == 1)", "True counts as 1 in arithmetic."),
    ("python", 'print("5" * 2, [0] * 2)', "Multiplying a string or a list repeats it."),
    ("python", 'print(max("apple"), min("Zebra"))', "Uppercase letters come before lowercase in ASCII."),
    ("python", 'print(sorted("bAc"))', "Uppercase letters sort before lowercase."),
    ("python", 'print(int("7") + 1, str(7) + "1", float("7"))', "int adds, str joins, and float prints with a decimal point."),
    ("python", "print(range(3)[-1], list(range(0)))", "range(3) is 0, 1, 2, and range(0) is empty."),
    ("python", 'print("x" in "xyz", 2 in [1, 2], "a" in {"a": 1})', "in checks substrings, list items and dict keys."),
    ("python", "print(None == False, not None)", "None is falsy, but it isn't equal to False."),
    ("python", "a = [1, 2, 3]; print(a.index(2), a.count(2), len(a))", "index gives the position, and count gives how many."),
    ("python", 'print(chr(ord("a") + 1), ord("b") - ord("a"))', "ord turns a character into its code, and chr goes back."),
    ("python", 'print(3 == 3.0, "3" == 3)', "An int can equal a float, but a string never equals a number."),
    ("python", "print(1 / 2, 2 ** -1, 4 ** 0.5)", "A negative power divides, and a power of 0.5 is a square root."),
    ("python", 'print("%d-%s" % (3, "a"), f"{3 + 1}{\'a\' * 2}")', "An f-string runs the code inside the braces."),
    ("python", 'x = 5; print(x > 3 and "yes" or "no")', "and returns its last value when both are truthy, so or never runs."),
    ("python", "print(abs(-3), pow(2, 5), 7 % 2 == 1)", "pow(2, 5) is 2 ** 5."),
    ("python", "print(1 == 1.0, [] == [], (1, 2) == (1, 2))", "== compares values, so two lists with the same items are equal."),
    ("python", 'print("a" < "b", "a" < "B", "abc" < "abd")', "Strings compare by character code, and uppercase comes first."),
    ("python", "print(5 // 2, -5 // 2, 5 % -2)", "// rounds down, and % takes the sign of the right side."),
    ("js", "console.log(typeof undefined, typeof null)", "typeof null is object, which is an old quirk."),
    ("js", "console.log(typeof [], typeof {})", "Arrays are objects in JS."),
    ("js", "console.log(typeof (() => 1), typeof NaN)", "NaN is still a number."),
    ("js", 'console.log(typeof "a", typeof 1, typeof true)', "typeof gives a lowercase string."),
    ("js", "console.log(typeof typeof 1)", "typeof always returns a string."),
    ("js", 'console.log(Boolean(""), Boolean("0"), Boolean(" "))', "Only the empty string is falsy."),
    ("js", "console.log(Boolean([]), Boolean({}), Boolean(0))", "Empty arrays and objects are truthy in JS."),
    ("js", "console.log(Boolean(NaN), Boolean(null), Boolean(undefined))", "NaN, null and undefined are all falsy."),
    ("js", "console.log([1, 2].push(3), [1, 2].pop())", "push returns the new length, and pop returns the removed item."),
    ("js", "console.log([3, 1, 2].sort().reverse()[0])", "sort and reverse both return the array, so they chain."),
    ("js", 'console.log("abc".indexOf("z"), [1, 2].includes(2))', "indexOf gives -1 when it's missing."),
    ("js", "console.log([1, 2, 3].splice(1, 1), [1, 2, 3].slice(1, 2))", "splice returns the removed items, and slice returns a copy."),
    ("js", 'console.log("a-b-c".split("-").length, "abc".split("").length)', 'split("") splits into single characters.'),
    ("js", 'console.log(NaN == NaN, isNaN("x"))', "NaN isn't equal to anything, even itself."),
    ("js", "console.log(null == undefined, null === undefined)", "== treats null and undefined as equal, and === doesn't."),
    ("js", "console.log(1 / 0, -1 / 0, 0 / 0)", "Dividing by zero gives Infinity, and 0 / 0 is NaN."),
    ("js", 'console.log("b" > "a", "B" > "a", "10" > "9")', "Strings compare character by character."),
    ("js", "console.log(2 ** 10, 7 % 3, 7 / 2 | 0)", "| 0 cuts off the decimal part."),
    ("js", 'console.log("abc".length, [1, [2, 3]].length)', "A nested array counts as one item."),
    ("js", 'console.log("5" == 5, "5" === 5, 5 == 5.0)', "== converts types first, and === doesn't."),
    ("js", 'console.log(+"3" + +"4", "3" + "4")', "A unary + turns a string into a number."),
    ("js", 'console.log([1, 2, 3].at(-1), "abc".at(-1), "abc"[-1])', "at() takes negative indexes, and brackets don't."),
    ("js", 'console.log("abc".substring(1), "abc".slice(-1), "abc".charAt(0))', "slice(-1) is the last character."),
    ("js", "console.log(Math.max(1, 5, 3), Math.abs(-2), Math.sqrt(16))", "These are the Math helpers worth knowing."),
    ("js", "console.log(Math.round(2.5), Math.round(-2.5), Math.round(2.4))", "Math.round sends .5 upward, so -2.5 becomes -2."),
    ("js", 'console.log(parseInt("08"), parseFloat("3.5kg"), Number(""))', "An empty string becomes 0 with Number."),
    ("js", "console.log(String(null), String([1, 2]), String(true))", "An array turns into its items joined by commas."),
    ("js", "console.log([1, 2] + [3], [] + 1)", "+ turns arrays into strings first."),
    ("js", 'console.log("a".repeat(3), "ab".padStart(4, "*"))', "padStart fills on the left up to the length."),
    ("js", "let u; console.log(u, typeof u, u + 1)", "A let with no value is undefined, and undefined + 1 is NaN."),
    ("js", "console.log(0.1 + 0.2 === 0.3, 0.5 + 0.25 === 0.75)", "0.1 and 0.2 aren't exact in binary, but halves and quarters are."),
    ("js", "const a = [1, 2, 3]; a.length = 1; console.log(a)", "Setting length cuts the array."),
    ("js", 'console.log([..."ab"].map(c => c.charCodeAt(0)), String.fromCharCode(66))', "charCodeAt gives the code, and fromCharCode goes back."),
    ("js", "console.log(null + 1, undefined + 1, true + 1)", "null becomes 0, undefined becomes NaN, and true becomes 1."),
    ("js", 'console.log([] == false, "" == 0, null == 0)', "== converts [] and an empty string to 0, but null only equals undefined."),
    ("js", "console.log([1, 2, 3].indexOf(2), [1, 2, 3].find(v => v > 1), [1, 2, 3].findIndex(v => v > 5))", "find returns the item, and findIndex returns -1 when nothing matches."),
    ("c", 'printf("%x %X %o", 255, 255, 8);', "%x is hex, %X is uppercase hex, and %o is octal."),
    ("c", 'printf("%05d %03d", 42, 7);', "A 0 before the width pads with zeros."),
    ("c", 'printf("%.2f %.1f", 3.14159, 2.96);', "The precision rounds to that many decimal places."),
    ("c", "printf(\"%c %d\", 65, 'A');", "%c prints the character, and %d prints its code."),
    ("c", "printf(\"%c%c\", 'a' - 32, 'A' + 32);", "Uppercase and lowercase are 32 apart."),
    ("c", "printf(\"%d %d\", '0', '9' - '0');", "'0' is 48, and subtracting '0' turns a digit character into its number."),
    ("c", 'printf("%d %d", (int)sizeof(char), (int)sizeof(int));', "A char is 1 byte, and an int is 4 bytes on a normal PC."),
    ("c", 'printf("%d %d", (int)sizeof(double), (int)sizeof(float));', "A double is 8 bytes, and a float is 4."),
    ("c", 'int a[5]; printf("%d", (int)(sizeof(a) / sizeof(a[0])));', "sizeof the array over sizeof one item gives the item count."),
    ("c", 'printf("%d", INT_MAX);', "INT_MAX is 2 to the 31st minus 1."),
    ("c", 'printf("%d %d", 7 / 2, -7 / 2);', "int division cuts toward zero."),
    ("c", 'printf("%d %d", 7 % 3, -7 % 3);', "% takes the sign of the left side."),
    ("c", 'printf("%d", (int)strlen("hello\\n"));', "\\n is one character."),
    ("c", 'char s[] = "abc"; printf("%d %d", (int)sizeof(s), (int)strlen(s));', "sizeof counts the ending null character, and strlen doesn't."),
    ("c", 'printf("%d %d", 1 << 10, 1 << 4);', "1 << n is 2 to the n."),
    ("c", 'printf("%d %d %d", 5 & 3, 5 | 3, 5 ^ 3);', "5 is 101 and 3 is 011 in binary."),
    ("c", 'printf("%d %d %d", 10 == 10.0, !0, !7);', "True is 1, false is 0, and any nonzero value is true."),
    ("c", "printf(\"%d\", 'a' < 'B');", "Uppercase letters come first in ASCII."),
    ("c", 'int x; x = (3, 5); printf("%d", x);', "The comma operator gives its last value."),
    ("c", 'printf("%d", 3 + 4 * 2 % 3);', "* and % have the same rank and run left to right, before +."),
    ("c", 'int a = 5, b = 2; printf("%.2f", (double)a / b);', "The cast makes it floating point division."),
    ("c", 'printf("%s", "abc" + 1);', "Adding 1 to a string pointer skips the first character."),
    ("c", "char c = 'a'; c++; printf(\"%c\", c);", "A char is a small number, so ++ moves to the next letter."),
    ("c", 'int a[3] = {1}; printf("%d %d", a[0], a[2]);', "Missing initializers become 0."),
    ("c", 'int i = 3; i *= 2 + 1; printf("%d", i);', "i *= 2 + 1 means i = i * (2 + 1)."),
    ("c", 'printf("%d", 07 + 010);', "A leading 0 makes an octal literal, so 010 is 8."),
    ("c", 'printf("%d %d", 0x1F, 0xFF);', "Hex F is 15, so 0x1F is 16 + 15."),
    ("c", 'printf("%d", 1 + 2 == 3);', "+ runs before ==."),
    ("c", 'printf("%d %d", 5 / 2 * 2, 5 * 2 / 2);', "/ and * run left to right, and int division loses the half."),
    ("c", 'int a = 1, b = 2; printf("%d", a++ + b);', "a++ gives the old a here."),
    ("c", 'printf("%d %d", 1 && 2, 1 & 2);', "&& is logic and gives 1. & is bitwise, and 01 & 10 is 0."),
    ("c", 'printf("%d", (int)3.99 + (int)-3.99);', "A cast to int cuts toward zero on both sides."),
    ("c", 'char s[] = "hi"; printf("%d", s[2]);', "The character after the last letter is the null character, which is 0."),
]


def memo():
    kind = R(1, 8)
    if kind == 1:
        n = R(0, 16)
        return "any", f"2 to the power of {n}", "The powers of 2 go 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024.", str(2**n)
    if kind == 2:
        n = R(3, 12)
        return "any", f"{2**n} is 2 to the power of what", "The powers of 2 go 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024.", str(n)
    if kind == 3:
        d = R(10, 15)
        return "any", f"hex digit {d:X} in decimal", "Hex A to F are 10 to 15.", str(d)
    lang, code, rule = random.choice(FACTS)
    return lang, code, rule, None


# ── traps ────────────────────────────────────────────────────────────


def traps():
    a, b = R(5, 19), R(2, 6)
    x, y, k, n = R(1, 4), R(5, 9), R(2, 4), R(3, 6)
    odd = 2 * y + 1
    left = "% takes the sign of the left side here. Python would give a positive result."
    chain = "This runs left to right, so 3 > 2 gives a true value, and that isn't greater than 1."
    lang, code, rule = random.choice(
        [
            ("python", f"print(-{a} % {b})", "Python % takes the sign of the right side. C and JS take the sign of the left."),
            ("js", f"console.log(-{a} % {b})", left),
            ("c", f'printf("%d", -{a} % {b});', left),
            ("python", f"print({odd} // 2 + {odd} // 2, -{odd} // 2)", "Python // rounds down, even for negatives."),
            ("js", f"console.log({odd} / 2 + {odd} / 2)", "JS / never drops the decimal, so the halves add back up. C would lose 1."),
            ("c", f'printf("%d", {odd} / 2 + {odd} / 2);', "C int division drops the decimal, so each half loses 0.5."),
            ("python", f"print({x} and {y}, {x} or {y}, 0 or {y})", "Python and/or give back one of the values, not True or False."),
            ("js", f"console.log({x} && {y}, {x} || {y})", "JS && and || give back one of the values, not true or false."),
            ("c", f'printf("%d %d", {x} && {y}, {x} || {y});', "C logic operators give 1 or 0. Python and JS give back one of the values."),
            ("python", 'print(bool([]), bool("0"), bool(0))', "Python treats an empty list as falsy. JS treats [] as truthy."),
            ("js", 'console.log(Boolean([]), Boolean("0"), Boolean(0))', "JS treats [] as truthy. Python treats an empty list as falsy."),
            ("c", 'if ("") printf("yes"); else printf("no");', "A string literal is a non-null pointer, so it's true in C even when empty."),
            ("python", f'print("{x}" * {k}, "{x}" + "{y}")', "Python repeats a string with *. JS would turn it into a number."),
            ("js", f'console.log("{x}" * {k}, "{x}" + {y})', "JS * turns the string into a number, and + joins."),
            ("c", f"printf(\"%d %c\", '{x}' + {k}, '{x}' + {k});", "A char is a number in C, and '0' is 48."),
            ("python", f"print(2 ** {k}, 2 ^ {k})", "** is power and ^ is XOR."),
            ("js", f"console.log(2 ** {k}, 2 ^ {k})", "** is power and ^ is XOR."),
            ("c", f'printf("%d", 2 ^ {k});', "C has no power operator, and ^ is XOR."),
            ("python", 'print(1 == 1.0, [1] == [1], "1" == 1)', "Python == compares values and never turns a string into a number."),
            ("js", 'console.log(1 == "1", [1] == [1], 1 === 1.0)', "JS == converts types, but two arrays are equal only when they're the same object."),
            ("c", 'char a[] = "hi", b[] = "hi"; printf("%d %d", a == b, strcmp(a, b));', "== on C strings compares addresses. strcmp returns 0 when they match."),
            ("python", "print(1 < 2 < 3, 3 > 2 > 1)", "Python chains comparisons, so 3 > 2 > 1 means 3 > 2 and 2 > 1."),
            ("js", "console.log(1 < 2 < 3, 3 > 2 > 1)", chain),
            ("c", 'printf("%d %d", 1 < 2 < 3, 3 > 2 > 1);', chain),
            ("python", f"print(sorted([10, 9, {x}, 1]))", "Python sorts numbers as numbers. JS sort() would compare them as strings."),
            ("js", f"console.log([10, 9, {x}, 1].sort())", "sort() with no function compares items as strings. Python sorts numbers as numbers."),
            ("js", f"console.log([10, 9, {x}, 1].sort((p, q) => p - q))", "A compare function p - q sorts numbers going up."),
            ("python", "print(round(2.5), round(3.5), round(-2.5))", "Python rounds .5 to the nearest even number. JS Math.round sends it up."),
            ("js", "console.log(Math.round(2.5), Math.round(3.5), Math.round(-2.5))", "Math.round sends .5 upward. Python rounds it to the nearest even number."),
            ("c", f'printf("%d %d", (int){x}.9, (int)-{x}.9);', "A cast to int cuts toward zero."),
            ("python", f"print(int({x}.9), int(-{x}.9), -{x}.9 // 1)", "int() cuts toward zero, and // rounds down."),
            ("js", f'console.log(parseInt("{x}.9"), Math.floor(-{x}.9), Math.trunc(-{x}.9))', "parseInt and trunc cut toward zero, and floor goes down."),
            ("python", 'print("abc"[-1], [1, 2, 3][-2])', "Python brackets take negative indexes."),
            ("js", 'console.log("abc"[-1], [1, 2, 3].at(-2))', "JS brackets don't take negative indexes, and at() does."),
            ("python", f"for i in range({n}): pass\nprint(i)", "The loop variable keeps its last value, which is one below the range end."),
            ("js", f"for (var i = 0; i < {n}; i++) {{}} console.log(i)", "The loop stops when i reaches the limit, so i ends at the limit."),
            ("c", f'int i; for (i = 0; i < {n}; i++); printf("%d", i);', "The loop stops when i reaches the limit, so i ends at the limit."),
            ("python", "print(True + True, 3 > 2)", "Python prints True with a capital T, and True counts as 1."),
            ("js", "console.log(true + true, 3 > 2)", "JS prints true in lowercase, and true counts as 1."),
            ("c", 'printf("%d", (3 > 2) + (2 > 1));', "A comparison gives 1 in C."),
            ("python", "print([1, 2] + [3], [0] * 2)", "Python + joins two lists into one."),
            ("js", "console.log([1, 2] + [3])", "JS + turns both arrays into strings and joins them."),
            ("js", "console.log([...[1, 2], 3], [1, 2].concat(3))", "Spread and concat are how JS joins arrays."),
            ("python", 'print("10" < "9", 10 < 9)', "Strings compare character by character, and 1 comes before 9."),
            ("js", 'console.log("10" < "9", "10" < 9)', "Two strings compare by character. A string and a number compare as numbers."),
            ("python", "a = [1, 2, 3]; b = a; b[0] = 9; print(a[0])", "b = a is the same list."),
            ("js", "const a = [1, 2, 3]; const b = [...a]; b[0] = 9; console.log(a[0])", "The spread makes a copy, so a doesn't change."),
            ("c", 'int a[] = {1, 2, 3}; int *b = a; b[0] = 9; printf("%d", a[0]);', "b points at the same array."),
            ("python", "print(7 // 2 * 2 == 7, 7 / 2 * 2 == 7)", "// loses the half, and / keeps it."),
            ("js", "console.log(7 / 2 * 2 == 7)", "JS / keeps the decimal, so nothing is lost."),
            ("c", 'printf("%d", 7 / 2 * 2 == 7);', "C int division loses the half, so 7 / 2 * 2 is 6."),
            ("python", 's = "abc"; s = s.replace("a", "x"); print(s)', "Python strings can't change in place, so replace returns a new one."),
            ("js", 'let s = "abc"; s[0] = "x"; console.log(s)', "JS strings can't change in place, and the assignment is silently ignored."),
            ("c", "char s[] = \"abc\"; s[0] = 'x'; printf(\"%s\", s);", "A C char array can be changed in place."),
            ("python", "print(None or 0, 0 or None)", "or returns the last value when nothing is truthy."),
            ("js", 'console.log(null ?? "d", 0 ?? "d", 0 || "d")', "?? only replaces null and undefined. || replaces any falsy value."),
            ("python", "x = 1\ndef f():\n    x = 2\nf()\nprint(x)", "Assigning inside a function makes a local x, so the outer one stays."),
            ("js", "var x = 1; function f() { x = 2; } f(); console.log(x)", "With no var or let inside, f changes the outer x."),
            ("js", "var x = 1; function f() { var x = 2; } f(); console.log(x)", "var inside f makes a local x."),
            ("c", 'int x = 1; { int x = 2; } printf("%d", x);', "The inner x only lives inside its block."),
            ("python", f"print({odd} / 2, type({odd} / 2).__name__)", "Python / always gives a float."),
            ("c", f'printf("%.1f %.1f", {odd} / 2 * 1.0, {odd} * 1.0 / 2);', "The int division happens before the 1.0 can turn it into a double."),
            ("python", 'print(len("hello"), "hello"[1], "hello".index("l"))', "Python uses len() as a function."),
            ("js", 'console.log("hello".length, "hello"[1], "hello".indexOf("l"))', "JS uses the length property."),
            ("c", 'printf("%d %c", (int)strlen("hello"), "hello"[1]);', "C uses strlen from string.h."),
        ]
    )
    return lang, code, rule, None


TOPICS = {
    "mod": (mod, "Python // rounds down, and % takes the sign of the right side."),
    "cdiv": (cdiv, "C / drops the decimal part, and % takes the sign of the left side."),
    "bits": (
        bits,
        "Write both in binary. & needs both 1, | needs either, ^ needs different. "
        "<< n multiplies by 2^n, >> n divides by 2^n, and ~x is -(x + 1).",
    ),
    "prec": (
        prec,
        "** goes first (right to left), then unary minus, then * // %, "
        "then + -, then << >>, then &.",
    ),
    "base": (base, "Binary places are 32 16 8 4 2 1. Hex is 16s and 1s. Octal is 8s and 1s."),
    "py": (py, ""),
    "js": (js, ""),
    "c": (c, ""),
    "ascii": (ascii_, "'0' is 48, 'A' is 65 and 'a' is 97."),
    "memo": (memo, ""),
    "traps": (traps, ""),
}
GROUPS = {
    "math": ["mod", "cdiv", "bits", "prec", "base"],
    "skills": ["py", "js", "c"],
    "memory": ["ascii", "memo"],
    "traps": ["traps"],
}
BANK_SIZE = {"math": 45, "skills": 110, "memory": 125, "traps": 130}


def make(topic):
    maker, topic_rule = TOPICS[topic]
    lang, q, rule, answer = maker()
    if answer is None:
        answer = RUN[lang](q)
    answer = " ".join(answer.split("\n"))
    assert answer != "", q
    return lang, q, answer, rule or topic_rule


# Spaces and quotes don't count, so [1,2] matches [ 1, 2 ].
def clean(text):
    return "".join(ch for ch in text.lower() if ch not in " '\"\n")


def dump(path):
    random.seed(14)
    rules = []
    items = []
    for group, topics in GROUPS.items():
        for topic in topics:
            seen = set()
            for _ in range(BANK_SIZE[group] * 12):
                lang, q, answer, rule = make(topic)
                if q in seen:
                    continue
                seen.add(q)
                if rule not in rules:
                    rules.append(rule)
                items.append([group, lang, q, answer, rules.index(rule)])
                if len(seen) == BANK_SIZE[group]:
                    break
            print(f"{topic} has {len(seen)} questions")
    with open(path, "w") as f:
        json.dump({"rules": rules, "items": items}, f, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {len(items)} questions to {path}")


def play(args):
    count = next((int(a) for a in args if a.isdigit()), 20)
    picked = []
    for a in args:
        picked += GROUPS.get(a, [a] if a in TOPICS else [])
    pools = [picked] if picked else list(GROUPS.values())
    names = ", ".join(t for p in pools for t in p)
    print(f"{count} questions on {names}. Type what it prints, or q to stop.\n")

    right = 0
    asked = 0
    missed = []
    spent = {}
    start = time.time()
    for number in range(1, count + 1):
        topic = random.choice(random.choice(pools))
        lang, q, answer, rule = make(topic)
        prompt = f"{LABEL[lang]:<7} {q}"
        gap = "\n    = " if len(prompt) > 50 or "\n" in q else "  = "
        t0 = time.time()
        try:
            got = input(f"{number:>2}. {prompt}{gap}").strip()
        except EOFError:
            break
        if got.lower() == "q":
            break
        took = time.time() - t0
        asked += 1
        spent.setdefault(topic, []).append(took)
        if clean(got) == clean(answer):
            right += 1
            print(f"    correct, {took:.1f}s")
        else:
            missed.append(f"{prompt}  = {answer}  (you typed {got or 'nothing'})")
            print(f"    no, it's {answer}. {rule}")

    if not asked:
        return
    total = time.time() - start
    print(f"\nYou got {right} of {asked} in {total:.0f}s, about {total / asked:.1f}s each.")
    slowest = max(spent, key=lambda t: sum(spent[t]) / len(spent[t]))
    print(f"Your slowest topic was {slowest}.")
    if missed:
        print("\nYou missed these.")
        for line in missed:
            print(f"  {line}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--dump"]:
        dump(sys.argv[2])
    else:
        play(sys.argv[1:])
