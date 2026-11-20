// src/data/get14-cheatsheet.js

// Cheat sheet for /get14/cheatsheet. Every result is checked against a real
// run by scripts/verify-get14.mjs. A line with no print call is run as an
// expression.

// The same idea in each language. A cell is [code, result, C printf format].
export const compare = [
  {
    what: "divide two whole numbers",
    python: ["7 / 2", "3.5"],
    js: ["7 / 2", "3.5"],
    c: ["7 / 2", "3"],
  },
  {
    what: "divide a negative and drop the decimal",
    python: ["-7 // 2", "-4"],
    js: ["Math.trunc(-7 / 2)", "-3"],
    c: ["-7 / 2", "-3"],
  },
  {
    what: "remainder of a negative",
    python: ["-7 % 3", "2"],
    js: ["-7 % 3", "-1"],
    c: ["-7 % 3", "-1"],
  },
  {
    what: "remainder by a negative",
    python: ["7 % -3", "-2"],
    js: ["7 % -3", "1"],
    c: ["7 % -3", "1"],
  },
  {
    what: "power",
    python: ["2 ** 3", "8"],
    js: ["2 ** 3", "8"],
    c: ["pow(2, 3)", "8"],
  },
  {
    what: "the ^ operator is XOR",
    python: ["2 ^ 3", "1"],
    js: ["2 ^ 3", "1"],
    c: ["2 ^ 3", "1"],
  },
  {
    what: "two comparisons in a row",
    python: ["3 > 2 > 1", "True"],
    js: ["3 > 2 > 1", "false"],
    c: ["3 > 2 > 1", "0"],
  },
  {
    what: "bitwise & next to ==",
    python: ["6 & 3 == 2", "True"],
    js: ["6 & 3 == 2", "0"],
    c: ["6 & 3 == 2", "0"],
  },
  {
    what: "and",
    python: ["3 and 5", "5"],
    js: ["3 && 5", "5"],
    c: ["3 && 5", "1"],
  },
  {
    what: "or",
    python: ["0 or 5", "5"],
    js: ["0 || 5", "5"],
    c: ["0 || 5", "1"],
  },
  {
    what: "how true prints",
    python: ["3 > 2", "True"],
    js: ["3 > 2", "true"],
    c: ["3 > 2", "1"],
  },
  {
    what: "is an empty list or string true",
    python: ["bool([])", "False"],
    js: ["Boolean([])", "true"],
    c: ['!!""', "1"],
  },
  {
    what: "a digit character plus a number",
    python: ['"3" + 4', "TypeError"],
    js: ['"3" + 4', "34"],
    c: ["'3' + 4", "55"],
  },
  {
    what: "a digit character times a number",
    python: ['"3" * 2', "33"],
    js: ['"3" * 2', "6"],
    c: ["'3' * 2", "102"],
  },
  {
    what: "a string compared with a number",
    python: ['"1" == 1', "False"],
    js: ['"1" == 1', "true"],
    c: ["'1' == 1", "0"],
  },
  {
    what: "round 2.5",
    python: ["round(2.5)", "2"],
    js: ["Math.round(2.5)", "3"],
    c: ["(int)2.5", "2"],
  },
  {
    what: "sort 10, 9, 1",
    python: ["sorted([10, 9, 1])", "[1, 9, 10]"],
    js: ["[10, 9, 1].sort()", "[1, 10, 9]"],
    c: null,
  },
  {
    what: "last character",
    python: ['"abc"[-1]', "c"],
    js: ['"abc".at(-1)', "c"],
    c: ['"abc"[2]', "c", "%c"],
  },
  {
    what: "length of a string",
    python: ['len("hello")', "5"],
    js: ['"hello".length', "5"],
    c: ['strlen("hello")', "5"],
  },
];

export const sheets = [
  {
    id: "numbers",
    lang: "python",
    title: "numbers to memorize",
    groups: [
      {
        title: "character codes",
        rows: [
          {
            code: 'ord("0"), ord("A"), ord("a"), ord(" ")',
            out: "48 65 97 32",
            why: "Digits start at 48, uppercase at 65 and lowercase at 97.",
          },
          {
            code: 'ord("a") - ord("A"), chr(65 + 25), chr(97 + 25)',
            out: "32 Z z",
            why: "The two cases are 32 apart, and the alphabet has 26 letters.",
          },
        ],
      },
      {
        title: "bases",
        rows: [
          {
            code: "0b1010, 0o17, 0xFF",
            out: "10 15 255",
            why: "0b is binary, 0o is octal and 0x is hex. In C, a plain leading 0 means octal.",
          },
          {
            code: "bin(10), oct(8), hex(255)",
            out: "0b1010 0o10 0xff",
            why: "Hex A to F are 10 to 15.",
          },
          {
            code: "2 ** 8, 2 ** 10, 2 ** 16",
            out: "256 1024 65536",
            why: "An int in C holds up to 2 to the 31st minus 1, which is 2147483647.",
          },
        ],
      },
      {
        title: "bit operations",
        rows: [
          {
            code: "5 & 3, 5 | 3, 5 ^ 3",
            out: "1 7 6",
            why: "5 is 101 and 3 is 011. & needs both bits, | needs either, ^ needs them different.",
          },
          {
            code: "3 << 2, 20 >> 2, ~5",
            out: "12 5 -6",
            why: "<< n multiplies by 2 to the n, >> n divides by it, and ~x is -(x + 1).",
          },
        ],
      },
    ],
  },
  {
    id: "python",
    lang: "python",
    title: "Python",
    groups: [
      {
        title: "slices and range",
        rows: [
          {
            code: '"python"[1:4], "python"[-2:], "python"[:-2]',
            out: "yth on pyth",
            why: "A slice starts at the first index and stops before the second. -1 is the last item.",
          },
          {
            code: '"python"[::-1], "python"[::2]',
            out: "nohtyp pto",
            why: "The third number is the step. -1 reverses.",
          },
          {
            code: "list(range(2, 11, 3)), list(range(5, 0, -2))",
            out: "[2, 5, 8] [5, 3, 1]",
            why: "range stops before the end value, going up or down.",
          },
        ],
      },
      {
        title: "lists",
        rows: [
          {
            code: "a = [1, 2, 3]; b = a; b.append(9); print(len(a))",
            out: "4",
            why: "b = a is the same list.",
          },
          {
            code: "a = [1, 2, 3]; b = a[:]; b.append(9); print(len(a))",
            out: "3",
            why: "a[:] is a copy.",
          },
          {
            code: "a = [3, 1, 2]; print(a.pop(), a.pop(0), a)",
            out: "2 3 [1]",
            why: "pop() takes the last item, and pop(0) takes the first.",
          },
          {
            code: "print([3, 1, 2].sort(), sorted([3, 1, 2]))",
            out: "None [1, 2, 3]",
            why: "sort() changes the list and returns None. sorted() returns a new list.",
          },
          {
            code: "[0] * 3, [1, 2] + [3]",
            out: "[0, 0, 0] [1, 2, 3]",
            why: "* repeats a list, and + joins two lists.",
          },
          {
            code: "[v * 2 for v in range(5) if v % 2]",
            out: "[2, 6]",
            why: "The if keeps odd v, since 1 is truthy.",
          },
        ],
      },
      {
        title: "numbers",
        rows: [
          {
            code: "7 / 2, 6 / 2, 7 // 2, -7 // 2",
            out: "3.5 3.0 3 -4",
            why: "/ always gives a float. // rounds down, even for negatives.",
          },
          {
            code: "int(-3.7), round(2.5), round(3.5)",
            out: "-3 2 4",
            why: "int() cuts toward zero. round() sends a .5 to the nearest even number.",
          },
          {
            code: "7 % 3, -7 % 3, 7 % -3",
            out: "1 2 -2",
            why: "% takes the sign of the right side.",
          },
          {
            code: "2 ** 3 ** 2, -2 ** 2",
            out: "512 -4",
            why: "** runs right to left, and it runs before the minus sign.",
          },
        ],
      },
      {
        title: "truth and logic",
        rows: [
          {
            code: "0 or 5, 3 and 5, 3 and 0",
            out: "5 5 0",
            why: "or gives the first truthy value. and gives the first falsy value, or the last one.",
          },
          {
            code: 'bool(""), bool("0"), bool([]), bool([0])',
            out: "False True False True",
            why: "0, empty strings, empty containers and None are falsy. Everything else is truthy.",
          },
          {
            code: "1 < 2 < 3, 3 > 2 > 1",
            out: "True True",
            why: "Python chains comparisons, so this means 3 > 2 and 2 > 1.",
          },
        ],
      },
      {
        title: "strings",
        rows: [
          {
            code: '"ab" * 3, "-".join("abc")',
            out: "ababab a-b-c",
            why: "* repeats, and join puts the separator between items.",
          },
          {
            code: '"a,b,c".split(","), "a b  c".split()',
            out: "['a', 'b', 'c'] ['a', 'b', 'c']",
            why: "split() with no argument skips repeated spaces.",
          },
          {
            code: '"banana".count("a"), "banana".find("n"), "banana".find("z")',
            out: "3 2 -1",
            why: "find gives the first index, or -1 when it's missing.",
          },
        ],
      },
    ],
  },
  {
    id: "javascript",
    lang: "js",
    title: "JavaScript",
    groups: [
      {
        title: "plus and types",
        rows: [
          {
            code: '"5" + 2, "5" * 2, "5" - 2',
            out: "52 10 3",
            why: "+ joins when either side is a string. The other math operators turn strings into numbers.",
          },
          {
            code: '1 + 2 + "3", "1" + 2 + 3',
            out: "33 123",
            why: "+ runs left to right, so numbers add until a string shows up.",
          },
          {
            code: "7 / 2, Math.floor(-7 / 2), Math.trunc(-7 / 2)",
            out: "3.5 -4 -3",
            why: "/ keeps the decimal. floor goes down, and trunc goes toward zero.",
          },
          {
            code: 'parseInt("12px"), Number("12px"), Number("")',
            out: "12 NaN 0",
            why: "parseInt reads digits until a non-digit. Number needs the whole string.",
          },
          {
            code: "typeof null, typeof [], typeof NaN, typeof undefined",
            out: "object object number undefined",
            why: "null and arrays both say object.",
          },
          {
            code: '1 == "1", 1 === "1", null == undefined, NaN == NaN',
            out: "true false true false",
            why: "== converts types and === doesn't. NaN equals nothing.",
          },
        ],
      },
      {
        title: "truthy and falsy",
        rows: [
          {
            code: 'Boolean([]), Boolean({}), Boolean("0")',
            out: "true true true",
            why: "Empty arrays and objects are truthy, unlike Python.",
          },
          {
            code: 'Boolean(0), Boolean(""), Boolean(NaN), Boolean(null)',
            out: "false false false false",
            why: "The falsy values are 0, an empty string, NaN, null, undefined and false.",
          },
          {
            code: "0 || 7, 0 ?? 7, null ?? 7, 0 && 7, 3 && 7",
            out: "7 0 7 0 7",
            why: "|| skips any falsy value. ?? skips only null and undefined.",
          },
        ],
      },
      {
        title: "arrays",
        rows: [
          {
            code: "const a = [1, 2, 3]; console.log(a.push(4), a.pop(), a.shift(), a)",
            out: "4 4 1 [2, 3]",
            why: "push returns the new length. pop and shift return the removed item.",
          },
          {
            code: "[1, 2, 3, 4].slice(1, 3)",
            out: "[2, 3]",
            why: "slice copies, and it stops before the end index.",
          },
          {
            code: "const a = [1, 2, 3, 4]; a.splice(1, 2); console.log(a)",
            out: "[1, 4]",
            why: "splice(start, count) removes items in place.",
          },
          {
            code: "[10, 9, 1].sort(), [10, 9, 1].sort((a, b) => a - b)",
            out: "[1, 10, 9] [1, 9, 10]",
            why: "sort() compares as strings unless it gets a compare function.",
          },
          {
            code: "[1, 2, 3].map(v => v * 2), [1, 2, 3].filter(v => v % 2)",
            out: "[2, 4, 6] [1, 3]",
            why: "map changes every item, and filter keeps the truthy ones.",
          },
          {
            code: "[1, 2, 3].reduce((s, v) => s + v, 0), [1, 2] + [3]",
            out: "6 1,23",
            why: "reduce carries s along. + turns arrays into strings.",
          },
        ],
      },
      {
        title: "functions and scope",
        rows: [
          {
            code: "const f = x => x * 2; const g = x => { x * 2 }; console.log(f(3), g(3))",
            out: "6 undefined",
            why: "With no braces the value is returned. With braces you need return.",
          },
          {
            code: "for (var i = 0; i < 3; i++) {} console.log(i)",
            out: "3",
            why: "var lives outside the loop, and the loop ends when i hits 3.",
          },
          {
            code: "for (var i = 0; i < 3; i++) setTimeout(() => console.log(i), 0)",
            out: "3 3 3",
            why: "The callbacks run after the loop and share one i.",
          },
          {
            code: "for (let i = 0; i < 3; i++) setTimeout(() => console.log(i), 0)",
            out: "0 1 2",
            why: "let gives each turn of the loop its own i.",
          },
          {
            code: 'setTimeout(() => console.log("a"), 0); console.log("b")',
            out: "b a",
            why: "A setTimeout callback waits until the current code is done.",
          },
          {
            code: "let i = 5; console.log(i++, ++i, i)",
            out: "5 7 7",
            why: "i++ gives the old value, and ++i gives the new one.",
          },
        ],
      },
    ],
  },
  {
    id: "c",
    lang: "c",
    title: "C",
    groups: [
      {
        title: "division and casts",
        rows: [
          {
            code: 'printf("%d %d %d %d", 7 / 2, -7 / 2, 7 % 3, -7 % 3);',
            out: "3 -3 1 -1",
            why: "int division cuts toward zero, and % takes the sign of the left side.",
          },
          {
            code: 'printf("%.1f %.1f", (float)7 / 2, (float)(7 / 2));',
            out: "3.5 3.0",
            why: "A cast on 7 happens before the division. A cast on (7 / 2) happens after.",
          },
          {
            code: 'printf("%.1f %.1f", 7 / 2 * 1.0, 7 * 1.0 / 2);',
            out: "3.0 3.5",
            why: "It runs left to right, so 7 / 2 is already 3 before the 1.0 arrives.",
          },
        ],
      },
      {
        title: "increments",
        rows: [
          {
            code: 'int i = 5; int a = i++; int b = ++i; printf("%d %d %d", a, b, i);',
            out: "5 7 7",
            why: "i++ gives the old value, and ++i gives the new one.",
          },
          {
            code: 'int i = 0, s = 0; while (i++ < 3) s += i; printf("%d %d", s, i);',
            out: "6 4",
            why: "The test uses the old i, and i still goes up on the failing test.",
          },
        ],
      },
      {
        title: "characters",
        rows: [
          {
            code: "printf(\"%d %d %d\", '0', 'A', 'a');",
            out: "48 65 97",
            why: "A char is a small number.",
          },
          {
            code: "printf(\"%c %d %c\", 'a' + 2, '7' - '0', 'a' - 32);",
            out: "c 7 A",
            why: "Subtracting '0' turns a digit character into its number.",
          },
        ],
      },
      {
        title: "pointers and arrays",
        rows: [
          {
            code: 'int a[] = {5, 6, 7, 8}; int *p = a + 1; printf("%d %d %d", *p, *(p + 1), p[-1]);',
            out: "6 7 5",
            why: "p + 1 moves one element, and p[i] means *(p + i).",
          },
          {
            code: 'int a[] = {5, 6, 7}; int *p = a; printf("%d %d", *p + 1, *(p + 1));',
            out: "6 6",
            why: "*p + 1 adds to the value. *(p + 1) reads the next item.",
          },
          {
            code: 'int a[] = {5, 6, 7}; int *p = a; int v = *p++; printf("%d %d", v, *p);',
            out: "5 6",
            why: "*p++ reads the item, then moves the pointer.",
          },
          {
            code: 'int x = 3; int *p = &x; *p += 4; printf("%d", x);',
            out: "7",
            why: "*p is x itself.",
          },
        ],
      },
      {
        title: "strings",
        rows: [
          {
            code: 'char s[] = "abc"; printf("%d %d", (int)strlen(s), (int)sizeof(s));',
            out: "3 4",
            why: "sizeof counts the ending null character, and strlen doesn't.",
          },
          {
            code: 'char s[] = "python"; printf("%s ", s + 2); s[3] = 0; printf("%s", s);',
            out: "thon pyt",
            why: "s + 2 starts at index 2, and a 0 ends the string.",
          },
        ],
      },
      {
        title: "traps",
        rows: [
          {
            code: 'int v = 5; if (v = 0) v = 9; printf("%d", v);',
            out: "0",
            why: "A single = assigns. The result is 0, which is false.",
          },
          {
            code: 'int v = 1, c = 0; switch (v) { case 1: c += 1; case 2: c += 2; break; case 3: c += 3; } printf("%d", c);',
            out: "3",
            why: "Without break, a case falls into the next one.",
          },
          {
            code: 'printf("%d %d", 010, 0x1F);',
            out: "8 31",
            why: "A leading 0 is octal, and 0x is hex.",
          },
          {
            code: 'unsigned char u = 250; u += 10; int a[3] = {1}; printf("%d %d", u, a[2]);',
            out: "4 0",
            why: "An unsigned char wraps at 256. Missing array values become 0.",
          },
        ],
      },
      {
        title: "printf",
        rows: [
          {
            code: 'printf("%05d|%x|%o|%c", 42, 255, 8, 65);',
            out: "00042|ff|10|A",
            why: "%05d pads with zeros, %x is hex, %o is octal and %c is a character.",
          },
          {
            code: 'printf("%.2f|%.0f|%s", 3.14159, 2.7, "hi");',
            out: "3.14|3|hi",
            why: "The number after the dot is how many decimals to round to.",
          },
        ],
      },
    ],
  },
];
