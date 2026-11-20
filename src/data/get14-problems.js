// src/data/get14-problems.js

// Practice problems for /get14. Every answer is checked against a real run by
// scripts/verify-get14.mjs.

export const LEVELS = ["easy", "medium", "hard", "spicy"];

export const LANGS = [
  { id: "python", label: "Python" },
  { id: "js", label: "JavaScript" },
  { id: "c", label: "C" },
];

// Trims the ends and trailing spaces so a stray space doesn't fail an answer.
export function normalize(text) {
  return text
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trimEnd())
    .join("\n")
    .trim();
}

export const problems = [
  {
    id: "py-sum-evens",
    lang: "python",
    level: 1,
    code: `total = 0
for i in range(1, 6):
    if i % 2 == 0:
        total += i
print(total)`,
    answer: "6",
    why: "range(1, 6) stops at 5, so only 2 and 4 get added.",
  },
  {
    id: "py-slices",
    lang: "python",
    level: 1,
    code: `s = "quarter"
print(s[0], s[-1], s[2:5])`,
    answer: "q r art",
    why: "s[-1] is the last letter. s[2:5] takes indexes 2, 3 and 4, and print puts a space between each value.",
  },
  {
    id: "py-division",
    lang: "python",
    level: 1,
    code: "print(7 // 2, 7 / 2, 7 % 2, 2 ** 3)",
    answer: "3 3.5 1 8",
    why: "// floors, / always gives a float, % is the remainder and ** is a power.",
  },
  {
    id: "py-negative-floor",
    lang: "python",
    level: 2,
    code: "print(-7 // 2, -7 % 2)",
    answer: "-4 1",
    why: "Python floors toward negative infinity, so -3.5 becomes -4. The remainder takes the sign of the divisor, since -4 * 2 + 1 is -7.",
  },
  {
    id: "py-alias",
    lang: "python",
    level: 2,
    code: `a = [1, 2, 3]
b = a
b.append(4)
c = a[:]
c.append(5)
print(a, c)`,
    answer: "[1, 2, 3, 4] [1, 2, 3, 4, 5]",
    why: "b is the same list as a, so the append shows up in a. a[:] makes a copy, so c changes alone.",
  },
  {
    id: "py-pattern",
    lang: "python",
    level: 2,
    code: `for i in range(1, 4):
    print("*" * i + "-" * (3 - i))`,
    answer: "*--\n**-\n***",
    why: "Each row has i stars and 3 - i dashes. The last row has zero dashes.",
  },
  {
    id: "py-default-list",
    lang: "python",
    level: 3,
    code: `def add(x, bag=[]):
    bag.append(x)
    return bag

print(add(1))
print(add(2))`,
    answer: "[1]\n[1, 2]",
    why: "The default list is made once, when the function is defined. Both calls append to that same list.",
  },
  {
    id: "py-for-else",
    lang: "python",
    level: 3,
    code: `for n in range(2, 10):
    if 9 % n == 0:
        break
else:
    print("none")
print(n)`,
    answer: "3",
    why: "The else of a for loop only runs when the loop never breaks. It breaks at 3, so only the last print runs.",
  },
  {
    id: "py-count-letters",
    lang: "python",
    level: 3,
    code: `word = "mississippi"
count = {}
for ch in word:
    count[ch] = count.get(ch, 0) + 1
print(count["s"], count["p"], len(count))`,
    answer: "4 2 4",
    why: "There are 4 s's and 2 p's. The dict has one key per distinct letter, which is m, i, s and p.",
  },
  {
    id: "py-recursion-order",
    lang: "python",
    level: 4,
    code: `def go(n):
    if n == 0:
        return
    print(n, end=" ")
    go(n - 1)
    print(n, end=" ")

go(3)`,
    answer: "3 2 1 1 2 3",
    why: "Each call prints on the way down, then prints again after its inner call returns.",
  },
  {
    id: "py-comprehension",
    lang: "python",
    level: 4,
    code: `nums = [x * x for x in range(6) if x % 2]
print(nums, nums[::-1][0], sum(nums))`,
    answer: "[1, 9, 25] 25 35",
    why: "x % 2 is truthy for odd x, so the list holds the squares of 1, 3 and 5. Reversing it puts 25 first.",
  },
  {
    id: "py-late-binding",
    lang: "python",
    level: 4,
    code: `fs = [lambda: i * 2 for i in range(3)]
print([f() for f in fs])`,
    answer: "[4, 4, 4]",
    why: "Each lambda looks up i when it's called, not when it's made. By then the loop is done and i is 2.",
  },
  {
    id: "js-continue",
    lang: "js",
    level: 1,
    code: `let s = 0;
for (let i = 0; i < 5; i++) {
  if (i % 2) continue;
  s += i;
}
console.log(s);`,
    answer: "6",
    why: "i % 2 is truthy for odd numbers, so those get skipped. 0 + 2 + 4 is 6.",
  },
  {
    id: "js-coercion",
    lang: "js",
    level: 1,
    code: 'console.log(1 + "2", "3" - 1, "5" * "2");',
    answer: "12 2 10",
    why: "+ with a string joins text. - and * only work on numbers, so the strings get converted.",
  },
  {
    id: "js-array-methods",
    lang: "js",
    level: 1,
    code: `const a = [3, 1, 2];
a.push(4);
a.shift();
console.log(a.length, a[0], a.join("-"));`,
    answer: "3 1 1-2-4",
    why: "push adds 4 to the end and shift removes the 3 from the front, which leaves 1, 2 and 4.",
  },
  {
    id: "js-equality",
    lang: "js",
    level: 2,
    code: 'console.log(0 == "", 0 === "", null == undefined, NaN == NaN);',
    answer: "true false true false",
    why: "== converts types first, and an empty string becomes 0. === doesn't convert. NaN is never equal to anything, not even itself.",
  },
  {
    id: "js-default-sort",
    lang: "js",
    level: 2,
    code: 'console.log([10, 9, 1, 100].sort().join(" "));',
    answer: "1 10 100 9",
    why: 'sort with no compare function sorts as text, so "10" and "100" come before "9".',
  },
  {
    id: "js-typeof",
    lang: "js",
    level: 2,
    code: "console.log(typeof null, typeof [], typeof NaN, typeof undefined);",
    answer: "object object number undefined",
    why: "typeof null is an old bug that stayed. Arrays are objects, and NaN is still a number type.",
  },
  {
    id: "js-var-timeout",
    lang: "js",
    level: 3,
    code: `for (var i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 0);
}
console.log("done");`,
    answer: "done\n3\n3\n3",
    why: "The timeouts run after the current code finishes, so done prints first. var makes one shared i, and it's already 3 by then.",
  },
  {
    id: "js-chain",
    lang: "js",
    level: 3,
    code: `const r = [1, 2, 3, 4, 5]
  .filter((x) => x % 2)
  .map((x) => x * x)
  .reduce((a, b) => a + b, 0);
console.log(r);`,
    answer: "35",
    why: "filter keeps 1, 3 and 5. map squares them to 1, 9 and 25, and reduce adds them up.",
  },
  {
    id: "js-references",
    lang: "js",
    level: 3,
    code: `const a = { n: 1 };
const b = a;
b.n++;
const c = { ...a };
c.n++;
console.log(a.n, b.n, c.n);`,
    answer: "2 2 3",
    why: "b points to the same object as a. The spread makes a copy, so c.n grows without touching a.",
  },
  {
    id: "js-hoisting",
    lang: "js",
    level: 4,
    code: `console.log(x);
var x = 5;
function f() {
  console.log(x);
  var x = 10;
}
f();
console.log(x);`,
    answer: "undefined\nundefined\n5",
    why: "var declarations are hoisted, but their values aren't. Inside f, the local x shadows the outer one and is still undefined at the log.",
  },
  {
    id: "js-truthy",
    lang: "js",
    level: 4,
    code: 'console.log(!!"0", !![], null ?? "x", 0 || "y", 0 ?? "z");',
    answer: "true true x y 0",
    why: "A non-empty string and an empty array are both truthy. || falls through on any falsy value, but ?? only falls through on null and undefined.",
  },
  {
    id: "js-banana",
    lang: "js",
    level: 4,
    code: `console.log(0.1 + 0.2 === 0.3, (0.1 + 0.2).toFixed(2));
console.log(parseInt("08px"), "b" + "a" + +"a" + "a");`,
    answer: "false 0.30\n8 baNaNa",
    why: 'Floats can\'t store 0.1 exactly, so the sum is a bit over 0.3. parseInt reads digits until it hits a letter, and +"a" is NaN.',
  },
  {
    id: "c-int-division",
    lang: "c",
    level: 1,
    code: `int x = 7, y = 2;
printf("%d %d %.1f\\n", x / y, x % y, (float)x / y);`,
    answer: "3 1 3.5",
    why: "int divided by int drops the fraction. Casting one side to float makes it a real division.",
  },
  {
    id: "c-loop-step",
    lang: "c",
    level: 1,
    code: `int i, s = 0;
for (i = 1; i <= 10; i += 3)
    s += i;
printf("%d %d\\n", i, s);`,
    answer: "13 22",
    why: "i takes 1, 4, 7 and 10, which add up to 22. The loop ends when i becomes 13.",
  },
  {
    id: "c-char-math",
    lang: "c",
    level: 1,
    code: `char c = 'A';
c += 2;
printf("%c %d\\n", c, c);`,
    answer: "C 67",
    why: "A char is a small number. 'A' is 65, so adding 2 gives 67, which is 'C'.",
  },
  {
    id: "c-increments",
    lang: "c",
    level: 2,
    code: `int a = 5, b;
b = a++;
int c = ++a;
printf("%d %d %d\\n", a, b, c);`,
    answer: "7 5 7",
    why: "a++ gives the old value and then adds 1. ++a adds 1 first and gives the new value.",
  },
  {
    id: "c-fallthrough",
    lang: "c",
    level: 2,
    code: `int n = 2;
switch (n) {
    case 1: printf("one ");
    case 2: printf("two ");
    case 3: printf("three ");
        break;
    default: printf("none ");
}`,
    answer: "two three",
    why: "With no break, case 2 falls through into case 3. The break there stops it before default.",
  },
  {
    id: "c-negative-division",
    lang: "c",
    level: 2,
    code: 'printf("%d %d\\n", -7 / 2, -7 % 2);',
    answer: "-3 -1",
    why: "C cuts toward zero, so -3.5 becomes -3. The remainder keeps the sign of the left side. Python gives -4 and 1 here.",
  },
  {
    id: "c-pointer-walk",
    lang: "c",
    level: 3,
    code: `int a[] = {10, 20, 30, 40};
int *p = a + 1;
printf("%d %d %d\\n", *p, p[2], *(p - 1));`,
    answer: "20 40 10",
    why: "p points at the second item. p[2] is two items after that, and p - 1 steps back to the first.",
  },
  {
    id: "c-swap",
    lang: "c",
    level: 3,
    code: `#include <stdio.h>

void swap(int a, int b) { int t = a; a = b; b = t; }
void pswap(int *a, int *b) { int t = *a; *a = *b; *b = t; }

int main(void) {
    int x = 1, y = 2;
    swap(x, y);
    printf("%d %d\\n", x, y);
    pswap(&x, &y);
    printf("%d %d\\n", x, y);
    return 0;
}`,
    answer: "1 2\n2 1",
    why: "swap only changes its own copies. pswap gets the addresses, so it changes the real x and y.",
  },
  {
    id: "c-string-cut",
    lang: "c",
    level: 3,
    code: `char s[] = "encore";
int n = 0;
while (s[n] != '\\0') n++;
s[2] = '\\0';
printf("%d %s %lu\\n", n, s, sizeof(s));`,
    answer: "6 en 7",
    why: "The word has 6 letters. Writing '\\0' at index 2 ends the string early, but the array still holds 7 bytes with the terminator.",
  },
  {
    id: "c-assign-in-if",
    lang: "c",
    level: 3,
    code: `int x = 0;
if (x = 5)
    printf("yes %d\\n", x);
else
    printf("no %d\\n", x);`,
    answer: "yes 5",
    why: "A single = assigns. The condition is the assigned value 5, which counts as true.",
  },
  {
    id: "c-unsigned-wrap",
    lang: "c",
    level: 4,
    code: `unsigned char c = 250;
c += 10;
printf("%d\\n", c);`,
    answer: "4",
    why: "An unsigned char holds 0 to 255. 260 wraps around, and 260 - 256 is 4.",
  },
  {
    id: "c-bitwise",
    lang: "c",
    level: 4,
    code: 'printf("%d %d %d %d\\n", 5 & 3, 5 | 3, 5 ^ 3, 1 << 4);',
    answer: "1 7 6 16",
    why: "5 is 101 and 3 is 011. AND gives 001, OR gives 111 and XOR gives 110. Shifting 1 left by 4 is 16.",
  },
  {
    id: "c-recursion-bits",
    lang: "c",
    level: 4,
    code: `#include <stdio.h>

int f(int n) { return n < 2 ? n : f(n / 2) + n % 2; }

int main(void) {
    printf("%d\\n", f(13));
    return 0;
}`,
    answer: "3",
    why: "f counts the 1 bits of n. 13 is 1101 in binary, which has three 1s.",
  },
];
