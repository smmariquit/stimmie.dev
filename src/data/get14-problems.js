// src/data/get14-problems.js

// Practice problems for /get14. Every answer is checked against a real run by
// scripts/verify-get14.mjs.

import { coding } from "./get14-coding.js";

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

const trace = [
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
  {
    id: "c-deref-copy",
    lang: "c",
    level: 1,
    code: `int y = 25;
int *p = &y;
int x = *p;
y = 30;
printf("%d %d %d", y, x, *p);`,
    answer: "30 25 30",
    why: "x got a copy of 25 before y changed. *p reads y itself, so it shows the new 30.",
  },
  {
    id: "c-write-through",
    lang: "c",
    level: 1,
    code: `int a = 5, b = 9;
int *p = &a;
*p = b;
b = 1;
printf("%d %d", a, b);`,
    answer: "9 1",
    why: "*p = b copies 9 into a, because p points to a. Changing b afterward doesn't touch a.",
  },
  {
    id: "c-pointer-copy",
    lang: "c",
    level: 1,
    code: `int x = 10;
int *p1 = &x, *p2;
p2 = p1;
*p2 += 5;
printf("%d %d", x, *p1);`,
    answer: "15 15",
    why: "p2 = p1 makes both pointers hold the address of x, so a write through p2 shows up through p1 too.",
  },
  {
    id: "c-array-name",
    lang: "c",
    level: 1,
    code: `int list[4] = {10, 20, 30, 40};
int *p = list;
printf("%d %d %d", *p, *(p + 2), p[3]);`,
    answer: "10 30 40",
    why: "An array name is the address of its first element. *p is list[0], *(p + 2) is list[2] and p[3] means *(p + 3).",
  },
  {
    id: "c-type-sizes",
    lang: "c",
    level: 1,
    code: `printf("%d %d %d %d", (int)sizeof(char), (int)sizeof(int),
       (int)sizeof(float), (int)sizeof(double));`,
    answer: "1 4 4 8",
    why: "A char takes 1 byte, an int 4, a float 4 and a double 8.",
  },
  {
    id: "c-step-by-type",
    lang: "c",
    level: 2,
    code: `int a[] = {5, 6, 7, 8};
int *p = a;
p++;
p += 2;
printf("%d %d", *p, (int)(p - a));`,
    answer: "8 3",
    why: "Each step moves by one int, not one byte. p ends at index 3, so *p is 8 and p - a is 3.",
  },
  {
    id: "c-post-increment-deref",
    lang: "c",
    level: 2,
    code: `int a[] = {1, 2, 3};
int *p = a;
printf("%d ", *p++);
printf("%d ", *p);
printf("%d ", (*p)++);
printf("%d", a[1]);`,
    answer: "1 2 2 3",
    why: "*p++ reads a[0] and then moves p to a[1]. (*p)++ adds one to the value a[1] itself, and it prints the old 2 first.",
  },
  {
    id: "c-pointer-sum",
    lang: "c",
    level: 2,
    code: `int list[5] = {4, 8, 15, 16, 23};
int *p, sum = 0;
for (p = list; p <= &list[4]; ++p)
    sum = sum + *p;
printf("%d", sum);`,
    answer: "66",
    why: "The loop visits every element through the pointer. The last visited address is &list[4], and 4 + 8 + 15 + 16 + 23 is 66.",
  },
  {
    id: "c-upper-loop",
    lang: "c",
    level: 2,
    code: `char name[] = "ab c";
char *p;
for (p = name; *p != '\\0'; ++p)
    if (*p >= 'a' && *p <= 'z')
        *p -= 32;
printf("%s", name);`,
    answer: "AB C",
    why: "Lowercase letters are 32 above uppercase ones. The space is skipped and the loop stops at the null character.",
  },
  {
    id: "c-substring",
    lang: "c",
    level: 2,
    code: `char name[] = "Mitch Andaya";
char *p = &name[6];
printf("%s|%s", p, name + 3);`,
    answer: "Andaya|ch Andaya",
    why: "A pointer to a middle character acts as the start of a shorter string. &name[6] starts at the A, and name + 3 starts at the c.",
  },
  {
    id: "c-call-by-value",
    lang: "c",
    level: 2,
    code: `#include <stdio.h>

void bump(int n) { n += 1; }
void pbump(int *n) { *n += 1; }

int main(void) {
    int x = 5;
    bump(x);
    pbump(&x);
    pbump(&x);
    printf("%d", x);
    return 0;
}`,
    answer: "7",
    why: "bump changes a copy, so it does nothing to x. Each pbump call adds 1 to the real x, so 5 becomes 7.",
  },
  {
    id: "c-pointer-compare",
    lang: "c",
    level: 2,
    code: `int a[] = {1, 2, 3};
int *p = a, *q = a + 2;
printf("%d %d %d", p < q, (int)(q - p), *q - *p);`,
    answer: "1 2 2",
    why: "p < q is true, which prints 1. Subtracting two pointers counts elements, so q - p is 2. *q - *p is 3 - 1.",
  },
  {
    id: "c-byte-distance",
    lang: "c",
    level: 3,
    code: `int a[5];
char *c1 = (char *)&a[0], *c2 = (char *)&a[3];
printf("%d %d", (int)(&a[3] - &a[0]), (int)(c2 - c1));`,
    answer: "3 12",
    why: "Subtracting int pointers counts ints, which gives 3. The same two addresses seen as char pointers are 3 * 4 = 12 bytes apart.",
  },
  {
    id: "c-reverse-pointers",
    lang: "c",
    level: 3,
    code: `char s[] = "Mitch";
char *a = s, *b = s + 4;
while (a < b) {
    char t = *a;
    *a = *b;
    *b = t;
    a++;
    b--;
}
printf("%s", s);`,
    answer: "hctiM",
    why: "The two pointers swap letters and walk toward each other, and they stop when they meet in the middle. The string reads backward.",
  },
  {
    id: "c-find-end",
    lang: "c",
    level: 3,
    code: `char s[] = "pointers";
char *p = s;
while (*p)
    p++;
printf("%d", (int)(p - s));`,
    answer: "8",
    why: "p stops on the null character, which counts as false. That's 8 steps from the start, so p - s is the length.",
  },
  {
    id: "c-out-params",
    lang: "c",
    level: 3,
    code: `#include <stdio.h>

void minmax(int *lo, int *hi, int a, int b) {
    if (a < b) { *lo = a; *hi = b; }
    else       { *lo = b; *hi = a; }
}

int main(void) {
    int low, high;
    minmax(&low, &high, 9, 3);
    printf("%d %d", low, high);
    return 0;
}`,
    answer: "3 9",
    why: "The function writes its results through the two addresses, so low and high are set without any return value.",
  },
  {
    id: "c-array-param",
    lang: "c",
    level: 3,
    code: `#include <stdio.h>

void f(int a[]) { a[0] = 99; a = 0; }

int main(void) {
    int v[3] = {1, 2, 3};
    f(v);
    printf("%d %d", v[0], v[1]);
    return 0;
}`,
    answer: "99 2",
    why: "An array argument is passed as an address, so a[0] = 99 changes the real array. a = 0 only changes the function's own copy of that address.",
  },
  {
    id: "c-skip-by-two",
    lang: "c",
    level: 3,
    code: `int a[] = {1, 2, 3, 4, 5, 6};
int *p = a, s = 0;
while (p < a + 6) {
    s += *p;
    p += 2;
}
printf("%d", s);`,
    answer: "9",
    why: "p visits a[0], a[2] and a[4], which hold 1, 3 and 5. The next step lands on a + 6 and ends the loop.",
  },
  {
    id: "c-offsets",
    lang: "c",
    level: 3,
    code: `int a[] = {10, 20, 30, 40, 50};
int *p = &a[1];
printf("%d %d %d", p[1], p[-1], (int)(&p[2] - a));`,
    answer: "30 10 3",
    why: "A pointer can be indexed from where it points, and a negative index goes backward. &p[2] is a[3], which is 3 ints from a.",
  },
  {
    id: "c-deref-precedence",
    lang: "c",
    level: 3,
    code: `int a[] = {10, 20, 30};
int *p = a + 1;
printf("%d %d %d %d", *p + 1, *(p + 1), *p - 1, *(p - 1));`,
    answer: "21 30 19 10",
    why: "*p + 1 adds to the value, and *(p + 1) reads the next element. The brackets decide which one happens.",
  },
  {
    id: "c-count-odd",
    lang: "c",
    level: 3,
    code: `int v[] = {4, 6, 5, 7, 8, 11, 10, 3, 18, 9};
int *p = v, odd = 0;
while (p < v + 10) {
    if (*p % 2)
        odd++;
    p++;
}
printf("%d odd", odd);`,
    answer: "5 odd",
    why: "Odd numbers leave a remainder of 1, which counts as true. The odd ones are 5, 7, 11, 3 and 9.",
  },
  {
    id: "c-pointer-to-pointer",
    lang: "c",
    level: 4,
    code: `int x = 1, y = 2;
int *p = &x;
int **pp = &p;
**pp = 10;
*pp = &y;
**pp += 5;
printf("%d %d %d", x, y, *p);`,
    answer: "10 7 7",
    why: "**pp = 10 writes into x. *pp = &y then repoints p at y, so **pp += 5 changes y, and *p now reads y.",
  },
  {
    id: "c-capitalize-words",
    lang: "c",
    level: 4,
    code: `char s[] = "mitCh anDaya";
char *p = s;
int start = 1;
for (; *p; p++) {
    if (*p == ' ') {
        start = 1;
    } else if (start) {
        if (*p >= 'a' && *p <= 'z') *p -= 32;
        start = 0;
    } else if (*p >= 'A' && *p <= 'Z') {
        *p += 32;
    }
}
printf("%s", s);`,
    answer: "Mitch Andaya",
    why: "The first letter of each word is made uppercase, and every other letter is made lowercase. A space starts a new word.",
  },
  {
    id: "c-edit-string",
    lang: "c",
    level: 4,
    code: `char a[] = "hello";
char *p = a;
*p = 'j';
p += 4;
*p = 'y';
printf("%s %c", a, *(a + 1));`,
    answer: "jelly e",
    why: "p starts at h, so *p = 'j' makes jello. p += 4 lands on the last letter, which becomes y. a + 1 is still e.",
  },
  {
    id: "py-func-reference",
    lang: "python",
    level: 2,
    code: `def add(lst):
    lst.append(1)
    lst = [9]
    lst.append(2)

a = [0]
add(a)
print(a)`,
    answer: "[0, 1]",
    why: "The first append changes the list that a refers to. lst = [9] then makes lst refer to a new list, so the second append never reaches a.",
  },
  {
    id: "py-shallow-copy",
    lang: "python",
    level: 3,
    code: `a = [[1, 2], [3]]
b = a[:]
b[0].append(9)
b.append([4])
print(a, len(b))`,
    answer: "[[1, 2, 9], [3]] 3",
    why: "a[:] copies only the outer list, so both lists share the same inner lists. The append to b[0] shows up in a, but the new item stays in b only.",
  },
  {
    id: "py-is-vs-equal",
    lang: "python",
    level: 3,
    code: `a = [1]
b = [1]
c = a
print(a == b, a is b, a is c)`,
    answer: "True False True",
    why: "== compares the contents and is compares whether it's the same object. a and b hold equal contents but are different lists, and c is the same list as a.",
  },
  {
    id: "js-object-reference",
    lang: "js",
    level: 2,
    code: `function f(o) {
  o.x = 1;
  o = { x: 2 };
  o.y = 3;
}
const a = {};
f(a);
console.log(JSON.stringify(a));`,
    answer: '{"x":1}',
    why: "o.x = 1 changes the object that a refers to. o = { x: 2 } then points o at a new object, so the y never reaches a.",
  },
  {
    id: "js-const-array",
    lang: "js",
    level: 2,
    code: `const a = [1];
const b = a;
b.push(2);
console.log(a.length, a === b);`,
    answer: "2 true",
    why: "const stops reassignment, not changes to the array. b is the same array as a, so push shows up in both.",
  },
  {
    id: "js-closure-counter",
    lang: "js",
    level: 3,
    code: `function make() {
  let n = 0;
  return () => ++n;
}
const c1 = make(), c2 = make();
c1();
c1();
console.log(c1(), c2());`,
    answer: "3 1",
    why: "Each call to make creates its own n. c1 has been called three times, and c2 has been called once.",
  },
  {
    id: "py-class-attribute",
    lang: "python",
    level: 3,
    code: `class Counter:
    total = 0

    def __init__(self):
        Counter.total += 1
        self.id = Counter.total

a = Counter()
b = Counter()
b.total = 10
print(a.id, b.id, Counter.total, a.total, b.total)`,
    answer: "1 2 2 2 10",
    why: "total is shared by the class. b.total = 10 creates a separate total on b only, so the class and a still see 2.",
  },
  {
    id: "py-super-override",
    lang: "python",
    level: 3,
    code: `class Animal:
    def speak(self):
        return self.sound() + "!"

    def sound(self):
        return "..."

class Dog(Animal):
    def sound(self):
        return "woof"

class Puppy(Dog):
    def sound(self):
        return super().sound() + " yip"

print(Animal().speak(), Dog().speak(), Puppy().speak())`,
    answer: "...! woof! woof yip!",
    why: "self.sound() picks the version in the object's own class. super().sound() in Puppy calls Dog's version and then adds yip.",
  },
  {
    id: "py-init-chain",
    lang: "python",
    level: 4,
    code: `class A:
    def __init__(self):
        print("A start")
        self.setup()
        print("A end")

    def setup(self):
        print("A setup")

class B(A):
    def __init__(self):
        print("B start")
        super().__init__()
        print("B end")

    def setup(self):
        print("B setup")

B()`,
    answer: "B start\nA start\nB setup\nA end\nB end",
    why: "B() runs B's __init__ first. super().__init__() runs A's, and its self.setup() picks B's setup because the object is a B.",
  },
  {
    id: "js-lost-this",
    lang: "js",
    level: 3,
    code: `class Box {
  constructor() { this.n = 1; }
  get() { return this?.n; }
}
const b = new Box();
const g = b.get;
console.log(b.get(), g(), g.call({ n: 7 }));`,
    answer: "1 undefined 7",
    why: "A method called as b.get() has b as this. Pulled out as g(), it has no this, and class code runs in strict mode so this is undefined. call sets this to the object you pass.",
  },
  {
    id: "js-field-order",
    lang: "js",
    level: 4,
    code: `class A {
  constructor() { this.show(); }
  show() { console.log("A"); }
}
class B extends A {
  x = 5;
  show() { console.log(this.x); }
}
new B();`,
    answer: "undefined",
    why: "The parent constructor runs before the child's x = 5 is set. show is B's version, so it prints undefined.",
  },
];

export const problems = [...trace, ...coding];
