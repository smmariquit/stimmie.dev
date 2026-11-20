// src/data/get14-coding.js

// Fill-in-the-blank and write-the-code problems for /get14. The verifier runs
// every one, so the expected output of each blank and each test is real.
//
// kind "blank". code has one ____. answer and accept are the text that goes
// there, and expect is what the finished program prints.
//
// kind "write". solution is the model answer. tests is a list of
// [expression, expected] pairs that the solution must pass.

export const coding = [
  // fill in the blank
  {
    id: "py-blank-range",
    kind: "blank",
    lang: "python",
    level: 1,
    task: "Fill in the blank so it prints 15.",
    code: `total = 0
for i in range(1, ____):
    total += i
print(total)`,
    answer: "6",
    expect: "15",
    why: "range stops before its end value. 1 + 2 + 3 + 4 + 5 is 15, so the end value is 6.",
  },
  {
    id: "py-blank-slice",
    kind: "blank",
    lang: "python",
    level: 1,
    task: "Fill in the blank so it prints gram.",
    code: `s = "programming"
print(s[____])`,
    answer: "3:7",
    expect: "gram",
    why: "Index 3 is the g, and a slice stops before its end index. The letters at 3, 4, 5 and 6 are g, r, a and m.",
  },
  {
    id: "py-blank-comprehension",
    kind: "blank",
    lang: "python",
    level: 2,
    task: "Fill in the blank so it prints the first five squares.",
    code: `print([____ for n in range(1, 6)])`,
    answer: "n * n",
    accept: ["n ** 2", "n*n", "n**2", "pow(n, 2)"],
    expect: "[1, 4, 9, 16, 25]",
    why: "The expression before for is computed once for each n, so n * n squares every value from 1 to 5.",
  },
  {
    id: "py-blank-condition",
    kind: "blank",
    lang: "python",
    level: 2,
    task: "Fill in the blank so it counts the even numbers and prints 3.",
    code: `count = 0
for n in [3, 8, 5, 6, 2]:
    if ____:
        count += 1
print(count)`,
    answer: "n % 2 == 0",
    accept: ["not n % 2", "n%2==0", "not n%2", "n % 2 < 1"],
    expect: "3",
    why: "An even number leaves a remainder of 0 when divided by 2. The even numbers in the list are 8, 6 and 2.",
  },
  {
    id: "js-blank-reduce",
    kind: "blank",
    lang: "js",
    level: 2,
    task: "Fill in the blank so it prints 10.",
    code: `console.log([1, 2, 3, 4].reduce((s, v) => ____, 0));`,
    answer: "s + v",
    accept: ["v + s", "s+v", "v+s"],
    expect: "10",
    why: "reduce hands the running total s and the next value v to the function. Returning s + v adds each value to the total.",
  },
  {
    id: "js-blank-filter",
    kind: "blank",
    lang: "js",
    level: 2,
    task: "Fill in the blank so it keeps only the words longer than one letter.",
    code: `console.log(["a", "bb", "ccc"].filter(w => ____).join(","));`,
    answer: "w.length > 1",
    accept: ["w.length >= 2", "w.length>1", "w.length>=2", "w.length !== 1"],
    expect: "bb,ccc",
    why: "filter keeps each word for which the function returns true. bb and ccc have a length above 1.",
  },
  {
    id: "js-blank-let",
    kind: "blank",
    lang: "js",
    level: 3,
    task: "Fill in the blank so it prints 0, then 1, then 2 on separate lines.",
    code: `for (____ i = 0; i < 3; i++) setTimeout(() => console.log(i), 0);`,
    answer: "let",
    expect: "0\n1\n2",
    why: "let gives each turn of the loop its own i. With var, all three callbacks share one i and print 3 three times.",
  },
  {
    id: "c-blank-loop",
    kind: "blank",
    lang: "c",
    level: 1,
    task: "Fill in the blank so it prints 0 2 4 6 8.",
    code: `for (int i = 0; i < ____; i++)
    printf("%d ", i * 2);`,
    answer: "5",
    expect: "0 2 4 6 8",
    why: "i runs from 0 up to one less than the limit. Five turns print 0, 2, 4, 6 and 8.",
  },
  {
    id: "c-blank-pointer",
    kind: "blank",
    lang: "c",
    level: 2,
    task: "Fill in the blank so it prints 9.",
    code: `int a[] = {3, 6, 9};
int *p = a;
printf("%d", *(p + ____));`,
    answer: "2",
    expect: "9",
    why: "p + 2 moves two ints forward, which is a[2]. The 9 sits at index 2.",
  },
  {
    id: "c-blank-address",
    kind: "blank",
    lang: "c",
    level: 2,
    task: "Fill in the blank so it prints 7.",
    code: `#include <stdio.h>

void set(int *n) { *n = 7; }

int main(void) {
    int x = 0;
    set(____);
    printf("%d", x);
    return 0;
}`,
    answer: "&x",
    accept: ["& x"],
    expect: "7",
    why: "set needs the address of x so it can write into the real variable. The & operator gives that address.",
  },
  {
    id: "c-blank-format",
    kind: "blank",
    lang: "c",
    level: 2,
    task: "Fill in the blank so it prints 3.14.",
    code: `printf("%____", 3.14159);`,
    answer: ".2f",
    expect: "3.14",
    why: "%f prints a float. The number after the dot is how many decimals to keep, and it rounds.",
  },
  {
    id: "c-blank-swap",
    kind: "blank",
    lang: "c",
    level: 3,
    task: "Fill in the blank so it swaps a and b and prints 2 1.",
    code: `int a = 1, b = 2, t;
t = a;
a = b;
____;
printf("%d %d", a, b);`,
    answer: "b = t",
    accept: ["b=t"],
    expect: "2 1",
    why: "t holds the old a. After a takes b's value, b needs the old a back, which is t.",
  },

  // write the code
  {
    id: "py-write-count-vowels",
    kind: "write",
    lang: "python",
    level: 2,
    task: "Write count_vowels(s). It returns how many vowels (a, e, i, o, u, in either case) are in the string s.",
    starter: "def count_vowels(s):\n    ",
    solution: `def count_vowels(s):
    count = 0
    for ch in s.lower():
        if ch in "aeiou":
            count += 1
    return count`,
    tests: [
      ['count_vowels("education")', "5"],
      ['count_vowels("rhythm")', "0"],
      ['count_vowels("AEIOU")', "5"],
    ],
    why: "Lowercasing first means one check covers both cases. Then each letter that is in aeiou adds 1.",
  },
  {
    id: "py-write-fizz",
    kind: "write",
    lang: "python",
    level: 2,
    task: "Write fizz(n). It returns Fizz when n is a multiple of 3, Buzz when it's a multiple of 5, FizzBuzz when it's both, and otherwise the number as a string.",
    starter: "def fizz(n):\n    ",
    solution: `def fizz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)`,
    tests: [
      ["fizz(9)", "Fizz"],
      ["fizz(10)", "Buzz"],
      ["fizz(15)", "FizzBuzz"],
      ["fizz(7)", "7"],
    ],
    why: "Check the both case first. A multiple of 15 is also a multiple of 3 and 5, so it would match an earlier check.",
  },
  {
    id: "py-write-reverse-words",
    kind: "write",
    lang: "python",
    level: 3,
    task: "Write reverse_words(s). It returns the words of s in reverse order, joined by single spaces.",
    starter: "def reverse_words(s):\n    ",
    solution: `def reverse_words(s):
    return " ".join(s.split()[::-1])`,
    tests: [
      ['reverse_words("pen and paper")', "paper and pen"],
      ['reverse_words("one")', "one"],
      ['reverse_words("a b c d")', "d c b a"],
    ],
    why: "split() cuts the string into a list of words, [::-1] reverses the list, and join puts the spaces back.",
  },
  {
    id: "py-write-second-largest",
    kind: "write",
    lang: "python",
    level: 3,
    task: "Write second_largest(nums). It returns the second largest distinct value in the list. The list always has at least two distinct values.",
    starter: "def second_largest(nums):\n    ",
    solution: `def second_largest(nums):
    big = max(nums)
    second = None
    for n in nums:
        if n != big and (second is None or n > second):
            second = n
    return second`,
    tests: [
      ["second_largest([4, 9, 2, 9, 7])", "7"],
      ["second_largest([5, 5, 1])", "1"],
      ["second_largest([3, 1, 2])", "2"],
    ],
    why: "Skipping every copy of the biggest value handles duplicates. The best of what remains is the second largest.",
  },
  {
    id: "py-write-digit-sum",
    kind: "write",
    lang: "python",
    level: 3,
    task: "Write digit_sum(n). It returns the sum of the digits of the whole number n, without turning n into a string.",
    starter: "def digit_sum(n):\n    ",
    solution: `def digit_sum(n):
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total`,
    tests: [
      ["digit_sum(1234)", "10"],
      ["digit_sum(9)", "9"],
      ["digit_sum(1000)", "1"],
    ],
    why: "n % 10 is the last digit and n // 10 drops it. The loop repeats until no digits are left.",
  },
  {
    id: "js-write-longest",
    kind: "write",
    lang: "js",
    level: 2,
    task: "Write longest(words). It returns the longest word in the array. When two words tie, it returns the first one.",
    starter: "function longest(words) {\n  \n}",
    solution: `function longest(words) {
  let best = words[0];
  for (const w of words) {
    if (w.length > best.length) best = w;
  }
  return best;
}`,
    tests: [
      ['longest(["pen", "paper", "ink"])', "paper"],
      ['longest(["ab", "cd"])', "ab"],
      ['longest(["x"])', "x"],
    ],
    why: "Using > and not >= keeps the first word when there's a tie, because a tie never replaces best.",
  },
  {
    id: "js-write-sum-odd",
    kind: "write",
    lang: "js",
    level: 2,
    task: "Write sumOdd(arr). It returns the sum of the odd numbers in the array, or 0 when there are none.",
    starter: "function sumOdd(arr) {\n  \n}",
    solution: `function sumOdd(arr) {
  return arr.filter(v => v % 2 === 1).reduce((s, v) => s + v, 0);
}`,
    tests: [
      ["sumOdd([1, 2, 3, 4, 5])", "9"],
      ["sumOdd([2, 4])", "0"],
      ["sumOdd([])", "0"],
    ],
    why: "filter keeps the odd values and reduce adds them. The starting value 0 makes an empty list give 0.",
  },
  {
    id: "c-write-fact",
    kind: "write",
    lang: "c",
    level: 2,
    task: "Write int fact(int n). It returns n factorial, which is 1 * 2 * ... * n. fact(0) is 1.",
    starter: "int fact(int n) {\n    \n}",
    solution: `int fact(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}`,
    tests: [
      ["fact(5)", "120"],
      ["fact(0)", "1"],
      ["fact(1)", "1"],
    ],
    why: "result starts at 1, so n = 0 and n = 1 skip the loop and return 1. Otherwise it multiplies in 2 up to n.",
  },
  {
    id: "c-write-strlen",
    kind: "write",
    lang: "c",
    level: 3,
    task: "Write int my_strlen(char *s). It returns the length of the string. Use a pointer and no index, and don't call strlen.",
    starter: "int my_strlen(char *s) {\n    \n}",
    solution: `int my_strlen(char *s) {
    char *p = s;
    while (*p)
        p++;
    return p - s;
}`,
    tests: [
      ['my_strlen("pointers")', "8"],
      ['my_strlen("")', "0"],
      ['my_strlen("a b")', "3"],
    ],
    why: "p walks until it reaches the null character, which counts as false. Subtracting the start pointer gives the number of steps.",
  },
  {
    id: "c-write-count-char",
    kind: "write",
    lang: "c",
    level: 4,
    task: "Write int count_char(char *s, char c). It returns how many times the character c appears in the string s. Use a pointer.",
    starter: "int count_char(char *s, char c) {\n    \n}",
    solution: `int count_char(char *s, char c) {
    int n = 0;
    for (; *s; s++)
        if (*s == c)
            n++;
    return n;
}`,
    tests: [
      ["count_char(\"banana\", 'a')", "3"],
      ["count_char(\"banana\", 'n')", "2"],
      ["count_char(\"banana\", 'z')", "0"],
    ],
    why: "s is a copy of the address, so moving it with s++ doesn't affect the caller. The loop ends at the null character.",
  },
];
