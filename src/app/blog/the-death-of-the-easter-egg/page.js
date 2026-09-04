// src/app/blog/the-death-of-the-easter-egg/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "The Death of the Easter Egg",
  description:
    "Somebody hid a flight simulator inside a spreadsheet once. Then the code started writing itself, and nobody had a reason to leave a piece of themselves behind.",
};

function Figure({ src, alt, caption, narrow }) {
  return (
    <figure className={narrow ? "my-6 mx-auto max-w-sm" : "my-6"}>
      <img
        src={src}
        alt={alt}
        className="w-full border-2 border-current"
        loading="lazy"
      />
      <figcaption className="mt-2 text-sm neo-muted font-mono">
        {caption}
      </figcaption>
    </figure>
  );
}

export default function TheDeathOfTheEasterEggPage() {
  return (
    <PageShell
      title="The Death of the Easter Egg"
      current="/blog"
      maxWidth="52rem"
    >
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">September 4, 2026</p>

        <p>
          There is a flight simulator inside Microsoft Excel 97.
        </p>

        <p>
          I wasn&apos;t around for it. I learned about it the way most people my
          age learn about anything from before they were born, which is to say
          late at night, from a video, with the volume low, and with the vague
          feeling of having missed something. You open a blank sheet,
          press F5, type <code>X97:L97</code>, press Tab, and hold Ctrl and
          Shift while clicking the chart wizard. The spreadsheet disappears.
          What is left is a range of purple hills under a black sky, and you
          are flying over them.
        </p>

        <Figure
          src="/blog/easter-egg/excel97-purple-hills.jpg"
          alt="Jagged purple hills under a black sky with a thin orange sunset line at the horizon, rendered in the Excel 97 flight simulator"
          caption="This is a screenshot of Microsoft Excel."
        />

        <p>
          If you flew long enough you would find a grey monolith standing in
          the middle of the terrain, and on its face the names of the people
          who built Excel would scroll upward, slowly, like the credits of a
          film. There was a lagoon nearby. Nobody asked for a lagoon.
        </p>

        <Figure
          src="/blog/easter-egg/excel97-blue-terrain.jpg"
          alt="Rolling blue mountains in the Excel 97 flight simulator, low resolution"
          caption="The same terrain, another day. It was generated at random, which meant no two people saw quite the same hills."
        />

        <p>
          I think about this more than I probably should. A handful of
          engineers put an entire small world inside a program meant for
          accountants, and for a few years every finance department on earth
          carried it around without knowing, which is a strange thing to be
          moved by and yet here I am. It had no purpose and it wasn&apos;t in the
          manual. It existed, as far as I can tell, because the people who made
          Excel wanted, in some quiet and probably unexamined way, to be found.
        </p>

        <p>
          That won&apos;t happen again. I want to be careful about how I say this,
          because it sounds like nostalgia for a decade I didn&apos;t live in, and I
          am suspicious of my own nostalgia in general. But I&apos;ve come to
          believe that the easter egg, the real kind, the kind no one approved,
          is gone, and that I know, or at least strongly suspect, what took it.
        </p>

        <h2>The first one</h2>

        <p>
          The first easter egg came out of a grudge. In 1979 Atari did not print the names of its programmers anywhere. A
          person would spend the better part of a year alone with four
          kilobytes, and the box would go out with a painting of a dragon and no
          human being credited on it. Warren Robinett wrote Adventure under
          those terms and decided he would not be erased.
        </p>

        <p>
          So he built a room. To reach it you had to find a single grey dot
          hidden in a wall, carry it across the map, and squeeze through a gap
          that was not supposed to be there. Inside, in flashing letters, the
          game said: Created by Warren Robinett. He told no one and left the company. About a year later a teenager found the room and wrote to
          Atari, and Atari, after working out what a recall would cost, decided
          to leave it in.
        </p>

        <Figure
          src="/blog/easter-egg/adventure-robinett-room.png"
          alt="Blocky Atari 2600 screen showing the words Created by Warren Robinett in a hidden room of Adventure"
          caption="The room."
        />

        <p>
          I keep returning to what the room needed in order to exist. It
          needed one person holding the entire program in their head, with no
          one else reading the code, on a cartridge that shipped once and could
          never be patched. Most of all it needed someone who had been in there
          long enough to feel that the thing was theirs.
        </p>

        <h2>What happened to it first</h2>

        <p>
          Several things had already gone wrong with the easter egg by the
          time I was old enough to notice, though I did not notice.
        </p>

        <p>
          Microsoft stopped allowing them in 2002. Bill Gates had sent a memo
          about trust, and somebody realised that a customer like a government
          cannot buy a product with code in it that nobody documented. A hidden flight simulator and a hidden backdoor look the same from
          the outside. So the flight simulator went, and the pinball game in
          Word went with it, and no one at Microsoft has been allowed to hide
          anything since. That&apos;s compliance for you.
        </p>

        <p>
          Everywhere else, code review did the same job. A secret needs a place to
          hide, and there is nowhere to hide in a codebase where two other
          people read every line before it is merged. If someone opens a pull
          request and finds four hundred lines of terrain generation in a
          spreadsheet, they ask what it is for, and the moment there is an
          answer it becomes a ticket with an owner.
        </p>

        <p>
          Around the same time software started being measured, and an easter
          egg has no number attached to it, so it never got scheduled. And the
          internet made the whole thing brief anyway. A hidden room in a game
          today is on a wiki with the exact button presses within the hour,
          often before the game is even out.
        </p>

        <p>
          And still, through all of that, people kept hiding things.
        </p>

        <Figure
          src="/blog/easter-egg/google-askew.jpg"
          alt="Google search results for the word askew, with the entire page tilted a few degrees"
          caption="Search for the word askew. Google still tilts the page. It also, now, explains the word to you first."
          narrow
        />

        <Figure
          src="/blog/easter-egg/minecraft-removed-herobrine.jpg"
          alt="Official Minecraft 1.21.5 changelog with the line Removed Herobrine at the end of the list of changes"
          caption="Every Minecraft changelog for fifteen years has ended the same way. Herobrine was never in the game."
        />

        <Figure
          src="/blog/easter-egg/jeb-sheep.webp"
          alt="A Minecraft sheep with rainbow-coloured wool"
          caption="Name a sheep jeb_ and it will not stop changing colour."
          narrow
        />

        <Figure
          src="/blog/easter-egg/chrome-dino.png"
          alt="The pixel dinosaur from Chrome's offline page, mid-jump, with a game over message"
          caption="Chrome, when the internet is gone."
        />

        <p>
          I love all of these. I want that on the record before I say the next
          thing. They were planned, and specified, and someone in legal nodded
          at them, and still, behind each one, there was a person who wanted to
          be there. None of this killed it, though. It limped for a long time, a
          little more corporate every year, but it limped.
        </p>

        <h2>What finally took it</h2>

        <p>
          What I think actually killed it was the code starting to write
          itself. Ask why those engineers built a planet inside a spreadsheet. No user
          wanted it and no metric moved. They built it because they had been
          living inside that codebase for two years. They knew the charting engine the
          way you know a house you grew up in, including the parts that were
          ugly, because they had been awake at two in the morning fixing them.
          Somewhere in all of that the suffering became a kind of ownership,
          and ownership, without meaning to, turned into something like love,
          and love wanted to carve a name into the underside of the desk.
        </p>

        <p>
          Robinett&apos;s room was what months alone with four kilobytes does to a
          person.
        </p>

        <p>
          I have watched an app appear in the time it takes to make coffee,
          and I have made some of them myself. You describe what you want to a text box and
          you step away, and when you come back it is there: the routing, the
          login, a dark mode toggle, a settings page no one will ever visit. It
          works, mostly. It took eleven minutes, and in those eleven minutes there
          was no two in the morning and no ugly corner you came to know by
          heart, only a wall of plausible code that arrived all at once and had
          never met you.
        </p>

        <Figure
          src="/blog/easter-egg/text-box.jpg"
          narrow
          alt="A ChatGPT conversation. Someone asks for a JavaScript function to shuffle a deck of cards and receives the full code with an explanation."
          caption="Somebody else&apos;s text box. This is the illustration on the Wikipedia article for vibe coding."
        />

        <p>
          This is the part I find hard to say plainly, so I will say it the
          long way. It baffles me a little, but you don&apos;t want to leave a
          piece of yourself in that app, and I think it is because no piece of
          yourself went into it in the first place. The egg needed a person who had earned the right to be
          possessive about a codebase, and
          eleven minutes is not long enough to earn that.
        </p>

        <p>
          You could ask the text box for one, of course, and it would give you
          one, and it would be a Konami code that makes confetti, and it would carry
          exactly as much of you as a birthday greeting from your bank.
        </p>

        <Figure
          src="/blog/easter-egg/konami-code.png"
          alt="The Konami code as ten buttons: up, up, down, down, left, right, left, right, B, A"
          caption="The Konami code. Type it on the right website and confetti falls, or the page flips, or a dinosaur appears. It has been the default hidden thing since the eighties."
        />

        <Figure
          src="/blog/easter-egg/npm-konami.jpg"
          alt="npm search results for konami code, showing over a thousand packages that listen for the sequence"
          caption="Over a thousand packages on npm that listen for it. You do not write the easter egg. You install it."
          narrow
        />

        <h2>What I am not saying</h2>

        <p>
          It may be tempting to read all this as a plea to go back, and it
          isn&apos;t one. The old way of building software was slow and it hurt,
          and a great deal of the hurting was, if we are being honest,
          pointless. The tools
          are better and the things get built. I use the text box too, and I
          will keep using it, and I do not think there is anything to apologise
          for in that.
        </p>

        <p>
          But the unauthorised easter egg was proof that a human being had been
          inside the machine for a long and unpleasant stay. The lawyers made that risky, code review made it visible, the
          dashboards left it unfunded, and the internet made it brief. Then the
          generated code took away the one thing it could not survive without,
          which was someone who had been there long enough to care.
        </p>

        <p>
          The software still works, but it never met anyone.
        </p>

        <hr />

        <p>
          I write this more solemnly than the subject deserves, and I know it. It is a
          spreadsheet with a flight simulator in it, and a sheep that changes
          colour. But I have noticed that the things I make quickly, I do not
          remember, and the things I made slowly and badly, at hours I should have been
          asleep, I can still walk through with my eyes closed.
        </p>

        <p>
          If you are still building the slow way somewhere, on something no one
          asked you for, leave the room in, a line that only shows on one day of
          the year, or a command that is not in the help text and does
          something small and unnecessary. No one may ever find it, and that
          was never the point.
        </p>

        <p>
          Perhaps the point was only ever this: that somewhere in the binary
          there is a wall with a name on it, and the name is there because a
          person was.
        </p>
      </article>
    </PageShell>
  );
}
