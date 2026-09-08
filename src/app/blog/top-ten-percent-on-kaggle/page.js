import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

const desc =
  "My first Kaggle competition, start to finish: what the score means, what cross-validation is for, what the data generator left behind, and why 351st of 3,532 was the honest result.";

export const metadata = {
  title: "What I learned placing top 10% in a Kaggle competition",
  description: desc,
  openGraph: {
    title: "What I learned placing top 10% in a Kaggle competition",
    description: desc,
    url: "https://stimmie.dev/blog/top-ten-percent-on-kaggle",
    type: "article",
    images: [{ url: "/blog/kaggle/cover.png", width: 1200, height: 630, alt: "Kaggle S6E8 submissions, cross-validation against public and private scores" }],
  },
  twitter: { card: "summary_large_image", images: ["/blog/kaggle/cover.png"] },
};

function Figure({ src, alt, caption }) {
  return (
    <figure className="my-6">
      <img src={src} alt={alt} className="w-full border-2 border-current" loading="lazy" />
      {caption && <figcaption className="mt-2 text-sm neo-muted font-mono">{caption}</figcaption>}
    </figure>
  );
}

const SUBS = [
  ["Aug 19", "LightGBM baseline, 5-fold", "0.96321", "0.96487", "0.96471"],
  ["Aug 19", "v3, nested target encoding, slack feature, 3-model stack", "0.96809", "0.96950", "0.96920"],
  ["Aug 19", "v5, 4-model logistic-rank stack", "0.96817", "0.96950", "0.96925"],
  ["Aug 20", "v7b, hill climb over 10 members", "0.96860", "0.96967", "0.96940"],
  ["Aug 20", "v9b, hill climb over 81 public and own OOF arrays", "0.96973", "0.97068", "0.97044"],
  ["Aug 20", "v7d, self-built final, 3 transformer seeds and 2 CatBoosts", "0.96947", "0.97048", "0.97021"],
  ["Aug 20", "hybrid2, 0.4 self-built, 0.6 public blend", "", "0.97102", "0.97075"],
  ["Aug 23", "v10, ElasticNet stack over 84 members", "0.96995", "0.97097", "0.97068"],
  ["Aug 23", "overnight, 15-seed bagged ElasticNet", "0.96998", "0.97101", "0.97071"],
  ["Aug 23", "the public blend, forked verbatim", "", "0.97117", "0.97090"],
];

export default function KagglePage() {
  return (
    <PageShell title="What I learned placing top 10% in a Kaggle competition" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">September 2026</p>

        <p>
          On the 19th of August I had five free hours and a message in the guild chat about an AI contest in
          December. I figured the quickest way to find out what I didn&apos;t know was to enter a Kaggle competition
          that same afternoon. I also wanted something to post on LinkedIn. I&apos;m not going to pretend otherwise.
        </p>

        <p>
          I had never entered one before. Twelve days later I finished 351st of 3,532 teams on one leaderboard and
          369th on the other, which is the top ten percent on one and a hair outside it on the other. This is the
          whole thing in order, written for the version of me who opened the page on day one and didn&apos;t know what
          any of the words meant.
        </p>

        <h2>What a Kaggle competition actually is</h2>

        <p>
          You get two files. <code>train.csv</code> has 691,369 rows, each one a person, with twelve columns about
          them (hours of screen time, hours on social media, hours gaming, hours of sleep, notifications per day,
          age, and so on) plus one final column called <code>addicted_label</code> that&apos;s 1 or 0. That last column
          is the answer.
        </p>

        <p>
          <code>test.csv</code> has 296,000 more people and no answer column. Your job is to guess. You upload a
          two-column file, one row per person, and it looks like this:
        </p>

        <pre className="text-sm"><code>{`id,addicted_label
691369,0.8421
691370,0.1337
691371,0.9002`}</code></pre>

        <p>
          That second number is a confidence between 0 and 1 rather than a yes or no, which I hadn&apos;t expected.
          Kaggle scores the file, puts you on a leaderboard, and lets you upload five times a day. The competition
          I entered was{" "}
          <a href="https://www.kaggle.com/competitions/playground-series-s6e8">Playground Series Season 6, Episode 8</a>,
          which ran from the start of August to the 31st. Playground competitions are the practice tier: no prize
          money, thousands of entrants, and the data is generated rather than collected, which matters later.
        </p>

        <h2>The score, and why it isn&apos;t accuracy</h2>

        <p>
          The metric here is ROC AUC. It took me a few tries to get it, and this is the phrasing that stuck.
        </p>

        <p>
          Take one person who really is addicted and one who really isn&apos;t, both at random. Look at the two numbers
          your model gave them. Did it give the addicted one the higher number? AUC is simply how often that&apos;s
          true. Guess randomly and you&apos;re right half the time, so 0.5. Get it right every time and you score 1.0.
        </p>

        <p>
          The important part is that AUC only cares about the <em>order</em>, not the actual values. A model that
          outputs 0.9 and 0.8 scores the same as one that outputs 0.02 and 0.01, as long as the right person is on
          top. This is why you submit probabilities and never round them to 0 and 1: rounding throws away the
          ordering inside each group and your score collapses. I know because the first thing I wanted to do was
          round them.
        </p>

        <Figure
          src="/blog/kaggle/what-auc-measures.png"
          alt="Two overlapping histograms of my model's predicted probabilities across all 691,369 training people, one for those actually addicted and one for those not, showing heavy separation but real overlap."
          caption="My model&apos;s guess for each of the 691,369 training people, split by what they actually were. Blue piles up near zero, pink near one, and the purple is where they overlap. AUC is the chance a random pink sits to the right of a random blue, which here came to 0.96995. Counts are on a log scale, since the bar at 1.0 is otherwise forty times taller than anything else."
        />

        <h2>The first hour</h2>

        <p>
          The baseline was <a href="https://lightgbm.readthedocs.io/">LightGBM</a>, a gradient boosting library,
          run five times over the data with default settings. It took about ten minutes to write and run. It scored
          0.96487 on the leaderboard. The person in first place had 0.97134.
        </p>

        <p>
          My first question was how anyone finds the extra 0.0065, and my second was whether 0.0065 is even a lot.
          On this problem it&apos;s, because the entire field lives inside about half a percent. A ten-minute script
          put me within 0.65 percent of first place, and the remaining twelve days went on the last 0.006. I had
          assumed the gap between a beginner and the top would be wide, and most of it closed in ten minutes.
        </p>

        <p>
          The second thing I tried was adding a second library, <a href="https://xgboost.readthedocs.io/">XGBoost</a>,
          and averaging the two sets of guesses. The score moved by 0.0004. I asked what that had actually done and why
          it worked, which turned out to be the question I kept asking for the rest of the week.
        </p>

        <h2>Cross-validation, and why the leaderboard lies</h2>

        <p>
          I didn&apos;t know what a fold was. A fold is one of the equal chunks the training data gets cut into. With
          five folds you train on four of them, predict the fifth, rotate until every chunk has been predicted
          once by a model that never saw it, then average the five scores. That average is called your CV, and it
          is your own private leaderboard.
        </p>

        <p>
          It matters more than the public one. The public leaderboard is scored on a slice of the test set, and
          every time you look at it and change something in response, you&apos;re quietly fitting your model to that
          slice. Do it fifty times and your leaderboard score is measuring how well you memorised the leaderboard.
        </p>

        <p>
          The one genuinely useful thing to do with the public score is to check it once against your CV at the
          start. Mine agreed to within 0.002, and the leaderboard was slightly higher than CV, which is the healthy
          direction. That told me my validation was honest. From then on I only submitted when CV went up, and I
          put the CV number in every submission description so I could tell later what had actually helped.
        </p>

        <h2>Leakage</h2>

        <p>
          Leakage is when your model sees something during training that it won&apos;t have at prediction time. The
          textbook example: predict whether a patient has diabetes, and include &quot;is taking insulin&quot; as a
          column. The model scores 0.99 and is useless, because in real life the insulin comes after the diagnosis
          you&apos;re trying to predict.
        </p>

        <p>
          The cheap check is to score every column on its own against the answer. If one column nearly solves the
          problem by itself, something is wrong. Here the strongest were daily screen time and weekend screen time,
          and neither came close to solving it alone, which is what a strong honest feature looks like.
        </p>

        <p>
          The subtler version bit other people. Target encoding is a common trick where you replace a category with
          the average answer for that category. Do that <em>before</em> you split into folds and each fold's
          training data now contains a summary of its own validation answers. Several popular public notebooks did
          exactly this. Their validation scores were beautiful and their leaderboard scores weren&apos;t. Until then I
          had thought of leakage as a mistake careless people make. It&apos;s also a mistake careful people make when
          the loop is one line too short.
        </p>

        <h2>What the data generator left behind</h2>

        <p>
          Playground data is synthetic. Kaggle takes a real dataset, trains a model to imitate it, and generates
          hundreds of thousands of fake rows. This one came from a{" "}
          <a href="https://www.kaggle.com/datasets/algozee/smartphone-addiction-prediction-data">7,500-row survey</a>{" "}
          that the competition page credits. The imitation is never perfect, and the imperfections are where most
          of the gain over the baseline came from.
        </p>

        <p>
          The first one you can see by counting. Screen time is recorded to two decimal places, so across 691,369
          people you would expect almost no exact repeats. Instead there are only 1,389 distinct values in the
          whole column, and the most common one shows up 3,434 times.
        </p>

        <Figure
          src="/blog/kaggle/repeated-values.png"
          alt="Bar chart of the 18 most common exact values of daily screen time hours, each appearing between roughly 1,700 and 3,434 times across 595,515 rows."
          caption="The eighteen most common values of daily screen time. There are only 1,389 distinct values across 595,515 rows, and the most common one repeats 3,434 times."
        />

        <p>
          Once you see that, the column stops being a measurement and starts being a set of labels that happen to
          look like numbers. Converting them to text and treating them as categories moved my score more than any
          hyperparameter did.
        </p>

        <p>
          The second one is a rule. In the generated data, daily screen time is never less than social media plus
          gaming plus work hours. Not rarely: never, in all 595,515 rows that have all four values. The real survey
          breaks that rule in more than half its rows, so the generator was enforcing something the humans didn&apos;t.
          The size of the gap, which I started calling slack, predicts the answer at 0.765 AUC entirely on its own.
        </p>

        <Figure
          src="/blog/kaggle/slack-constraint.png"
          alt="Histogram of daily screen time minus the sum of social media, gaming and work hours, with a dashed line at zero and no mass at all to the left of it."
          caption="Daily screen time minus social media, gaming and work hours, for every row that carries all four. None of the 595,515 rows falls below zero. The real survey breaks the same rule in more than half of its own rows."
        />

        <p>
          I also tried the obvious cheat. If the fake data came from a real survey, why not just find the survey
          and use the real answers? I found it, confirmed it was the source, and added it to the training data. The
          score got worse: 0.96632 to 0.96628 with one copy, and 0.96596 with ten. The generator had re-invented
          the answers from scratch, so the real ones had nothing to say about the fake people.
        </p>

        <h2>Eighty-four models</h2>

        <p>
          The last stretch was volume. By the end there were 84 different models: LightGBM, XGBoost and CatBoost
          across different random seeds and fold counts, plus a small neural network that treats each column as a
          word in a sentence, which I took from a public notebook after my own attempt at one came out worse.
        </p>

        <p>
          You then have to combine 84 sets of guesses into one. I tried three ways of doing it and they all landed
          within 0.0001 of each other, which told me the combining method wasn&apos;t where the score lived. What did
          move it was adding a model that made <em>different</em> mistakes from the others. Another copy of the same
          model with a new random seed was worth about 0.00003, which is nothing.
        </p>

        <h2>Forking the public notebook</h2>

        <p>
          Partway through I noticed a lot of identical scores on the leaderboard. That happens when one public
          notebook, itself a blend of other public notebooks, gets forked by everyone who opens it. Kaggle breaks
          exact ties by who submitted first, so a chunk of the board was less a ranking than a queue.
        </p>

        <p>
          On August 23 I joined it, submitting the blend verbatim with the description &quot;public blend, forked,
          max public&quot;. It scored 0.97117 and put me at rank 98 of 2,687. I took a screenshot. I want to be
          clear that the number wasn&apos;t mine.
        </p>

        <p>
          By the deadline the same file was rank 351 of 3,532. Nothing about it had changed. The notebook kept
          being improved and re-forked, and the newer forks piled into clusters above mine, 72 teams tied at
          0.97128 and 71 at 0.97130, with 344 teams above me in total. The queue had moved and I was standing where
          it used to be.
        </p>

        <Figure
          src="/blog/kaggle/leaderboard-clusters.png"
          alt="Bar chart of how many teams share each exact public leaderboard score near the top, with tall bars of 72 and 71 teams at 0.97128 and 0.97130, and my own cluster of 24 teams at 0.97117 highlighted."
          caption="How many teams share each exact score at the top of the final public leaderboard. Identical scores mean identical files. Mine is the pink bar, tied with 23 others, and the bars of 72 and 71 teams to its right are later forks of the same notebook."
        />

        <p>
          The fork handed me a score that other people were already busy improving on. I don&apos;t think I&apos;d have
          understood that from reading about it.
        </p>

        <h2>The two finals</h2>

        <p>
          Here is the part that makes Kaggle interesting. The leaderboard you watch all competition is computed on
          a slice of the test set. The real one, scored on everything else, is hidden until the deadline. You pick
          two submissions to be judged on it.
        </p>

        <p>
          One of my slots went to the fork. The other went to the best thing I had built myself, a fifteen-seed
          bagged ElasticNet stack, CV 0.96998, public 0.97101. My theory was that the fork was fitted to the public
          slice and would fall, the honest entry would hold, and I&apos;d come out ahead of the queue.
        </p>

        <p>
          On September 1 the fork went from 0.97117 to 0.97090, and the honest entry from 0.97101 to 0.97071. Both
          dropped by about the same amount, the fork still won, and I landed at 369th. The theory was wrong, or the
          effect was too small to see. The winner,{" "}
          <a href="https://www.kaggle.com/cdeotte">Chris Deotte</a>, scored 0.97207 public and 0.97176 private,
          clear of every cluster on both boards.
        </p>

        <p>
          His{" "}
          <a href="https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe">writeup</a>{" "}
          is not what I expected. He ran a swarm of language model agents, set two of them competing to build the
          best single model, and one of those models won the competition on its own, without an ensemble, which
          he says had not happened in a Playground competition in eighteen months. The comments underneath are
          worth reading too, because a lot of people finishing a few hundred places above and below me were asking
          the same question about what is left for a person to do.
        </p>

        <p>
          I have my own answer, which is that the thing I could not have outsourced was knowing whether to believe
          my own validation. Deotte says something close to this in the comments: humans cannot beat agents on
          coding speed any more, only on the insight the agent overlooked. Learning what a fold is for turns out
          to be the part that keeps mattering.
        </p>

        <p>
          Here is every submission I made, all of them public on{" "}
          <a href="https://www.kaggle.com/stimmie">my Kaggle profile</a>, with the number I trusted, the number
          everyone could see, and the number that counted.
        </p>

        <table className="my-6 w-full text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-current text-left">
              <th className="py-1 pr-3">Date</th><th className="py-1 pr-3">Submission</th><th className="py-1 pr-3 text-right">CV</th><th className="py-1 pr-3 text-right">Public</th><th className="py-1 text-right">Private</th>
            </tr>
          </thead>
          <tbody>
            {SUBS.map(([d, s, cv, pub, priv]) => (
              <tr key={s} className="border-t border-current">
                <td className="py-1 pr-3 whitespace-nowrap">{d}</td><td className="py-1 pr-3">{s}</td><td className="py-1 pr-3 text-right">{cv}</td><td className="py-1 pr-3 text-right">{pub}</td><td className="py-1 text-right">{priv}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-sm neo-muted font-mono">Ten of my fifteen submissions. The other five were the same ideas with different random seeds.</p>

        <h2>The ceiling</h2>

        <p>
          Somewhere around day three I asked why I couldn&apos;t simply keep going until the score hit 1.0. The answer
          has a name.
        </p>

        <p>
          For any set of features there&apos;s a true probability that a person with those exact numbers is addicted.
          Call it p. If p is only ever 0 or 1, meaning the numbers fully determine the answer, then a perfect model
          scores 1.0. But if two people can share every number and get different answers, no model can rank one
          above the other, and every pair like that costs you score. The loss you can&apos;t avoid is called the
          <em> Bayes error</em>, and the best score anyone could possibly reach is the{" "}
          <a href="https://en.wikipedia.org/wiki/Bayes_error_rate">Bayes-optimal</a> one.
        </p>

        <p>
          You can see it in this data. Round four of the columns and group the people who match. Most of them end
          up in a group where the answer isn&apos;t unanimous.
        </p>

        <Figure
          src="/blog/kaggle/bayes-ceiling.png"
          alt="A proportion bar showing 83 percent of people in look-alike groups sit in a group containing both answers, with the example of 3,339 people sharing four rounded values of whom 17 percent were labelled addicted."
          caption="The 685,219 people who share four rounded numbers with at least nineteen others. 83 percent of them sit in a group where the answer isn&apos;t unanimous. One such group holds 3,339 people who all reported 4h screen time, 1h social media, 7h sleep and 1h gaming, and 17 percent of them were labelled addicted."
        />

        <p>
          Which reframes the whole competition. Everyone above 0.97 was already pressed against that ceiling, and
          the fight was over the last thousandth. Kaggle gives you thirty free GPU hours a week, and I spent an
          evening getting my own laptop&apos;s graphics card working after finding it walled off by a virtual
          machine config I had set up a year earlier and forgotten. None of it changed the number, because the
          ceiling is a property of the data and not of how fast you can train.
        </p>

        <h2>Whether to do it again</h2>

        <p>
          Halfway through I asked whether someone who doesn&apos;t especially want an AI engineering job should grind
          this. My answer now is that one competition teaches you what cross-validation is for, what leakage looks
          like when it&apos;s subtle, how to read someone else&apos;s notebook and check its claims against the raw
          data, and how little the last three decimals are worth. The second competition teaches you the same
          things again.
        </p>

        <p>
          So I&apos;m not entering Episode 9. The December contest is a different shape, seven hours, a team of three,
          no leaderboard to probe, and what carries over is the discipline rather than the models. One fixed fold
          split, and a submission only when the honest number moves.
        </p>

        <p>
          What I posted on LinkedIn was the rank. What I saved was the list of questions I asked between the 19th
          and the 31st, starting with &quot;why AUC&quot; and ending with &quot;is there a theoretical maximum a
          perfect model can&apos;t reach&quot;. If I had to keep one of the two, it would be the list.
        </p>

        <hr />

        <p className="text-sm">
          Sources and further reading. The competition:{" "}
          <a href="https://www.kaggle.com/competitions/playground-series-s6e8">Predicting Smartphone Addiction</a>{" "}
          and its{" "}
          <a href="https://www.kaggle.com/competitions/playground-series-s6e8/leaderboard">final leaderboard</a>,
          which is where the rank and cluster counts come from, plus{" "}
          <a href="https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe">the winning writeup</a>{" "}
          and <a href="https://www.kaggle.com/stimmie">my own profile</a>. The source survey:{" "}
          <a href="https://www.kaggle.com/datasets/algozee/smartphone-addiction-prediction-data">Smartphone Addiction Prediction Data</a>.
          The libraries: <a href="https://lightgbm.readthedocs.io/">LightGBM</a>,{" "}
          <a href="https://xgboost.readthedocs.io/">XGBoost</a>, and{" "}
          <a href="https://catboost.ai/">CatBoost</a>. On the metric,{" "}
          <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics">scikit-learn on ROC AUC</a>{" "}
          and{" "}
          <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic">the Wikipedia article</a>. On
          the ceiling, <a href="https://en.wikipedia.org/wiki/Bayes_error_rate">Bayes error rate</a>. The charts of
          repeated values, the slack rule and the ceiling were computed from the competition&apos;s own{" "}
          <code>train.csv</code>; the leaderboard chart from Kaggle&apos;s public leaderboard export.
        </p>
      </article>
    </PageShell>
  );
}
