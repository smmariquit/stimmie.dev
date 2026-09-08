import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

const desc =
  "Twelve days on Kaggle Playground S6E8, from a 0.96487 baseline to 351st of 3,532. What cross-validation, leakage, synthetic-data artifacts, and forking a public notebook turned out to mean.";

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
        <p className="text-base neo-muted font-mono">September 10, 2026</p>

        <p>
          On the 19th of August I had five free hours and a message in the guild chat about an AI contest in
          December. I figured the quickest way to find out what I did not know was to enter a Kaggle competition
          that same afternoon. I also wanted something to post on LinkedIn. I am not going to pretend otherwise.
        </p>

        <p>
          The one running was Playground Series Season 6, Episode 8,{" "}
          <a href="https://www.kaggle.com/competitions/playground-series-s6e8">Predicting Smartphone Addiction</a>.
          Synthetic tabular data, 691 thousand training rows and 296 thousand test rows, twelve columns like daily
          screen time, hours on social media, hours gaming, hours of sleep, and a yes or no label for whether the
          person is addicted. The score is ROC AUC. Deadline August 31. I did the whole thing with Claude Code
          writing the code and me asking the questions, about forty of them over twelve days, and this post is
          mostly the answers.
        </p>

        <p>
          I finished 351st of 3,532 teams on the public leaderboard and 369th on the private one. That is the top
          ten percent on one board and a hair outside it on the other, which is the honest version of the title.
          Every submission I made is below, with the number I trusted (cross-validation), the number everyone sees
          (public), and the number that counts (private).
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
        <p className="text-sm neo-muted font-mono">Ten of my fifteen submissions. The other five were repeats of the same ideas with different seeds.</p>

        <h2>The first afternoon</h2>

        <p>
          The baseline was LightGBM with five-fold cross-validation, run from a template. It took about ten minutes.
          Cross-validation scored it 0.96321, the leaderboard scored it 0.96487, and the top of the leaderboard sat
          at 0.97134. So my first question was how anyone gets the extra 0.0065, and my second was whether 0.0065
          is a lot. On this problem it is, because the whole field lives inside about half a percent. A ten-minute
          script was already within 0.65 percent of first place, and the remaining twelve days went on the last
          0.006. Adding XGBoost to the same folds and averaging the two moved the score by 0.0004, and I asked what
          we had just done, which turned out to be the question I kept asking all week. I also realised, looking
          at the top hundred, that whoever was first had probably done everything I was about to do in the first
          hour, with a script they wrote for the previous episode.
        </p>

        <p>
          I did not know what a fold was. A fold is one of the equal chunks the training data gets cut into.
          Cross-validation trains on four chunks, scores on the fifth, rotates, and averages the five scores. That
          number is your own private leaderboard, and it matters more than the public one, because the public one
          is scored on a slice of the test set you never see, and every time you look at it and change something
          you are quietly fitting to it. The one useful thing to do with the public score is to check it once
          against your cross-validation. Mine agreed to within 0.002 on the first submission, so from then on I only
          submitted when the cross-validation number went up.
        </p>

        <p>
          Leakage was the next word. It means the model sees something during training that it will not have when
          it actually predicts. The cheap check is to score every column alone against the label. If one column
          nearly solves the problem by itself, something is wrong with the data. Daily screen time and weekend
          screen time were the strongest here, and neither came anywhere near solving it alone, which is what a
          strong honest feature looks like. It matched what anyone would guess about phone addiction, which
          is a small relief when the rest of the competition is about things nobody would guess.
        </p>

        <h2>Where the rest of the score came from</h2>

        <p>
          Playground competitions use synthetic data, generated from a real dataset by a model that learned its
          shape. This one came from a 7,500-row survey we found on Kaggle later. The generator leaves fingerprints,
          and the fingerprints are where most of the gain over the baseline lived.
        </p>

        <p>
          The clearest one was a rule the generator enforced on every row: daily screen time is never less than
          social media hours plus gaming hours plus work hours. The real survey breaks that rule in more than half
          its rows. The generated data never does. The gap between the two sides, which we called slack, predicted
          the label at 0.765 AUC by itself. Another showed up as a grid pattern in the histogram of sleep hours: the generator reused exact values,
          so a number like 6.5 was closer to a category than a measurement. Turning the numbers into strings and
          treating them as categorical, with target encoding done inside the
          cross-validation loop so it could not leak, moved the score more than any hyperparameter did. The
          notebooks that did target encoding outside the loop had beautiful validation numbers and fell over on the
          leaderboard. Until then I had thought of leakage as a mistake careless people make. It is also a
          mistake careful people make when the loop is one line too short.
        </p>

        <p>
          We also tried the obvious cheat. We found the original 7,500 rows, confirmed they were the source down to
          the value grids, and appended them to the training data. It made the score worse, 0.96632 to 0.96628 with
          one copy and 0.96596 with ten. The generator had re-synthesized the labels from the features, so the
          real survey&apos;s answers had nothing to say about the generated rows.
        </p>

        <p>
          After that it was models. By the end there were 84 of them: LightGBM, XGBoost, and CatBoost across seeds
          and fold counts, plus a small transformer that treats each column as a token, ported from a public
          notebook after my own attempt at an embedding network came out worse. Stacking them with logistic
          regression, rank averaging them, and hill climbing over them all gave the same number to within 0.0001,
          which told me the combiner was not where the score lived. Adding a model that made different mistakes
          from the others was the only thing that reliably moved it.
        </p>

        <h2>The wall</h2>

        <p>
          Partway through I noticed a lot of identical scores on the public leaderboard. That happens when one
          public notebook, itself a blend of other public notebooks, gets forked by everyone who opens it. Kaggle
          breaks exact ties by who submitted first, so a chunk of the board was less a ranking than a queue. On
          August 23 I joined it, submitting the blend verbatim with the description &quot;public blend, forked,
          max public&quot;. It scored 0.97117 and put me at rank 98 of 2,687. I took a screenshot. I want to be
          clear that the number was not mine.
        </p>

        <p>
          By the deadline that same submission was rank 351 of 3,532. Nothing about it had changed. The public
          notebook kept being improved and re-forked, and the newer forks piled into clusters above mine, 72 teams
          tied at 0.97128 and 71 at 0.97130, while 344 teams in total finished above me. The queue had moved and I
          was standing where it used to be. Borrowing a number gets you the number and not the position, and I do
          not think I would have understood that from reading about it.
        </p>

        <p>
          Kaggle lets you pick two submissions to be scored privately. One slot went to the fork. The other went
          to the best thing we built ourselves, a fifteen-seed bagged ElasticNet stack, cross-validation 0.96998
          and public 0.97101. My theory was that the fork was fitted to the public slice and would fall on private,
          the honest entry would hold, and I would come out ahead of the queue. On September 1 the fork went from
          0.97117 to 0.97090 and the honest one from 0.97101 to 0.97071. Both dropped by about the same amount,
          the fork still won, and I landed at 369th. The theory was wrong, or the effect was too small to see. The
          winner, Chris Deotte, scored 0.97207 public and 0.97176 private, clear of every cluster on both boards.
          On this problem, that is what a grandmaster is worth.
        </p>

        <h2>Why nobody scores 1.0</h2>

        <p>
          At one point I asked why we could not just keep going. The answer is that the data contains rows with
          identical features and different labels. When two people with the same screen time, the same sleep, and
          the same everything else get different answers, no model can separate them, and the best possible AUC
          drops below one. The competition was pressed against that ceiling from about day two. Kaggle gives
          everyone thirty free GPU hours a week, and I spent an evening getting my own laptop&apos;s GPU working
          (it had been walled off by a virtual machine passthrough config I set up a year ago and forgot), and none
          of that changed the number, because the ceiling was set by the data and not by how fast anyone could
          train.
        </p>

        <h2>What I am keeping</h2>

        <p>
          Halfway through I asked whether someone who does not want an AI engineering job should grind this. I
          think one competition teaches you what cross-validation is for, what leakage looks like when it is
          subtle, how to read other people&apos;s notebooks and check their claims against the raw data, and how
          little the last three decimals are worth. The second competition teaches the same things again. So I
          am not doing Episode 9. The December contest is a different format, seven hours, a team of three, no
          leaderboard to probe, and what transfers is the discipline rather than the models. One fixed fold
          split, and a submission only when the honest number moves.
        </p>

        <p>
          What I posted on LinkedIn was the rank. What I saved was the list of questions I asked between the
          19th and the 31st, starting with &quot;why AUC&quot; and ending with &quot;is there a theoretical maximum
          a perfect model cannot reach&quot;. If I had to choose one of the two to keep, it would be the list.
        </p>

        <hr />

        <p className="text-sm">
          Sources: my{" "}
          <a href="https://www.kaggle.com/stimmie">Kaggle submissions page</a> for the scores in the table; the{" "}
          <a href="https://www.kaggle.com/competitions/playground-series-s6e8/leaderboard">final leaderboard</a>;
          the source dataset,{" "}
          <a href="https://www.kaggle.com/datasets/algozee/smartphone-addiction-prediction-data">Smartphone Addiction Prediction Data</a>.
        </p>
      </article>
    </PageShell>
  );
}
