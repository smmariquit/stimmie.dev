import Image from "next/image";

export default function ScrapbookPhoto({
  src,
  alt,
  caption = "that's me!",
  width = 400,
  height = 400,
}) {
  return (
    <figure className="neo-scrapbook-photo">
      <span className="neo-scrapbook-tape neo-scrapbook-tape-left" aria-hidden="true" />
      <span className="neo-scrapbook-tape neo-scrapbook-tape-right" aria-hidden="true" />
      <div className="neo-scrapbook-frame">
        <Image
          src={src}
          alt={alt}
          width={width}
          height={height}
          className="neo-scrapbook-img"
          sizes="(max-width: 640px) 45vw, 200px"
          priority
          fetchPriority="high"
        />
      </div>
      {caption ? (
        <figcaption className="neo-scrapbook-caption">{caption}</figcaption>
      ) : null}
    </figure>
  );
}
