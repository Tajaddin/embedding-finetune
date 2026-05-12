"""Synthetic + real-domain triplet generators.

A *triplet* is ``(anchor, positive, negative)`` where ``positive`` is the
correct retrieval target for ``anchor`` and ``negative`` is a random
unrelated text. Training maximizes anchor–positive similarity over
anchor–negative similarity (the classic triplet-loss objective).

This module ships one synthetic generator that produces clean topical
triplets across cooking / sports / technology — enough to demonstrate
measurable recall lift in <1 minute of CPU training. Bring your own
``Triplet`` list for real domains.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Triplet:
    anchor: str
    positive: str
    negative: str
    topic: str = ""


_TOPICS: dict[str, list[tuple[str, str]]] = {
    "cooking": [
        ("how do I cook risotto", "Risotto is a creamy Italian rice dish made by gradually adding warm broth."),
        ("what is sourdough", "Sourdough is fermented bread that uses wild yeast and lactic-acid bacteria."),
        ("ratio for vinaigrette", "A classic vinaigrette is three parts oil to one part vinegar with salt."),
        ("when to flip pancakes", "Flip pancakes when the surface bubbles burst and edges look set."),
        ("how to caramelize onions", "Caramelize onions over low heat for 30+ minutes until deeply golden."),
        ("temperature for medium steak", "A medium steak is cooked to 135-145 F internal temperature."),
        ("difference between baking soda and powder", "Baking powder includes acid; baking soda needs an acidic ingredient."),
        ("how to clarify butter", "Melt butter slowly and skim the white milk solids to make clarified butter."),
        ("knead bread dough how long", "Knead bread dough about 10 minutes until smooth and elastic."),
        ("blanching vegetables", "Blanching is brief boiling followed by an ice bath to set color."),
        ("perfect rice ratio", "Use a 1 to 2 ratio of long-grain white rice to water."),
        ("season cast iron", "Season cast iron by oiling and baking at 450 F for an hour."),
        ("difference between stock and broth", "Stock is made from bones; broth is made primarily from meat or vegetables."),
        ("how to ferment kimchi", "Salt napa cabbage, mix gochugaru paste, and ferment at room temp 1 to 5 days."),
        ("emulsify mayonnaise", "Whisk oil slowly into egg yolk with mustard and lemon to emulsify mayo."),
        ("scoring sourdough", "Score sourdough deeply with a lame to control oven spring."),
        ("rest steak before cutting", "Rest steak 5-10 minutes after cooking so juices redistribute."),
        ("vegetable stock vs water", "Vegetable stock adds savory depth absent from plain water in soups."),
        ("difference between baking and roasting", "Both use dry oven heat; roasting usually means higher temperature and meat or veg."),
        ("how to make pesto", "Pesto is basil, garlic, pine nuts, parmesan, and olive oil blended together."),
    ],
    "sports": [
        ("offside rule in soccer", "A player is offside when nearer to the opponent goal than the ball and second-last defender at the moment the ball is played."),
        ("how to throw a curveball", "Grip the baseball with two fingers across the seam and snap the wrist downward on release."),
        ("nba shot clock duration", "The NBA shot clock is 24 seconds; it resets to 14 on offensive rebound."),
        ("tennis tiebreak first to", "A standard tennis tiebreak is first to 7 points, win by 2."),
        ("marathon distance", "A marathon is 42.195 kilometers or 26.2 miles long."),
        ("hockey power play", "A power play happens when one team has more skaters due to a penalty."),
        ("nfl downs per possession", "An NFL offense gets four downs to advance ten yards for a new set of downs."),
        ("rugby try points", "A try in rugby union is worth five points; conversion adds two."),
        ("cricket lbw rule", "An LBW dismissal happens when the ball would have hit the stumps but the batter's leg blocked it."),
        ("golf birdie definition", "A birdie is finishing a hole in one stroke under par."),
        ("ping pong serve rules", "A legal table-tennis serve must toss the ball at least 16 cm and strike it without spin from the palm."),
        ("formula one points system", "F1 awards 25 points for a win, scaling down to 1 point for tenth place."),
        ("how to do a pull-up", "Hang from a bar with palms forward, then pull until your chin clears the bar."),
        ("baseball batting average meaning", "Batting average is hits divided by official at-bats, often shown as .250 etc."),
        ("tour de france yellow jersey", "The yellow jersey is worn by the overall time leader in the Tour de France."),
        ("nba three-pointer distance", "The NBA three-point line is 23.75 feet from the basket at the top, 22 at the corners."),
        ("volleyball libero role", "The libero is a back-row defensive specialist who cannot attack above the net."),
        ("basketball traveling rule", "A player commits a traveling violation by moving the pivot foot illegally without dribbling."),
        ("offensive rebound definition", "An offensive rebound is when the shooting team recovers the ball after a missed shot."),
        ("how long is a soccer match", "A standard soccer match is two 45-minute halves with stoppage time added."),
    ],
    "technology": [
        ("what is tcp/ip", "TCP/IP is the protocol suite for routing and delivering packets across the internet."),
        ("difference between sql and nosql", "SQL stores structured data in tables; NoSQL covers document, key-value, and column stores."),
        ("what does http stand for", "HTTP stands for Hypertext Transfer Protocol used for web browsing."),
        ("git rebase vs merge", "Rebase rewrites commit history onto a new base; merge preserves history with a merge commit."),
        ("how does dns work", "DNS resolves human-readable domain names to IP addresses via a hierarchy of name servers."),
        ("what is a vpn", "A VPN tunnels your network traffic through an encrypted connection to a remote server."),
        ("explain rest api", "A REST API exposes resources via HTTP verbs and stateless requests."),
        ("difference between docker and vm", "Docker uses OS-level containers; a VM emulates a full operating system stack."),
        ("what is kubernetes", "Kubernetes is a system for orchestrating containerized workloads across clusters."),
        ("how does ssl work", "SSL/TLS uses asymmetric handshake then symmetric session keys to encrypt traffic."),
        ("oauth flow basics", "OAuth lets a user grant a third-party app limited access without sharing the password."),
        ("what is a load balancer", "A load balancer distributes network traffic across multiple backend servers."),
        ("microservices vs monolith", "Microservices split a system into independently deployable services; monoliths run as one unit."),
        ("what is graphql", "GraphQL is a query language for APIs where clients ask for the exact data shape they need."),
        ("difference between cdn and dns", "A CDN delivers cached content from edge servers; DNS resolves domain names."),
        ("explain cache eviction lru", "Least-Recently-Used eviction drops the cache entry not accessed for the longest time."),
        ("what is a webhook", "A webhook is an HTTP callback that pushes events from one service to another."),
        ("how do databases use indexes", "Indexes precompute sorted views of column values to speed up lookup queries."),
        ("explain bloom filter", "A Bloom filter is a probabilistic set that can give false positives but never false negatives."),
        ("what is rate limiting", "Rate limiting caps requests per client over a time window to protect a service."),
    ],
}


def synthetic_triplets(*, n_train: int = 800, n_eval: int = 200, seed: int = 7) -> tuple[list[Triplet], list[Triplet]]:
    """Build train + eval triplets from the built-in topic corpus.

    Each triplet's anchor + positive come from the same topic; the negative
    is sampled from a different topic. Same-topic anchors share semantics
    but use different phrasing so the projection head must learn topical
    grouping, not surface tokens.
    """
    rng = random.Random(seed)
    all_topics = list(_TOPICS.keys())
    by_topic: dict[str, list[tuple[str, str]]] = {t: list(_TOPICS[t]) for t in all_topics}

    triplets: list[Triplet] = []
    for topic in all_topics:
        items = by_topic[topic]
        for i in range(len(items)):
            anchor, positive = items[i]
            negative_topic = rng.choice([t for t in all_topics if t != topic])
            negative = rng.choice(by_topic[negative_topic])[1]
            triplets.append(Triplet(anchor=anchor, positive=positive, negative=negative, topic=topic))

    # Also build cross-anchor triplets to enlarge the set: use one topic's answer
    # as another item's anchor by swapping. (Duplicates of the topic structure
    # but different phrasing pairs.)
    extra: list[Triplet] = []
    for topic in all_topics:
        items = by_topic[topic]
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                # Use item i's question as anchor, item j's answer as a *related* positive
                # (same topic). This is a noisier positive but it grows the set.
                anchor_q, _ = items[i]
                _, pos_a = items[j]
                neg_topic = rng.choice([t for t in all_topics if t != topic])
                neg_a = rng.choice(by_topic[neg_topic])[1]
                extra.append(Triplet(anchor=anchor_q, positive=pos_a, negative=neg_a, topic=topic))

    rng.shuffle(extra)
    full = triplets + extra
    rng.shuffle(full)

    if n_train + n_eval > len(full):
        # Cap both at half the corpus.
        n_train = len(full) // 2
        n_eval = len(full) - n_train
    return full[:n_train], full[n_train : n_train + n_eval]
