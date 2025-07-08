import { TokenizerConfig } from "./types";

export class Tokenizer {
  vocab: Record<string, number>;
  invVocab: Record<number, string>;
  unkToken: string | null;
  merges: Map<string, string>;

  constructor(config: TokenizerConfig) {
    this.vocab = config.model.vocab;
    this.invVocab = Object.fromEntries(
      Object.entries(this.vocab).map(([k, v]) => [v, k])
    );
    this.unkToken = config.model.unk_token;
    this.merges = new Map(
      config.model.merges.map(([a, b]) => [a + b, a + " " + b])
    );
  }

  static fromConfig(config: TokenizerConfig): Tokenizer {
    return new Tokenizer(config);
  }

  encode(text: string): number[] {
    const words = this.simpleSplit(text);
    const tokens: string[] = [];

    for (const word of words) {
      const pieces = this.bytePairEncode(word);
      tokens.push(...pieces);
    }

    const tokenIds = tokens.map((token) => {
      if (token in this.vocab) return this.vocab[token];
      if (this.unkToken && this.unkToken in this.vocab)
        return this.vocab[this.unkToken];
      throw new Error(`Token not in vocab and no unk_token defined: ${token}`);
    });

    return tokenIds;
  }

  decode(tokenIds: number[]): string {
    const tokens = tokenIds.map(
      (id) => this.invVocab[id] ?? this.unkToken ?? "<unk>"
    );
    return tokens.join("");
  }

  private simpleSplit(text: string): string[] {
    const pattern = /[\w]+|[^\s\w]/g;
    const matches = text.match(pattern);
    return matches ?? [];
  }

  private bytePairEncode(word: string): string[] {
    const tokens = Array.from(word);

    while (true) {
      let minPairIndex = -1;
      let minPair: string | null = null;

      // Find first merge pair in tokens matching merges
      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = tokens[i] + tokens[i + 1];
        if (this.merges.has(pair)) {
          minPair = pair;
          minPairIndex = i;
          break;
        }
      }

      if (minPair === null) break;

      // Merge pair tokens[minPairIndex] and tokens[minPairIndex+1]
      tokens.splice(minPairIndex, 2, minPair);
    }

    return tokens;
  }
}

// const path = "../../moondream/data/tokenizer.json";
// const file = Bun.file(path);
// const config = await file.json();

// const tokenizer = Tokenizer.fromConfig(config as TokenizerConfig);

// const encoded = tokenizer.encode("Hello, world!");
// console.log("Encoded:", encoded);

// const decoded = tokenizer.decode(encoded);
// console.log("Decoded:", decoded);
