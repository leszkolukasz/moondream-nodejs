export type Dictionary<T> = Record<string, T>;

export type ModelConfig = {
  model_version: number;
  templates: {
    caption: {
      short: number[];
      normal: number[];
    };
    query: {
      prefix: number[];
      suffix: number[];
    };
    detect: {
      prefix: number[];
      suffix: number[];
    };
  };
  special_tokens: {
    bos: number;
    eos: number;
    coord: number;
    size: number;
  };
};

export type TokenizerConfig = {
  version: string;
  truncation: null;
  padding: null;
  added_tokens: [];
  normalizer: null;
  pre_tokenizer: {
    type: "ByteLevel";
    add_prefix_space: boolean;
    trim_offsets: boolean;
    use_regex: boolean;
  };
  post_processor: {
    type: "ByteLevel";
    add_prefix_space: boolean;
    trim_offsets: boolean;
    use_regex: boolean;
  };
  decoder: {
    type: "ByteLevel";
    add_prefix_space: boolean;
    trim_offsets: boolean;
    use_regex: boolean;
  };
  model: {
    type: "BPE";
    dropout: null;
    unk_token: null;
    continuing_subword_prefix: string;
    end_of_word_suffix: string;
    fuse_unk: boolean;
    byte_fallback: boolean;
    ignore_merges: boolean;
    vocab: Record<string, number>;
    merges: [string, string][];
  };
};
