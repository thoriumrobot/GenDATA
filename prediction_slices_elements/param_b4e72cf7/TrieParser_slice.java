// Source-based slice around line 36
// Method: <com.google.thirdparty.publicsuffix.TrieParser: ImmutableMap parseTrie(CharSequence)>

final class TrieParser {

  private static final Joiner DIRECT_JOINER = Joiner.on("");

  /**
   * Parses a serialized trie representation of a map of reversed public suffixes into an immutable
   * map of public suffixes. The encoded trie string may be broken into multiple chunks to avoid the
   * 64k limit on string literal size. In-memory strings can be much larger (2G).
   */
  static ImmutableMap<String, PublicSuffixType> parseTrie(CharSequence... encodedChunks) {
    String encoded = DIRECT_JOINER.join(encodedChunks);
    return parseFullString(encoded);
  }

  @VisibleForTesting
  static ImmutableMap<String, PublicSuffixType> parseFullString(String encoded) {
    ImmutableMap.Builder<String, PublicSuffixType> builder = ImmutableMap.builder();
    int encodedLen = encoded.length();
    int idx = 0;

