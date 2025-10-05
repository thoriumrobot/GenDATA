// Source-based slice around line 29
// Method: com.google.thirdparty.publicsuffix.TrieParser.DIRECT_JOINER

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import java.util.ArrayDeque;
import java.util.Deque;

/** Parser for a map of reversed domain names stored as a serialized radix tree. */
@GwtCompatible
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
