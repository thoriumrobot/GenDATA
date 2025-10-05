// Source-based slice around line 65
// Method: <com.google.thirdparty.publicsuffix.TrieParser: int doParseTrieToBuilder(Deque,CharSequence,int,ImmutableMap)>

   *
   * @param stack The prefixes that precede the characters represented by this node. Each entry of
   *     the stack is in reverse order.
   * @param encoded The serialized trie.
   * @param start An index in the encoded serialized trie to begin reading characters from.
   * @param builder A map builder to which all entries will be added.
   * @return The number of characters consumed from {@code encoded}.
   */
  private static int doParseTrieToBuilder(
      Deque<CharSequence> stack,
      CharSequence encoded,
      int start,
      ImmutableMap.Builder<String, PublicSuffixType> builder) {

    int encodedLen = encoded.length();
    int idx = start;
    char c = '\0';

    // Read all the characters for this node.
    for (; idx < encodedLen; idx++) {
