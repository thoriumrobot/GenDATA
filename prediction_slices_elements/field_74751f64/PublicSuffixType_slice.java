// Source-based slice around line 41
// Method: com.google.thirdparty.publicsuffix.PublicSuffixType.leafNodeCode

  /** Public suffix that is provided by a private company, e.g. "blogspot.com" */
  PRIVATE(':', ','),
  /** Public suffix that is backed by an ICANN-style domain name registry */
  REGISTRY('!', '?');

  /** The character used for an inner node in the trie encoding */
  private final char innerNodeCode;

  /** The character used for a leaf node in the trie encoding */
  private final char leafNodeCode;

  PublicSuffixType(char innerNodeCode, char leafNodeCode) {
    this.innerNodeCode = innerNodeCode;
    this.leafNodeCode = leafNodeCode;
  }

  char getLeafNodeCode() {
    return leafNodeCode;
  }

