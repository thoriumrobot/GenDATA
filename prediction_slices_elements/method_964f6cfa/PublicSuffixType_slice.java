// Source-based slice around line 48
// Method: <com.google.thirdparty.publicsuffix.PublicSuffixType: char getLeafNodeCode()>


  /** The character used for a leaf node in the trie encoding */
  private final char leafNodeCode;

  PublicSuffixType(char innerNodeCode, char leafNodeCode) {
    this.innerNodeCode = innerNodeCode;
    this.leafNodeCode = leafNodeCode;
  }

  char getLeafNodeCode() {
    return leafNodeCode;
  }

  char getInnerNodeCode() {
    return innerNodeCode;
  }

  /** Returns a PublicSuffixType of the right type according to the given code */
  static PublicSuffixType fromCode(char code) {
    for (PublicSuffixType value : values()) {
