// Source-based slice around line 307
// Method: <com.google.common.net.InternetDomainName: boolean validatePart(String,boolean)>


  /**
   * Helper method for {@link #validateSyntax(List)}. Validates that one part of a domain name is
   * valid.
   *
   * @param part The domain name part to be validated
   * @param isFinalPart Is this the final (rightmost) domain part?
   * @return Whether the part is valid
   */
  private static boolean validatePart(String part, boolean isFinalPart) {

    // These tests could be collapsed into one big boolean expression, but
    // they have been left as independent tests for clarity.

    if (part.length() < 1 || part.length() > MAX_DOMAIN_PART_LENGTH) {
      return false;
    }

    /*
     * GWT claims to support java.lang.Character's char-classification methods, but it actually only
