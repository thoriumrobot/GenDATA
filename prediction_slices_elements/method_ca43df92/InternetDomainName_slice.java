// Source-based slice around line 617
// Method: <com.google.common.net.InternetDomainName: InternetDomainName child(String)>

  /**
   * Creates and returns a new {@code InternetDomainName} by prepending the argument and a dot to
   * the current name. For example, {@code InternetDomainName.from("foo.com").child("www.bar")}
   * returns a new {@code InternetDomainName} with the value {@code www.bar.foo.com}. Only lenient
   * validation is performed, as described {@link #from(String) here}.
   *
   * @throws NullPointerException if leftParts is null
   * @throws IllegalArgumentException if the resulting name is not valid
   */
  public InternetDomainName child(String leftParts) {
    return from(checkNotNull(leftParts) + "." + name);
  }

  /**
   * Indicates whether the argument is a syntactically valid domain name using lenient validation.
   * Specifically, validation against <a href="http://www.ietf.org/rfc/rfc3490.txt">RFC 3490</a>
   * ("Internationalizing Domain Names in Applications") is skipped.
   *
   * <p>The following two code snippets are equivalent:
   *
