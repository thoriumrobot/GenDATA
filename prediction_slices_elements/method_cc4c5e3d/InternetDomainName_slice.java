// Source-based slice around line 358
// Method: <com.google.common.net.InternetDomainName: ImmutableList parts()>


    return true;
  }

  /**
   * Returns the individual components of this domain name, normalized to all lower case. For
   * example, for the domain name {@code mail.google.com}, this method returns the list {@code
   * ["mail", "google", "com"]}.
   */
  public ImmutableList<String> parts() {
    return parts;
  }

  /**
   * Indicates whether this domain name represents a <i>public suffix</i>, as defined by the Mozilla
   * Foundation's <a href="http://publicsuffix.org/">Public Suffix List</a> (PSL). A public suffix
   * is one under which Internet users can directly register names, such as {@code com}, {@code
   * co.uk} or {@code pvt.k12.wy.us}. Examples of domain names that are <i>not</i> public suffixes
   * include {@code google.com}, {@code foo.co.uk}, and {@code myblog.blogspot.com}.
   *
