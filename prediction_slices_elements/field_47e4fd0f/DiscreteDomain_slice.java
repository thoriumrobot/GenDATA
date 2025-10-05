// Source-based slice around line 247
// Method: com.google.common.collect.DiscreteDomain.supportsFastOffset


    @Override
    public String toString() {
      return "DiscreteDomain.bigIntegers()";
    }

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  final boolean supportsFastOffset;

  /** Constructor for use by subclasses. */
  protected DiscreteDomain() {
    this(false);
  }

  /** Private constructor for built-in DiscreteDomains supporting fast offset. */
  private DiscreteDomain(boolean supportsFastOffset) {
    this.supportsFastOffset = supportsFastOffset;
  }
