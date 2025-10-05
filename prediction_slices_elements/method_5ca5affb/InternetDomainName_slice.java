// Source-based slice around line 657
// Method: <com.google.common.net.InternetDomainName: boolean matchesType(Optional,Optional)>

    } catch (IllegalArgumentException e) {
      return false;
    }
  }

  /**
   * If a {@code desiredType} is specified, returns true only if the {@code actualType} is
   * identical. Otherwise, returns true as long as {@code actualType} is present.
   */
  private static boolean matchesType(
      Optional<PublicSuffixType> desiredType, Optional<PublicSuffixType> actualType) {
    return desiredType.isPresent() ? desiredType.equals(actualType) : actualType.isPresent();
  }

  /** Returns the domain name, normalized to all lower case. */
  @Override
  public String toString() {
    return name;
  }

