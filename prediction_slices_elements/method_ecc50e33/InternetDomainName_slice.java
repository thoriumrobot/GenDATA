// Source-based slice around line 688
// Method: <com.google.common.net.InternetDomainName: int hashCode()>

    if (object instanceof InternetDomainName) {
      InternetDomainName that = (InternetDomainName) object;
      return this.name.equals(that.name);
    }

    return false;
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }
}
