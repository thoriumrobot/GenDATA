// Source-based slice around line 51
// Method: <com.google.common.collect.testing.UnhashableObject: String toString()>

  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException();
  }

  // needed because otherwise Object.toString() calls hashCode()
  @Override
  public String toString() {
    return "DontHashMe" + value;
  }

  @Override
  public int compareTo(UnhashableObject o) {
    return Integer.compare(this.value, o.value);
  }
}
