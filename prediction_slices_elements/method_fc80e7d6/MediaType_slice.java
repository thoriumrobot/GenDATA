// Source-based slice around line 1194
// Method: <com.google.common.net.MediaType: boolean equals(Object)>

      return input.charAt(position);
    }

    boolean hasMore() {
      return (position >= 0) && (position < input.length());
    }
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj == this) {
      return true;
    } else if (obj instanceof MediaType) {
      MediaType that = (MediaType) obj;
      return this.type.equals(that.type)
          && this.subtype.equals(that.subtype)
          // compare parameters regardless of order
          && this.parametersAsMap().equals(that.parametersAsMap());
    } else {
      return false;
