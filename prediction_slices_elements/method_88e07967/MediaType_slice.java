// Source-based slice around line 1209
// Method: <com.google.common.net.MediaType: int hashCode()>

          && this.subtype.equals(that.subtype)
          // compare parameters regardless of order
          && this.parametersAsMap().equals(that.parametersAsMap());
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    // racy single-check idiom
    int h = hashCode;
    if (h == 0) {
      h = hash(type, subtype, parametersAsMap());
      hashCode = h;
    }
    return h;
  }

  private static final MapJoiner PARAMETER_JOINER = Joiner.on("; ").withKeyValueSeparator("=");
