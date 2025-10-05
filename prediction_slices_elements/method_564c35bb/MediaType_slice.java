// Source-based slice around line 1236
// Method: <com.google.common.net.MediaType: String computeToString()>

    // racy single-check idiom, safe because String is immutable
    String result = toString;
    if (result == null) {
      result = computeToString();
      toString = result;
    }
    return result;
  }

  private String computeToString() {
    StringBuilder builder = new StringBuilder().append(type).append('/').append(subtype);
    if (!parameters.isEmpty()) {
      builder.append("; ");
      Multimap<String, String> quotedParameters =
          Multimaps.transformValues(
              parameters,
              (String value) ->
                  (TOKEN_MATCHER.matchesAllOf(value) && !value.isEmpty())
                      ? value
                      : escapeAndQuote(value));
