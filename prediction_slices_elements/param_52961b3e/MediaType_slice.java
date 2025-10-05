// Source-based slice around line 1252
// Method: <com.google.common.net.MediaType: String escapeAndQuote(String)>

              (String value) ->
                  (TOKEN_MATCHER.matchesAllOf(value) && !value.isEmpty())
                      ? value
                      : escapeAndQuote(value));
      PARAMETER_JOINER.appendTo(builder, quotedParameters.entries());
    }
    return builder.toString();
  }

  private static String escapeAndQuote(String value) {
    StringBuilder escaped = new StringBuilder(value.length() + 16).append('"');
    for (int i = 0; i < value.length(); i++) {
      char ch = value.charAt(i);
      if (ch == '\r' || ch == '\\' || ch == '"') {
        escaped.append('\\');
      }
      escaped.append(ch);
    }
    return escaped.append('"').toString();
  }
