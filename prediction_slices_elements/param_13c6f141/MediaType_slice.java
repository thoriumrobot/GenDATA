// Source-based slice around line 1138
// Method: <com.google.common.net.MediaType: void consumeSeparator(Tokenizer,char)>

        }
        parameters.put(attribute, value);
      }
      return create(type, subtype, parameters.build());
    } catch (IllegalStateException e) {
      throw new IllegalArgumentException("Could not parse '" + input + "'", e);
    }
  }

  private static void consumeSeparator(Tokenizer tokenizer, char c) {
    tokenizer.consumeTokenIfPresent(LINEAR_WHITE_SPACE);
    tokenizer.consumeCharacter(c);
    tokenizer.consumeTokenIfPresent(LINEAR_WHITE_SPACE);
  }

  private static final class Tokenizer {
    final String input;
    int position = 0;

    Tokenizer(String input) {
