public class StringMethods {

    void testCodePointBefore(String s) {
        s.codePointBefore(0);
        if (s.length() > 0) {
            s.codePointBefore(s.length());
        }
    }
}
