public class StringMethods {

    void testSubstring(String s) {
        s.substring(0);
        s.substring(0, 0);
        s.substring(s.length());
        s.substring(s.length(), s.length());
        s.substring(0, s.length());
        s.substring(1);
        s.substring(0, 1);
    }
}
